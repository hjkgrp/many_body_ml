import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict
from mbeml.featurization import (
    get_ligand_features,
    get_core_features,
)
from mbeml.kernels import Masking, TwoBodyKernel, ThreeBodyKernel
from mbeml.constants import (
    LigandFeatures,
    ModelType,
    TargetProperty,
    cis_pairs,
    trans_pairs,
)
from sklearn.preprocessing import MaxAbsScaler
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, hp, tpe, STATUS_OK, space_eval
from functools import partial


# Enums used to specify which model type to train on which target values


def load_data_krr(data_dir: Path, features: LigandFeatures, target: TargetProperty):
    df_train = pd.read_csv(data_dir / "training_data.csv")
    df_val = pd.read_csv(data_dir / "validation_data.csv")

    ligs_train = get_ligand_features(df_train, features)
    ligs_val = get_core_features(df_val, features)
    if features is LigandFeatures.STANDARD_RACS:
        racs_norm = np.max(np.abs(ligs_train), axis=0)
        racs_norm[racs_norm < 1e-6] = 1.0
    elif features is LigandFeatures.LIGAND_RACS:
        racs_norm = np.max(
            np.abs(ligs_train.reshape((len(df_train), 6, 33))),
            axis=(0, 1),
        )
        racs_norm[racs_norm < 1e-6] = 1.0
        racs_norm = np.tile(racs_norm, 6)
    else:
        raise NotImplementedError(f"Unknown features {features.name}")

    core_train = get_core_features(df_train)
    core_val = get_core_features(df_val)

    X_train = np.concatenate([core_train, ligs_train], axis=-1)
    X_val = np.concatenate([core_val, ligs_val], axis=-1)

    # Build MaxAbsScaler that can be used as part of a sklearn.Pipeline
    input_scaler = MaxAbsScaler()
    # Manually set all the attributes:
    input_scaler.n_samples_seen_ = X_train.shape[0]
    input_scaler.n_features_in_ = X_train.shape[1]
    # The input scaling for the core features is just ones
    scale = np.concatenate([np.ones(core_train.shape[1]), racs_norm])
    input_scaler.max_abs_ = input_scaler.scale_ = scale

    y_train = df_train[target.full_name()].values.reshape(len(df_train), -1)
    y_val = df_val[target.full_name()].values.reshape(len(df_val), -1)

    return X_train, y_train, X_val, y_val, input_scaler


def get_kernel_by_name(name: str) -> Kernel:
    if name == "RBF":
        return RBF
    elif name == "Matern":
        return Matern
    raise NotImplementedError(f"Unknown kernel {name}")


def build_model(
    model_type: ModelType, params: Dict, n_features: int, n_core_features: int
):
    # The one-body term is common to all models
    core_mask = np.zeros(n_features, dtype=bool)
    core_mask[:n_core_features] = True
    kernel = Masking(core_mask, DotProduct(sigma_0=0.0, sigma_0_bounds="fixed"))

    if model_type is ModelType.STANDARD_RACS:
        kernel += get_kernel_by_name(params["kernel"])(
            length_scale=params["length_scale"],
            length_scale_bounds="fixed",
        )
    elif model_type in [ModelType.TWO_BODY, ModelType.THREE_BODY]:
        kernel += TwoBodyKernel(
            get_kernel_by_name(params["two_body_kernel"])(
                length_scale=params["two_body_length_scale"],
                length_scale_bounds="fixed",
            ),
            n_core_features=n_core_features,
        )
        if model_type is ModelType.THREE_BODY:
            kernel += ThreeBodyKernel(
                get_kernel_by_name(params["three_body_cis_kernel"])(
                    length_scale=params["three_body_cis_length_scale"],
                    length_scale_bounds="fixed",
                ),
                pairs=cis_pairs,
            )
            kernel += ThreeBodyKernel(
                get_kernel_by_name(params["three_body_trans_kernel"])(
                    length_scale=params["three_body_trans_length_scale"],
                    length_scale_bounds="fixed",
                ),
                pairs=trans_pairs,
            )
    else:
        raise NotImplementedError(f"Unknown model type {model_type}")
    model = GaussianProcessRegressor(
        kernel=kernel, alpha=params["l2"], normalize_y=True
    )
    return model


def train_model(model_type: ModelType, params: Dict, X_train, y_train, input_scaler):
    n_features = len(input_scaler.scale_)
    model = build_model(model_type, params, n_features=n_features, n_core_features=7)
    model.fit(input_scaler.transform(X_train), y_train)
    # This weird workaround of not fitting the input scaler with the
    # rest of the model is necessary because the ligand RACs are spread
    # out over several "columns" in X_train but need to be normalized
    # all together (which happens in load_data_krr)
    pipe = Pipeline([("input_scaler", input_scaler), ("gpr", model)])
    return pipe


def evaluate_single_point(
    params,
    model_type=ModelType.TWO_BODY,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    input_scaler=None,
):
    model = train_model(model_type, params, X_train, y_train, input_scaler)
    val_mae = mean_absolute_error(model.predict(X_val), y_val)
    return {"loss": val_mae, "status": STATUS_OK}


def main(model_type: ModelType, target: TargetProperty):
    data_dir = Path("../../data/")
    models_dir = Path("../../models")

    (X_train, y_train, X_val, y_val, input_scaler) = load_data_krr(
        data_dir, model_type.ligand_features(), target
    )

    # Define the hyperparameter space. This first part is common to all kernel models
    space = {
        "l2": hp.loguniform("l2", np.log(1e-3), np.log(1e3)),
    }
    if model_type is ModelType.STANDARD_RACS:
        space.update(
            {
                "kernel": hp.choice("kernel", ["RBF", "Matern"]),
                "length_scale": hp.loguniform(
                    "length_scale", np.log(1e-3), np.log(1e3)
                ),
            }
        )
    elif model_type is ModelType.TWO_BODY:
        space.update(
            {
                "two_body_kernel": hp.choice("two_body_kernel", ["RBF", "Matern"]),
                "two_body_length_scale": hp.loguniform(
                    "two_body_length_scale", np.log(1e-3), np.log(1e3)
                ),
            }
        )
    elif model_type is ModelType.THREE_BODY:
        # The two body parameters are read from run for the TWO_BODY model
        with open(
            models_dir / target.name.lower() / "krr_two_body_hyperparams.json",
            "r",
        ) as file:
            two_body_params = json.load(file)
        space.update(
            {
                "two_body_kernel": two_body_params["two_body_kernel"],
                "two_body_length_scale": two_body_params["two_body_length_scale"],
            }
        )
        # Three body hyperspace
        space.update(
            {
                "three_body_cis_kernel": hp.choice(
                    "three_body_cis_kernel", ["RBF", "Matern"]
                ),
                "three_body_cis_length_scale": hp.loguniform(
                    "three_body_cis_length_scale", np.log(1e-3), np.log(1e3)
                ),
                "three_body_trans_kernel": hp.choice(
                    "three_body_trans_kernel", ["RBF", "Matern"]
                ),
                "three_body_trans_length_scale": hp.loguniform(
                    "three_body_trans_length_scale", np.log(1e-3), np.log(1e3)
                ),
            }
        )

    # Construct an objective function
    objective_func = partial(
        evaluate_single_point,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_scaler=input_scaler,
    )

    max_evals = 100
    # Perform the actual optimization
    best_params = fmin(
        objective_func,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
    )

    final_params = space_eval(space, best_params)
    print("best_params: ", final_params)

    # (Re)train a model using these parameters
    model = train_model(model_type, final_params, X_train, y_train, input_scaler)

    # Save the model
    with open(
        models_dir / target.name.lower() / f"krr_{model_type.name.lower()}.pkl", "wb"
    ) as fout:
        pickle.dump(model, fout)

    # Write the optimal hyperparameters to a json file
    with open(
        models_dir
        / target.name.lower()
        / f"krr_{model_type.name.lower()}_hyperparams.json",
        "w",
    ) as file:
        json.dump(final_params, file)


if __name__ == "__main__":
    # See the definition of the two enums in mbeml.constants for possible values
    model_type = ModelType.THREE_BODY
    target = TargetProperty.ORBITALS
    main(model_type, target)
