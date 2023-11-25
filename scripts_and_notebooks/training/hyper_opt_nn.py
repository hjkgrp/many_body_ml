import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pathlib import Path
from mbeml.featurization import (
    get_ligand_features,
    get_core_features,
)
from mbeml.nn_layers import CustomNormalization
from mbeml.nn_models_functional import (
    build_two_body_model,
    build_three_body_model,
    build_standard_racs_model,
    build_nn_ensemble,
)
from mbeml.constants import (
    LigandFeatures,
    ModelType,
    TargetProperty,
)
from mbeml.metrics import mean_absolute_error
from hyperopt import fmin, hp, tpe, STATUS_OK, space_eval
from functools import partial


def load_data_nn(data_dir: Path, features: LigandFeatures, target: TargetProperty):
    df_train = pd.read_csv(data_dir / "training_data.csv")
    df_val = pd.read_csv(data_dir / "validation_data.csv")

    ligs_train = get_ligand_features(df_train, features, remove_trivial=True)
    ligs_val = get_ligand_features(df_val, features, remove_trivial=True)
    if features is LigandFeatures.STANDARD_RACS:
        racs_norm = np.max(np.abs(ligs_train), axis=0)
        racs_norm[racs_norm < 1e-6] = 1.0
    elif features is LigandFeatures.LIGAND_RACS:
        ligs_train = ligs_train.reshape((len(df_train), 6, -1))
        ligs_val = ligs_val.reshape((len(df_val), 6, -1))
        racs_norm = np.max(
            np.abs(ligs_train),
            axis=(0, 1),
        )
        racs_norm[racs_norm < 1e-6] = 1.0
        racs_norm = np.tile(
            racs_norm,
            (6, 1),
        )
    else:
        raise NotImplementedError(f"Unknown features {features.name}")

    core_train = get_core_features(df_train)
    core_val = get_core_features(df_val)

    X_train = {"core": core_train, "ligands": ligs_train}
    X_val = {"core": core_val, "ligands": ligs_val}

    # Even though this was intended for a normalization to a standard normal
    # distribution the math for the MaxAbs scaling is essentially the same:
    ligands_norm = CustomNormalization(
        mean=np.zeros_like(racs_norm), std=racs_norm, name="ligands_normalization"
    )

    mean = np.mean(df_train[target.full_name()], axis=0)
    std = np.std(df_train[target.full_name()], axis=0)
    target_norm = CustomNormalization(
        mean=mean, std=std, name=f"output_normalization_{target}"
    )

    y_train = df_train[target.full_name()].values.reshape(len(df_train), -1)
    y_val = df_val[target.full_name()].values.reshape(len(df_val), -1)

    return X_train, y_train, X_val, y_val, ligands_norm, target_norm


def build_model(
    model_type,
    hidden_units=[32, 32],
    dropout_rate=0.2,
    l2=0.01,
    ligands_norm=None,
    output_norm=None,
    spin_dependent=False,
    num_ligand_features=None,
    num_outputs=1,
    **model_kws,
):
    if model_type is ModelType.STANDARD_RACS:
        return build_standard_racs_model(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            l2=l2,
            racs_norm=ligands_norm,
            output_norm=output_norm,
            spin_dependent=spin_dependent,
            num_ligand_features=num_ligand_features,
            num_outputs=num_outputs,
        )
    elif model_type is ModelType.TWO_BODY:
        return build_two_body_model(
            two_body_units=hidden_units,
            dropout_rate=dropout_rate,
            l2=l2,
            racs_norm=ligands_norm,
            output_norm=output_norm,
            spin_dependent=spin_dependent,
            num_ligand_features=num_ligand_features,
            num_outputs=num_outputs,
        )
    elif model_type is ModelType.THREE_BODY:
        return build_three_body_model(
            two_body_units=hidden_units,
            three_body_units=model_kws.get("three_body_units", hidden_units),
            dropout_rate=dropout_rate,
            l2=l2,
            racs_norm=ligands_norm,
            output_norm=output_norm,
            spin_dependent=spin_dependent,
            num_ligand_features=num_ligand_features,
            num_outputs=num_outputs,
            masked=False,
            features_sym=True,
            two_body_terms=True,
        )
    else:
        raise NotImplementedError()


def train_model(
    params, model_type, X_train, y_train, X_val, y_val, ligands_norm, target_norm
):
    model = build_model(
        model_type,
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout"],
        l2=10 ** params["lambda"],
        spin_dependent=y_train.shape[-1] == 2,
        ligands_norm=ligands_norm,
        output_norm=target_norm,
        num_ligand_features=X_train["ligands"].shape[-1],
        num_outputs=4 if y_train.shape[-1] == 4 else 1,
    )
    # Build model by calling it
    _ = model(X_train)
    model.compile(
        # Using the legacy version for better comparison with previously
        # trained models
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.MeanAbsoluteError(),
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, restore_best_weights=True
    )

    if model_type is ModelType.THREE_BODY:
        # The training of the three-body model takes place in three phases
        # as described in the manuscript.
        # Phase I: Only training the two-body terms
        # Save the randomly initialized three-body weights for later and set
        # them zeros and untrainable for now
        three_body_weights = {}
        for layer in model.layers:
            if "three_body_nn" in layer.name:
                weights = layer.get_weights()
                # Save
                three_body_weights[layer.name] = weights
                # Set zero
                zero_weights = []
                for w in weights:
                    zero_weights.append(np.zeros_like(w))
                layer.set_weights(zero_weights)
                # Set untrainable
                layer.trainable = False

        model.fit(
            X_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=5000,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[callback],
        )

        # Phase II: Only training the three-body term
        # Restore three-body initial values and freeze two-body terms
        for layer in model.layers:
            if "three_body_nn" in layer.name:
                # Restore weights
                layer.set_weights(three_body_weights[layer.name])
                # Set trainable
                layer.trainable = True
            elif "two_body_nn" in layer.name:
                # Set untrainable
                layer.trainable = False

        model.fit(
            X_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=5000,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[callback],
        )

        # Phase III: Train all terms
        # Unfreeze the two-body terms
        for layer in model.layers:
            if "body_nn" in layer.name:
                # Set trainable
                layer.trainable = True

        model.fit(
            X_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=5000,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[callback],
        )
    else:
        # Just fit a regular model
        model.fit(
            X_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=5000,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[callback],
        )
    return model


def evaluate_single_point(
    params,
    model_type,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    ligands_norm=None,
    target_norm=None,
):
    model = train_model(
        params, model_type, X_train, y_train, X_val, y_val, ligands_norm, target_norm
    )
    val_mae = model.evaluate(X_val, y_val, verbose=0)[-1]
    tf.keras.backend.clear_session()
    return {"loss": val_mae, "status": STATUS_OK}


def main(model_type: ModelType, target: TargetProperty, random_seed=0):
    data_dir = Path("../../data/")
    models_dir = Path("../../models")

    (X_train, y_train, X_val, y_val, ligands_norm, target_norm) = load_data_nn(
        data_dir, model_type.ligand_features(), target
    )

    # Define the hyperparameter space.
    space = {
        "hidden_units": hp.choice("hidden_units", [[64, 64], [128, 128], [256, 256]]),
        "lambda": hp.quniform("lambda", -5, 0, 1),
        "dropout": hp.quniform("dropout", 0.0, 0.6, 0.1),
        "batch_size": hp.choice("batch_size", [64, 128, 256]),
    }

    objective_func = partial(
        evaluate_single_point,
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        ligands_norm=ligands_norm,
        target_norm=target_norm,
    )

    tf.keras.utils.set_random_seed(random_seed)
    tf.config.experimental.enable_op_determinism()

    max_evals = 100
    best_params = fmin(
        objective_func,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=np.random.default_rng(random_seed),
    )

    final_params = space_eval(space, best_params)
    print("best_params: ", final_params)

    # 10 training run using the optimal hyperparams
    n_runs = 10
    models = []
    for i in range(n_runs):
        model = train_model(
            final_params,
            model_type,
            X_train,
            y_train,
            X_val,
            y_val,
            ligands_norm,
            target_norm,
        )
        print(
            f"Evaluating run {i} val MAE: "
            f"{model.evaluate(X_val, y_val, verbose=0)[-1]:.2f}"
        )
        # Rename because model names need to be unique
        model._name = f"{model.name}_{i}"
        models.append(model)
        tf.keras.backend.clear_session()

    # Assemble into an ensemble model and save the ensemble
    ensemble = build_nn_ensemble(models, return_std=True)
    # Build the model by calling it
    y_mean, _ = ensemble.predict(X_val)
    print(f"Ensemble validation MAE: {mean_absolute_error(y_val, y_mean):.2f} kcal/mol")
    ensemble.compile()
    ensemble.save(models_dir / target.name.lower() / f"nn_{model_type.name.lower()}")

    # Write the optimal hyperparameters to a json file
    with open(
        models_dir
        / target.name.lower()
        / f"nn_{model_type.name.lower()}_hyperparams.json",
        "w",
    ) as file:
        json.dump(final_params, file)


if __name__ == "__main__":
    # See the definition of the two enums in mbeml.constants for possible values
    model_type = ModelType.THREE_BODY
    target = TargetProperty.SSE
    main(model_type, target)
