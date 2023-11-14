import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from mbeml.constants import LigandFeatures, TargetProperty
from mbeml.featurization import get_core_features, get_ligand_features
from mbeml.metrics import mean_absolute_error, r2_score, mean_negative_log_likelihood


def data_prep(
    df: pd.DataFrame,
    features: LigandFeatures,
    target: TargetProperty,
    is_nn: bool = False,
):
    y = df[target.full_name()].values.reshape(len(df), -1)

    core_features = get_core_features(df)
    racs_features = get_ligand_features(df, features=features)
    if is_nn:
        if features is LigandFeatures.LIGAND_RACS:
            racs_features = racs_features.reshape(len(df), 6, 33)
        X = {"core": core_features, "ligands": racs_features}
    else:
        X = np.concatenate([core_features, racs_features], axis=-1)
    return X, y


def main():
    data_dir = Path("../../data/")

    data_sets = {
        "train": pd.read_csv(data_dir / "training_data.csv"),
        "validation": pd.read_csv(data_dir / "validation_data.csv"),
        "composition_test": pd.read_csv(data_dir / "composition_test_data.csv"),
        "ligand_test": pd.read_csv(data_dir / "ligand_test_data.csv"),
    }

    model_dir = Path("../../models/")

    @dataclass
    class Experiment:
        name: str
        features: LigandFeatures
        target: TargetProperty = TargetProperty.SSE
        is_nn: bool = False
        MAE_train: float = 0.0
        MAE_validation: float = 0.0
        MAE_composition_test: float = 0.0
        MAE_ligand_test: float = 0.0
        R2_train: float = 0.0
        R2_validation: float = 0.0
        R2_composition_test: float = 0.0
        R2_ligand_test: float = 0.0
        MNLL_train: float = 0.0
        MNLL_validation: float = 0.0
        MNLL_composition_test: float = 0.0
        MNLL_ligand_test: float = 0.0

    experiments = [
        Experiment(name="krr_standard_racs", features=LigandFeatures.STANDARD_RACS),
        Experiment(name="krr_two_body", features=LigandFeatures.LIGAND_RACS),
        Experiment(name="krr_three_body", features=LigandFeatures.LIGAND_RACS),
        Experiment(
            name="nn_standard_racs", features=LigandFeatures.STANDARD_RACS, is_nn=True
        ),
        Experiment(name="nn_two_body", features=LigandFeatures.LIGAND_RACS, is_nn=True),
        Experiment(
            name="nn_three_body", features=LigandFeatures.LIGAND_RACS, is_nn=True
        ),
    ]

    for experiment in experiments:
        for df_name, data_set in data_sets.items():
            X, y = data_prep(
                data_set, experiment.features, experiment.target, experiment.is_nn
            )
            if experiment.is_nn:
                model = tf.keras.models.load_model(
                    model_dir / experiment.target.name.lower() / experiment.name
                )
                y_mean, y_std = model.predict(X, verbose=0)
            else:
                with open(
                    model_dir
                    / experiment.target.name.lower()
                    / f"{experiment.name}.pkl",
                    "rb",
                ) as fin:
                    model = pickle.load(fin)
                y_mean, y_std = model.predict(X, return_std=True)
            # Evaluate MAE, and MNLL and save in experiment object
            setattr(experiment, f"MAE_{df_name}", mean_absolute_error(y, y_mean))
            setattr(experiment, f"R2_{df_name}", r2_score(y, y_mean))
            setattr(
                experiment,
                f"MNLL_{df_name}",
                mean_negative_log_likelihood(y, y_mean, y_std),
            )

    maes = pd.DataFrame(
        experiments,
        columns=[
            "name",
            "MAE_train",
            "MAE_validation",
            "MAE_composition_test",
            "MAE_ligand_test",
        ],
    )
    print(maes.round(2))

    r2s = pd.DataFrame(
        experiments,
        columns=[
            "name",
            "R2_train",
            "R2_validation",
            "R2_composition_test",
            "R2_ligand_test",
        ],
    )
    print(r2s.round(2))

    mnll = pd.DataFrame(
        experiments,
        columns=[
            "name",
            "MNLL_train",
            "MNLL_validation",
            "MNLL_composition_test",
            "MNLL_ligand_test",
        ],
    )
    print(mnll.round(2))


if __name__ == "__main__":
    main()
