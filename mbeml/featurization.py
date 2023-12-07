import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from mbeml.constants import LigandFeatures, TargetProperty


def generate_standard_racs_names(
    depth: int = 3,
    properties: Optional[List[str]] = None,
    start_scope: Optional[List[Tuple[str, str]]] = None,
    remove_trivial: bool = False,
) -> List[str]:
    if properties is None:
        properties = ["chi", "Z", "I", "T", "S"]

    if start_scope is None:
        start_scope = [
            ("f", "all"),
            ("f", "ax"),
            ("f", "eq"),
            ("lc", "ax"),
            ("lc", "eq"),
            ("D_lc", "ax"),
            ("D_lc", "eq"),
            ("mc", "all"),
            ("D_mc", "all"),
        ]
    names = [
        "misc-dent-ax",
        "misc-charge-ax",
        "misc-dent-eq",
        "misc-charge-eq",
    ] + [
        f"{start}-{prop}-{d}-{scope}"
        for start, scope in start_scope
        for prop in properties
        for d in range(0, depth + 1)
    ]
    if remove_trivial:
        for start, scope in start_scope:
            if start.startswith("D"):
                # Remove difference RACs for depth 0
                for prop in properties:
                    names.remove(f"{start}-{prop}-0-{scope}")
                # Remove difference RACs for property i
                for d in range(1, depth + 1):
                    names.remove(f"{start}-I-{d}-{scope}")
        # There are 5 more trivial RACs that are not part of RACs155:
        names.remove("lc-I-0-ax")
        names.remove("lc-I-0-eq")
        names.remove("mc-I-0-all")
        names.remove("mc-I-1-all")
        names.remove("mc-T-0-all")
    return names


def generate_ligand_racs_names(
    depth: int = 3,
    properties: Optional[List[str]] = None,
    remove_trivial: bool = False,
) -> List[str]:
    names = []
    # Unfortunately the ordering of RACS here is different
    # to standard RACs because of different implementations in
    # molSimplify vs tmc_tools
    if properties is None:
        properties = ["Z", "chi", "T", "I", "S"]
    for i in range(1, 7):
        # Ligand_charge:
        names.append(f"lig{i}_charge")
        # Product RACs
        for d in range(depth + 1):
            for prop in properties:
                # P for product
                names.append(f"lig{i}_P_{prop}_{d}")
        # Difference RACs
        # Skip depth 0
        for d in range(1, depth + 1):
            for prop in properties:
                # skip property I:
                if prop != "I":
                    # D for difference
                    names.append(f"lig{i}_D_{prop}_{d}")
    if remove_trivial:
        for i in range(1, 7):
            # remove depth 0 identity RAC
            names.remove(f"lig{i}_P_I_0")
    return names


def generate_ligand_racs_names_old():
    names = []
    for i in range(1, 7):
        names.extend([f"lig{i}_charge"] + [f"lig{i}_racs{j:02d}" for j in range(32)])
    return names


def get_ligand_features(df: pd.DataFrame, features: LigandFeatures, **kwargs):
    if features is LigandFeatures.STANDARD_RACS:
        racs_names = generate_standard_racs_names(**kwargs)
    elif features is LigandFeatures.LIGAND_RACS:
        racs_names = generate_ligand_racs_names(**kwargs)
    else:
        raise NotImplementedError(f"Unknown features {features}")
    return df[racs_names].values


def get_core_features(df):
    core_encoding = {
        ("co", 2): [1, 0, 0, 0, 0, 0, 1],
        ("co", 3): [0, 1, 0, 0, 0, 1, 0],
        ("fe", 2): [1, 0, 0, 0, 0, 1, 0],
        ("fe", 3): [0, 1, 0, 0, 1, 0, 0],
        ("mn", 2): [1, 0, 0, 0, 1, 0, 0],
        ("mn", 3): [0, 1, 0, 1, 0, 0, 0],
        ("cr", 2): [1, 0, 0, 1, 0, 0, 0],
        ("cr", 3): [0, 1, 1, 0, 0, 0, 0],
    }

    return np.array(
        [core_encoding[(metal, ox)] for metal, ox in df[["metal", "ox"]].values]
    )


def data_prep(
    df: pd.DataFrame,
    features: LigandFeatures,
    target: TargetProperty,
    is_nn: bool = False,
):
    y = df[target.full_name()].values.reshape(len(df), -1)

    core_features = get_core_features(df)
    racs_features = get_ligand_features(df, features=features, remove_trivial=True)
    if is_nn:
        if features is LigandFeatures.LIGAND_RACS:
            racs_features = racs_features.reshape(len(df), 6, -1)
        X = {"core": core_features, "ligands": racs_features}
    else:
        X = np.concatenate([core_features, racs_features], axis=-1)
    return X, y
