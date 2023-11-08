import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


target_long_names = {
    "sse": "spin_splitting_kcal/mol",
    "homo": ["energetic_homo_ls_eV", "energetic_homo_hs_eV"],
    "lumo": ["energetic_lumo_ls_eV", "energetic_lumo_hs_eV"],
    "gap": ["energetic_gap_ls_eV", "energetic_gap_hs_eV"],
    "orbitals": [
        "energetic_homo_ls_eV",
        "energetic_homo_hs_eV",
        "energetic_lumo_ls_eV",
        "energetic_lumo_hs_eV",
    ],
}


def generate_standard_racs_names(
    depth: int = 3,
    properties: Optional[List[str]] = None,
    start_scope: Optional[List[Tuple[str, str]]] = None,
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
    return names


def generate_ligand_racs_names():
    names = []
    for i in range(1, 7):
        names.extend([f"lig{i}_charge"] + [f"lig{i}_racs{j:02d}" for j in range(32)])
    return names


def get_racs_features(df: pd.DataFrame, features: str):
    if features == "standard_racs":
        racs_names = generate_standard_racs_names()
    elif features == "ligand_racs":
        racs_names = generate_ligand_racs_names()
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
