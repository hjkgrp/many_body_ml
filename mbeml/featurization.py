import numpy as np
from typing import Optional, List, Tuple


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
