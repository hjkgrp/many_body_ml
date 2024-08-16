import operator
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from itertools import permutations
from molSimplify.Classes.mol2D import Mol2D
from molSimplify.Informatics.graph_racs import atom_centered_AC
from mbeml.constants import (
    LigandFeatures,
    TargetProperty,
    atomic_numbers,
    covalent_radii,
    electronegativity,
)


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
    # to standard RACs because of different implementations for
    # mol2D versus mol3D
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


def get_ligand_features(df: pd.DataFrame, features: LigandFeatures, **kwargs):
    if features is LigandFeatures.STANDARD_RACS:
        racs_names = generate_standard_racs_names(**kwargs)
    elif features is LigandFeatures.LIGAND_RACS:
        racs_names = generate_ligand_racs_names(**kwargs)
    else:
        raise NotImplementedError(f"Unknown features {features}")
    return df[racs_names].values


def sort_connecting_atoms(atoms, c_atoms, cis_threshold=22.5):
    assert len(c_atoms) == 6
    for c0, c1, c2, c3, c4, c5 in permutations(c_atoms, 6):
        angles = atoms.get_angles(
            [
                [c0, 0, c1],
                [c1, 0, c2],
                [c2, 0, c3],
                [c3, 0, c0],
                [c0, 0, c4],
                [c1, 0, c4],
                [c2, 0, c4],
                [c3, 0, c4],
                [c0, 0, c5],
                [c1, 0, c5],
                [c2, 0, c5],
                [c3, 0, c5],
            ]
        )
        if all(abs(angles - 90.0) < cis_threshold):
            return [c0, c1, c2, c3, c4, c5]
    raise ValueError("Cannot find octahedral arangement of connecting atoms")


def featurize_single_ligand(lig: Mol2D, connecting_atom: int, depth: int = 3):
    # 5 properties x (depth + 1) for product racs
    # + 4 properties x depth for difference racs
    rac_vector = np.zeros(5 * (depth + 1) + 4 * depth)
    rac_vector[: 5 * (depth + 1)] = atom_centered_AC(
        lig,
        connecting_atom,
        depth=depth,
        property_fun=ligand_racs_property_vector,
    ).T.flatten()  # Product racs
    # Difference racs, drop depth 0 and identity property
    rac_vector[5 * (depth + 1) :] = (
        atom_centered_AC(
            lig,
            connecting_atom,
            depth=depth,
            property_fun=ligand_racs_property_vector,
            operation=operator.sub,
        )
        .T[1:, [True, True, True, False, True]]
        .flatten()
    )
    return rac_vector


def ligand_racs_property_vector(mol: Mol2D, node: int) -> np.ndarray:
    """Calculates the property vector for a given node (atom) in a molecular graph.

    Parameters
    ----------
    mol : Mol2D
        molecular graph
    node : int
        index of the node

    Returns
    -------
    np.ndarray
        property vector of the node
    """
    output = np.zeros(5)
    symbol = mol.nodes[node]["symbol"]
    Z = atomic_numbers[symbol]
    # property (i): nuclear charge Z
    output[0] = Z
    # property (ii): Pauling electronegativity chi
    output[1] = electronegativity[Z]
    # property (iii): topology T, coordination number
    output[2] = len(list(mol.neighbors(node)))
    # property (iv): identity
    output[3] = 1.0
    # property (v): covalent radius S
    output[4] = covalent_radii[Z]
    return output


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
