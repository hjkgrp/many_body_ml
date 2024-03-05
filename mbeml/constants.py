from enum import Enum, auto
from typing import List

cis_pairs = [
    (0, 1),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
]
trans_pairs = [(0, 2), (1, 3), (4, 5)]

unique_cores = ["cr3", "cr2", "mn3", "mn2", "fe3", "fe2", "co3", "co2"]
roman_numerals = {"2": "II", "3": "III"}

target_full_names = {
    "sse": ["spin_splitting_kcal/mol"],
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
# For backward compatibilty
target_long_names = target_full_names


class LigandFeatures(Enum):
    STANDARD_RACS = auto()
    LIGAND_RACS = auto()


class ModelType(Enum):
    STANDARD_RACS = auto()
    TWO_BODY = auto()
    THREE_BODY = auto()

    def ligand_features(self) -> LigandFeatures:
        if self is ModelType.STANDARD_RACS:
            return LigandFeatures.STANDARD_RACS
        return LigandFeatures.LIGAND_RACS


class TargetProperty(Enum):
    SSE = auto()
    ORBITALS = auto()
    HOMO = auto()
    LUMO = auto()
    GAP = auto()

    def full_name(self) -> List[str]:
        return target_full_names[self.name.lower()]
