import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.mark.parametrize(
    "df_name",
    [
        "training_data.csv",
        "validation_data.csv",
        "composition_test_data.csv",
        "ligand_test_data.csv",
    ],
)
def test_gap_calculation(df_name, atol=1e-6):
    base_path = Path("data")
    df = pd.read_csv(base_path / df_name)

    for spin in ["ls", "hs"]:
        np.testing.assert_allclose(
            df[f"energetic_gap_{spin}_eV"].values,
            df[f"energetic_lumo_{spin}_eV"].values
            - df[f"energetic_homo_{spin}_eV"].values,
            atol=atol,
        )
