from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(__file__).parent / "reference"


def load_matrix_reference(
    name: str, distance: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load input/output matrix pair for a dataset config."""
    d = REF_DIR / f"{name}_{distance}"
    inp = pd.read_csv(d / "input_matrix.csv", index_col=0)
    out = pd.read_csv(d / "output_matrix.csv", index_col=0)
    log2_file = d / "log2.txt"
    use_log2 = log2_file.exists() and log2_file.read_text().strip() == "true"
    return inp.values.astype(np.float64), out.values.astype(np.float64), use_log2


def load_bojkova_reference(
    distance: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load bojkova2020 input/output for a distance mode."""
    d = REF_DIR / f"bojkova2020_{distance}"
    inp_int = pd.read_csv(d / "input_intensities.csv")
    inp_meta = pd.read_csv(d / "input_metadata.csv")
    out_int = pd.read_csv(d / "output_intensities.csv")
    imp_rows = pd.read_csv(d / "imputed_rows.csv")
    return inp_int, inp_meta, out_int, imp_rows
