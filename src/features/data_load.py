"""Load datasets for bug report classification."""

import numpy as np
import pandas as pd

from ..config import AVAILABLE_DATASETS, DATASETS_DIR


def load_dataset(dataset_name: str) -> tuple[pd.Series, pd.Series]:
    """Load one dataset and return text inputs and labels.

    Args:
        dataset_name: Dataset stem from datasets/ (default: none)

    Returns:
        Tuple of X_text and y series

    Raises:
        FileNotFoundError: If dataset CSV is missing
        ValueError: If dataset name or required columns are invalid
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Invalid dataset '{dataset_name}'. Must be one of {AVAILABLE_DATASETS}"
        )

    dataset_path = DATASETS_DIR / f"{dataset_name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)

    required_columns = {"Title", "Body", "class"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_columns)}"
        )

    data["Title"] = data["Title"].fillna("")
    data["Body"] = data["Body"].fillna("")

    # Merge textual content together with a separator
    data["text"] = np.where(
        data["Body"].ne(""),  # if not empty,
        data["Title"] + ". " + data["Body"],  # concatenate
        data["Title"],  # else just use title
    )

    x_text = data["text"].astype(str)
    y = data["class"].astype(int)
    return x_text, y
