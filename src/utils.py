"""
src/utils.py
------------
Shared utility functions used across all model pipelines.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FNAME_RE = re.compile(
    r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg"
)

TAXONOMY_CLASSES = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_soundscape_filename(name: str) -> dict:
    """
    Parse a BirdCLEF 2026 soundscape filename into its components.

    Parameters
    ----------
    name : str
        Filename e.g. 'BC2026_Train_0001_S08_20250606_030007.ogg'

    Returns
    -------
    dict with keys: file_id, site, date, time_utc, hour_utc, month
    """
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": "unknown",
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def parse_label_string(x) -> list:
    """
    Parse a semicolon-separated label string into a list of label strings.

    Parameters
    ----------
    x : str or None
        e.g. 'barswa;comsan'

    Returns
    -------
    list of str
    """
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]


def union_labels(series) -> list:
    """
    Aggregate multiple label strings into a single sorted unique list.

    Parameters
    ----------
    series : pd.Series of str

    Returns
    -------
    sorted list of unique label strings
    """
    out = set()
    for x in series:
        for label in parse_label_string(x):
            out.add(label)
    return sorted(out)


def build_label_matrix(
    sc_df: pd.DataFrame,
    label_to_idx: dict,
    n_classes: int
) -> np.ndarray:
    """
    Build a multi-hot label matrix from a soundscape dataframe.

    Parameters
    ----------
    sc_df : pd.DataFrame
        Must have a 'label_list' column containing lists of label strings.
    label_to_idx : dict
        Mapping from label string to column index.
    n_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray of shape (len(sc_df), n_classes), dtype uint8
    """
    Y = np.zeros((len(sc_df), n_classes), dtype=np.uint8)
    for i, labels in enumerate(sc_df["label_list"]):
        for lbl in labels:
            if lbl in label_to_idx:
                Y[i, label_to_idx[lbl]] = 1
    return Y


# ---------------------------------------------------------------------------
# Taxonomy utilities
# ---------------------------------------------------------------------------

def build_taxon_masks(primary_labels: list, taxonomy_df: pd.DataFrame) -> dict:
    """
    Build per-taxon boolean index arrays for the 5 BirdCLEF taxonomic classes.

    Parameters
    ----------
    primary_labels : list of str
        Ordered list of species labels matching submission column order.
    taxonomy_df : pd.DataFrame
        Must have 'primary_label' and 'class_name' columns.

    Returns
    -------
    dict mapping class_name -> np.ndarray of int indices
    """
    label_to_taxon = dict(
        zip(
            taxonomy_df["primary_label"].astype(str),
            taxonomy_df["class_name"].astype(str),
        )
    )
    return {
        taxon: np.array(
            [i for i, l in enumerate(primary_labels)
             if label_to_taxon.get(l, "") == taxon]
        )
        for taxon in TAXONOMY_CLASSES
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged ROC-AUC, skipping classes with no positive labels.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_classes)
    y_score : np.ndarray of shape (n_samples, n_classes)

    Returns
    -------
    float — macro AUC over active classes only
    """
    from sklearn.metrics import roc_auc_score
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
    """
    import os
    import random
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
