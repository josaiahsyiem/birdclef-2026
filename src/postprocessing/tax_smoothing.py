"""
src/postprocessing/tax_smoothing.py
------------------------------------
Taxonomy-aware probability smoothing applied after ensemble blending.

Key insight: species sharing the same genus or taxonomic class tend to
co-occur in the same soundscape. Smoothing probabilities toward the
genus/class mean improves calibration and leaderboard score.

This post-processing step was identified as a shared pattern across all
public notebooks scoring 0.950+ on the BirdCLEF+ 2026 leaderboard.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def apply_tax_smoothing(
    submission: pd.DataFrame,
    taxonomy_path: str | Path,
    genus_alpha: float = 0.15,
    class_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply taxonomy-aware smoothing to a submission dataframe.

    For each group of species sharing the same genus, pulls each species
    probability slightly toward the group mean. Repeated at the broader
    taxonomic class level with a lighter alpha.

    Parameters
    ----------
    submission : pd.DataFrame
        Submission dataframe with species columns (no row_id column).
        Shape: (n_windows, n_classes)
    taxonomy_path : str or Path
        Path to taxonomy.csv from the competition data.
    genus_alpha : float
        Smoothing strength at genus level. 0 = no smoothing, 1 = full mean.
        Competition value: 0.15
    class_alpha : float
        Smoothing strength at class level.
        Competition value: 0.05

    Returns
    -------
    pd.DataFrame — smoothed submission, same shape and columns as input
    """
    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        print(
            f"[TAX_SMOOTHING] taxonomy.csv not found at {taxonomy_path} — skipping")
        return submission

    tax = pd.read_csv(taxonomy_path)

    # Build species -> genus and species -> class mappings
    species_to_genus = {}
    species_to_class = {}
    for _, row in tax.iterrows():
        label = str(row["primary_label"])
        sci = str(row.get("scientific_name", ""))
        cls = str(row.get("class_name", ""))
        genus = sci.split(" ")[0] if " " in sci else sci
        species_to_genus[label] = genus
        species_to_class[label] = cls

    cols = list(submission.columns)

    # Group species by genus and class
    genus_groups = {}
    class_groups = {}
    for col in cols:
        genus = species_to_genus.get(col, col)
        cls = species_to_class.get(col, "")
        genus_groups.setdefault(genus, []).append(col)
        if cls:
            class_groups.setdefault(cls, []).append(col)

    # Only smooth groups with more than one member
    multi_genus = {g: m for g, m in genus_groups.items() if len(m) > 1}
    multi_class = {c: m for c, m in class_groups.items() if len(m) > 1}

    print(f"[TAX_SMOOTHING] genus groups: {len(multi_genus)}, "
          f"class groups: {len(multi_class)}")

    probs = submission.values.astype(np.float32).copy()

    # Genus-level smoothing
    for genus, members in multi_genus.items():
        idx = [cols.index(m) for m in members]
        mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1 - genus_alpha) * probs[:, idx] + genus_alpha * mean

    # Class-level smoothing
    for cls, members in multi_class.items():
        idx = [cols.index(m) for m in members]
        mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1 - class_alpha) * probs[:, idx] + class_alpha * mean

    probs = np.clip(probs, 0.0, 1.0)

    smoothed = pd.DataFrame(probs, index=submission.index, columns=cols)
    print(f"[TAX_SMOOTHING] done. mean={smoothed.values.mean():.4f}")
    return smoothed
