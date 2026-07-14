"""
src/ensemble.py
---------------
Ensemble blending logic for combining predictions from multiple models.

The winning blend uses division_attention weighting across three models:
  - Model_21 (ProtoSSM):          weight 0.014
  - Model_52 (ProtoSSM sub):      weight 0.021
  - Model_74 (Karnakbayev):       weight 0.965

Followed by TAX_SMOOTHING post-processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_submission(path: str | Path) -> pd.DataFrame:
    """
    Load a submission CSV and return it indexed by row_id.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pd.DataFrame indexed by row_id, columns = species labels
    """
    df = pd.read_csv(path, index_col="row_id")
    return df.astype(np.float32)


def division_attention_blend(
    submissions: list[pd.DataFrame],
    weights: list[float],
) -> pd.DataFrame:
    """
    Blend multiple submission dataframes using division attention weighting.

    Division attention normalizes weights so they sum to 1, then computes
    a weighted average. More stable than simple linear blend when one model
    dominates (e.g. Model_74 at 96.5% weight).

    Parameters
    ----------
    submissions : list of pd.DataFrame
        Each dataframe indexed by row_id, columns = species labels.
        All must have identical index and columns.
    weights : list of float
        Raw weights for each submission. Will be normalized to sum to 1.

    Returns
    -------
    pd.DataFrame — blended submission, same shape as inputs
    """
    assert len(submissions) == len(weights), \
        "Number of submissions must match number of weights"

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    print(f"[ENSEMBLE] Blending {len(submissions)} models")
    for i, w in enumerate(weights):
        print(f"  Model_{i+1}: weight={w:.4f}")

    blended = sum(
        w * df.values
        for w, df in zip(weights, submissions)
    )

    return pd.DataFrame(
        blended,
        index=submissions[0].index,
        columns=submissions[0].columns,
    )


def blend_from_config(config: dict, output_dir: str | Path) -> pd.DataFrame:
    """
    Run the full ensemble blend as specified in ensemble_config.yaml.

    Parameters
    ----------
    config : dict
        Parsed YAML config (ensemble section).
    output_dir : str or Path
        Directory where individual model submission CSVs are saved.

    Returns
    -------
    pd.DataFrame — final blended submission
    """
    from src.postprocessing.tax_smoothing import apply_tax_smoothing

    output_dir = Path(output_dir)
    models = config["ensemble"]["models"]

    submissions = []
    weights = []

    for model in models:
        path = output_dir / model["submission_file"]
        if not path.exists():
            raise FileNotFoundError(
                f"Submission file not found: {path}\n"
                f"Run training for {model['name']} first."
            )
        df = load_submission(path)
        submissions.append(df)
        weights.append(model["weight"])
        print(f"[ENSEMBLE] Loaded {model['name']} (LB={model['lb_score']}) "
              f"from {path.name}")

    blend_type = config["ensemble"]["blend_type"]

    if blend_type == "division_attention":
        blended = division_attention_blend(submissions, weights)
    else:
        raise ValueError(f"Unknown blend type: {blend_type}")

    # Apply TAX_SMOOTHING
    postprocess = config["ensemble"].get("postprocess", None)
    if postprocess == "TAX_SMOOTHING":
        tax_cfg = config.get("tax_smoothing", {})
        taxonomy_path = (
            Path(config["paths"]["competition_dir"]) / "taxonomy.csv"
        )
        blended = apply_tax_smoothing(
            blended,
            taxonomy_path=taxonomy_path,
            genus_alpha=tax_cfg.get("genus_alpha", 0.15),
            class_alpha=tax_cfg.get("class_alpha", 0.05),
        )

    return blended
