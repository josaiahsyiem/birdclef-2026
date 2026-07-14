"""
scripts/predict.py
-------------------
Full inference pipeline — loads all trained models and produces
the final ensemble submission.

Usage:
    python scripts/predict.py --config configs/ensemble_config.yaml

This script assumes all three models have already been trained and
their submission CSVs exist in the output directory:
    - subm_21.csv   (Model_21)
    - subm_52p.csv  (Model_52)
    - subm_74.csv   (Model_74)

It then runs the division_attention blend and TAX_SMOOTHING
post-processing to produce the final submission.csv.
"""

from src.utils import seed_everything
from src.ensemble import blend_from_config
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full ensemble inference pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ensemble_config.yaml",
        help="Path to ensemble_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission filename",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_submission(sub: pd.DataFrame, n_classes: int = 234) -> None:
    """
    Run basic sanity checks on the final submission.

    Parameters
    ----------
    sub : pd.DataFrame
    n_classes : int
    """
    print("\nValidating submission...")
    prob_cols = [c for c in sub.columns if c != "row_id"]

    checks = {
        "rows": len(sub),
        "columns": sub.shape[1],
        "class_columns": len(prob_cols),
        "missing_values": sub[prob_cols].isnull().sum().sum(),
        "min_probability": sub[prob_cols].values.min(),
        "max_probability": sub[prob_cols].values.max(),
        "duplicate_row_ids": sub["row_id"].duplicated().sum(),
    }

    for check, value in checks.items():
        print(f"  {check}: {value}")

    assert checks["missing_values"] == 0, "Submission contains missing values"
    assert checks["duplicate_row_ids"] == 0, "Submission has duplicate row_ids"
    assert checks["class_columns"] == n_classes, (
        f"Expected {n_classes} class columns, got {checks['class_columns']}"
    )
    assert 0.0 <= checks["min_probability"] <= 1.0, "Probabilities out of range"
    assert 0.0 <= checks["max_probability"] <= 1.0, "Probabilities out of range"

    print("Submission validation passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(args.seed)

    output_dir = Path(config["paths"]["output_dir"])
    n_classes = config["competition"]["num_classes"]

    print("BirdCLEF+ 2026 — Ensemble Inference")
    print("="*50)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    # Print ensemble summary
    print("\nEnsemble configuration:")
    for model in config["ensemble"]["models"]:
        print(f"  {model['name']}: weight={model['weight']} "
              f"LB={model['lb_score']}")
    print(f"  Blend type: {config['ensemble']['blend_type']}")
    print(f"  Post-process: {config['ensemble']['postprocess']}")

    # Run ensemble blend
    print("\nRunning ensemble blend...")
    blended = blend_from_config(config, output_dir)

    # Prepare final submission
    blended = blended.reset_index()
    if "row_id" not in blended.columns:
        blended.insert(0, "row_id", blended.index)

    # Clip to valid probability range
    prob_cols = [c for c in blended.columns if c != "row_id"]
    blended[prob_cols] = blended[prob_cols].clip(0.0, 1.0)

    # Validate
    validate_submission(blended, n_classes)

    # Save
    output_path = Path(args.output)
    blended.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")
    print(f"Shape: {blended.shape}")
    print(f"Min prob: {blended[prob_cols].values.min():.6f}")
    print(f"Max prob: {blended[prob_cols].values.max():.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
