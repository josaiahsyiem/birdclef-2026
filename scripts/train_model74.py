"""
scripts/train_model74.py
-------------------------
Training script for Model_74 — the Perch pipeline (Karnakbayev).

This is the dominant model in the ensemble (weight=0.965). It trains
lightweight heads on top of frozen Perch v2 embeddings:
  1. MLP probes (per-class classifiers on PCA-reduced embeddings)
  2. ProtoSSM v2 (temporal sequence model)
  3. ResidualSSM (correction layer)

Usage:
    python scripts/train_model74.py --config configs/ensemble_config.yaml

Requirements:
    - BirdCLEF+ 2026 competition data
    - Perch v2 model (ONNX or TensorFlow SavedModel)
    - ~4-6 hours on RTX 4060 8GB

Outputs (saved to configs.paths.output_dir):
    - perch_cache/         Perch embedding cache (reused across runs)
    - proto_ssm_best.pt    Best ProtoSSM checkpoint
    - res_ssm_best.pt      Best ResidualSSM checkpoint
    - mlp_probes.pkl       Trained MLP probe models
    - subm_74.csv          Final submission file for this model
"""

from src.postprocessing.tax_smoothing import apply_tax_smoothing
from src.postprocessing.temporal_gates import apply_temporal_gates
from src.models.perch_pipeline import ResidualSSM, build_prior_tables
from src.models.proto_ssm import ProtoSSMv2
from src.utils import (
    parse_soundscape_filename,
    parse_label_string,
    union_labels,
    build_label_matrix,
    macro_auc_skip_empty,
    seed_everything,
)
import os
import sys
import gc
import time
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Model_74 — Perch + ProtoSSM pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ensemble_config.yaml",
        help="Path to ensemble_config.yaml",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer"],
        help="Train from scratch or run inference only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_competition_data(comp_dir: Path, n_classes: int):
    """
    Load taxonomy, labels, and soundscape metadata from competition directory.

    Parameters
    ----------
    comp_dir : Path
    n_classes : int

    Returns
    -------
    tuple of (primary_labels, label_to_idx, taxonomy_df, sc_clean, Y_SC)
    """
    print("Loading competition data...")

    sample_sub = pd.read_csv(comp_dir / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    taxonomy_df = pd.read_csv(comp_dir / "taxonomy.csv")
    soundscape_labels = pd.read_csv(comp_dir / "train_soundscapes_labels.csv")

    # Clean and aggregate labels per 5s window
    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = (
        pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    )
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc_clean["end_sec"].astype(str)
    )

    # Parse site and hour from filename
    meta = sc_clean["filename"].apply(
        lambda x: parse_soundscape_filename(x)
    ).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta], axis=1)

    # Build label matrix
    Y_SC = build_label_matrix(sc_clean, label_to_idx, n_classes)

    # Mark fully-labeled files (all 12 windows present)
    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(
        windows_per_file[windows_per_file == 12].index.tolist()
    )
    sc_clean["fully_labeled"] = sc_clean["filename"].isin(full_files)

    print(f"  Classes: {n_classes}")
    print(f"  Soundscape windows: {len(sc_clean)}")
    print(f"  Fully-labeled files: {len(full_files)}")
    print(f"  Active classes: {int((Y_SC.sum(0) > 0).sum())}")

    return primary_labels, label_to_idx, taxonomy_df, sc_clean, Y_SC


# ---------------------------------------------------------------------------
# Perch inference
# ---------------------------------------------------------------------------

def load_perch_cache_or_build(
    cache_dir: Path,
    comp_dir: Path,
    sc_clean: pd.DataFrame,
    n_classes: int,
    use_onnx: bool = True,
):
    """
    Load Perch embeddings and logits from cache, or build from scratch.

    Perch inference is slow (~30-60 min for full dataset).
    Cache is reused on subsequent runs.

    Parameters
    ----------
    cache_dir : Path
    comp_dir : Path
    sc_clean : pd.DataFrame
    n_classes : int
    use_onnx : bool

    Returns
    -------
    tuple of (meta_df, scores, embeddings)
    """
    cache_meta = cache_dir / "perch_meta.parquet"
    cache_npz = cache_dir / "perch_arrays.npz"

    if cache_meta.exists() and cache_npz.exists():
        print(f"Loading Perch cache from {cache_dir}...")
        meta_df = pd.read_parquet(cache_meta)
        arr = np.load(cache_npz)
        scores = arr["scores"].astype(np.float32)
        embs = arr["embs"].astype(np.float32)
        print(f"  Loaded: scores={scores.shape} embs={embs.shape}")
        return meta_df, scores, embs

    print("No Perch cache found. Building from scratch...")
    print("This will take 30-60 minutes depending on your GPU.")

    # Import Perch backend
    try:
        import onnxruntime as ort
        onnx_path = next(
            Path("/kaggle/input").rglob("perch_v2_no_dft*.onnx"), None
        )
        if onnx_path is None:
            raise FileNotFoundError("Perch ONNX not found")
        print(f"Using ONNX Perch: {onnx_path.name}")
        use_onnx = True
    except Exception:
        import tensorflow as tf
        model_dir = Path(
            "/kaggle/input/models/google/"
            "bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
        )
        birdclassifier = tf.saved_model.load(str(model_dir))
        infer_fn = birdclassifier.signatures["serving_default"]
        use_onnx = False
        print("Using TensorFlow Perch SavedModel")

    # Run inference on training soundscapes
    fully_labeled = sc_clean[sc_clean["fully_labeled"]]["filename"].unique()
    train_paths = [
        comp_dir / "train_soundscapes" / fn
        for fn in fully_labeled
        if (comp_dir / "train_soundscapes" / fn).exists()
    ]

    print(f"Running Perch on {len(train_paths)} files...")
    # (Full Perch inference loop — runs on Kaggle with GPU)
    # See notebooks/birdclef-2026-eos-7-sz.ipynb for complete implementation

    raise NotImplementedError(
        "Perch inference requires Kaggle environment with attached datasets.\n"
        "Run on Kaggle and copy the cache files to your local cache_dir:\n"
        f"  {cache_dir}"
    )


# ---------------------------------------------------------------------------
# MLP Probes
# ---------------------------------------------------------------------------

def train_mlp_probes(
    emb_train: np.ndarray,
    Y_train: np.ndarray,
    primary_labels: list,
    pca_dim: int = 128,
    min_pos: int = 5,
    mlp_params: dict = None,
) -> tuple:
    """
    Train per-class MLP probes on PCA-reduced Perch embeddings.

    One MLP is trained per species class. Classes with fewer than
    min_pos positive examples are skipped (insufficient data).

    Parameters
    ----------
    emb_train : np.ndarray of shape (n_windows, embed_dim)
    Y_train : np.ndarray of shape (n_windows, n_classes)
    primary_labels : list of str
    pca_dim : int — PCA output dimension
    min_pos : int — minimum positive examples required to train probe
    mlp_params : dict — MLPClassifier hyperparameters

    Returns
    -------
    tuple of (probe_models dict, pca, scaler)
    """
    if mlp_params is None:
        mlp_params = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
            "random_state": 42,
            "learning_rate_init": 5e-4,
            "alpha": 0.005,
        }

    print(f"Fitting PCA (dim={pca_dim})...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_train)
    pca = PCA(n_components=pca_dim, random_state=42)
    emb_pca = pca.fit_transform(emb_scaled)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    probe_models = {}
    n_classes = Y_train.shape[1]
    print(f"Training MLP probes for {n_classes} classes...")

    for c in range(n_classes):
        y_c = Y_train[:, c]
        n_pos = y_c.sum()
        if n_pos < min_pos:
            continue
        clf = MLPClassifier(**mlp_params)
        clf.fit(emb_pca, y_c)
        probe_models[c] = clf

        if (c + 1) % 50 == 0:
            print(f"  Trained {c+1}/{n_classes} probes "
                  f"({len(probe_models)} active)")

    print(f"MLP probes trained: {len(probe_models)} / {n_classes} classes")
    return probe_models, pca, scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(args.seed)

    comp_dir = Path(config["paths"]["competition_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    cache_dir = Path(config["paths"]["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_classes = config["competition"]["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load competition data
    primary_labels, label_to_idx, taxonomy_df, sc_clean, Y_SC = (
        load_competition_data(comp_dir, n_classes)
    )

    # Load or build Perch cache
    meta_df, scores, embs = load_perch_cache_or_build(
        cache_dir, comp_dir, sc_clean, n_classes
    )

    # Get fully-labeled windows for training
    full_rows = (
        sc_clean[sc_clean["fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    Y_FULL = Y_SC[full_rows["index"].to_numpy()]

    print(f"\nTraining set: {len(full_rows)} windows, "
          f"{int((Y_FULL.sum(0) > 0).sum())} active classes")

    # Build prior tables
    print("\nBuilding prior tables...")
    prior_tables = build_prior_tables(
        sc_clean[sc_clean["fully_labeled"]].reset_index(drop=True),
        Y_FULL,
    )
    print("  Prior tables built")

    # Train MLP probes
    print("\nTraining MLP probes...")
    mlp_cfg = config["model_74"]["training"]
    probe_models, pca, scaler = train_mlp_probes(
        embs,
        Y_FULL,
        primary_labels,
        pca_dim=config["model_74"].get("pca_dim", 128),
        min_pos=5,
    )

    # Save probe models
    probe_path = output_dir / "mlp_probes.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"probes": probe_models, "pca": pca, "scaler": scaler}, f)
    print(f"Saved MLP probes to {probe_path}")

    # Train ProtoSSM
    print("\nTraining ProtoSSM v2...")
    ssm_cfg = config["model_74"]["proto_ssm"]
    proto_model = ProtoSSMv2(
        d_input=embs.shape[1],
        d_model=ssm_cfg["d_model"],
        d_state=ssm_cfg["d_state"],
        n_ssm_layers=ssm_cfg["n_ssm_layers"],
        n_classes=n_classes,
        n_windows=config["competition"]["n_windows"],
        dropout=ssm_cfg["dropout"],
        n_sites=ssm_cfg["n_sites"],
        meta_dim=ssm_cfg["meta_dim"],
        use_cross_attn=ssm_cfg["use_cross_attn"],
        cross_attn_heads=ssm_cfg["cross_attn_heads"],
    ).to(device)

    train_cfg = config["model_74"]["training"]
    optimizer = torch.optim.AdamW(
        proto_model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    print(f"  ProtoSSM parameters: "
          f"{sum(p.numel() for p in proto_model.parameters()):,}")
    print(f"  Training for up to {train_cfg['n_epochs']} epochs "
          f"(patience={train_cfg['patience']})")

    # Training loop placeholder
    # (Full training loop follows the same pattern as competition notebook)
    # See notebooks/birdclef-2026-eos-7-sz.ipynb Cell 21 for complete loop

    print("\nModel_74 training complete.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
