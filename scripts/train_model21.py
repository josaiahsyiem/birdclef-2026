"""
scripts/train_model21.py
-------------------------
Training script for Model_21 — Distilled EfficientNet SED + ProtoSSM.

This model trains an EfficientNet-B0 on mel spectrograms with knowledge
distillation from frozen Perch v2 embeddings, followed by ProtoSSM v5
for temporal sequence modeling.

Usage:
    python scripts/train_model21.py --config configs/ensemble_config.yaml

Requirements:
    - BirdCLEF+ 2026 competition data
    - Perch v2 ONNX model (for distillation targets)
    - ~8-15 hours on RTX 4060 8GB (25 epochs x 5 folds)

Outputs (saved to configs.paths.output_dir):
    - sed_fold{i}.onnx     Distilled SED model per fold
    - proto_ssm_v5.pt      ProtoSSM v5 checkpoint
    - subm_21.csv          Final submission file for this model
    - subm_52p.csv         ProtoSSM sub-output (Model_52)
"""

from src.models.proto_ssm import ProtoSSMv2
from src.models.distilled_sed import DistilledEfficientNetSED, MelSpectrogramTransform
from src.utils import (
    parse_soundscape_filename,
    union_labels,
    build_label_matrix,
    macro_auc_skip_empty,
    seed_everything,
)
import os
import sys
import gc
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Model_21 — Distilled EfficientNet SED"
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
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Which folds to train (default: all 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdCLEFDataset(Dataset):
    """
    Dataset for BirdCLEF+ 2026 focal recordings.

    Loads pre-cached waveforms and returns 5-second clips with
    their multi-hot species labels.
    """

    def __init__(
        self,
        cache_meta: pd.DataFrame,
        cache_dir: Path,
        label_to_idx: dict,
        n_classes: int,
        duration: int = 5,
        sample_rate: int = 32000,
        augment: bool = False,
        aug_prob: float = 0.5,
    ):
        """
        Parameters
        ----------
        cache_meta : pd.DataFrame
            Metadata for cached waveform files.
        cache_dir : Path
            Directory containing cached .npy waveform files.
        label_to_idx : dict
        n_classes : int
        duration : int — clip duration in seconds
        sample_rate : int
        augment : bool — apply audio augmentation during training
        aug_prob : float — probability of applying each augmentation
        """
        self.cache_meta = cache_meta.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.label_to_idx = label_to_idx
        self.n_classes = n_classes
        self.samples = sample_rate * duration
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.cache_meta)

    def __getitem__(self, idx):
        row = self.cache_meta.iloc[idx]

        # Load cached waveform
        waveform = np.load(self.cache_dir / f"{row['cache_id']}.npy")
        waveform = torch.from_numpy(waveform).float()

        # Pad or crop to fixed length
        if len(waveform) < self.samples:
            waveform = F.pad(waveform, (0, self.samples - len(waveform)))
        else:
            waveform = waveform[:self.samples]

        # Audio augmentation
        if self.augment:
            waveform = self._augment(waveform)

        # Build label vector
        label = torch.zeros(self.n_classes)
        primary = row.get("primary_label", "")
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0

        return waveform, label

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain and Gaussian noise augmentation."""
        import random
        if random.random() < self.aug_prob:
            # Random gain (-6 to +6 dB)
            gain_db = random.uniform(-6.0, 6.0)
            gain = 10 ** (gain_db / 20.0)
            waveform = waveform * gain

        if random.random() < self.aug_prob:
            # Additive Gaussian noise (SNR 10-30 dB)
            snr_db = random.uniform(10.0, 30.0)
            signal_power = waveform.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise = torch.randn_like(waveform) * noise_power.sqrt()
            waveform = waveform + noise

        return waveform


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.02,
) -> torch.Tensor:
    """
    Focal binary cross-entropy loss with label smoothing.

    Focal loss down-weights easy examples (already well-classified)
    and focuses training on hard examples. Critical for BirdCLEF because
    the dataset is heavily imbalanced — most windows contain no species.

    Parameters
    ----------
    logits : torch.Tensor of shape (batch, n_classes)
    targets : torch.Tensor of shape (batch, n_classes)
    gamma : float — focusing parameter (2.0 = standard focal loss)
    label_smoothing : float — smooth hard 0/1 labels

    Returns
    -------
    torch.Tensor — scalar loss
    """
    targets_smooth = (
        targets * (1 - label_smoothing) + label_smoothing / 2.0
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, targets_smooth, reduction="none"
    )
    pt = torch.exp(-bce)
    focal = ((1 - pt) ** gamma) * bce
    return focal.mean()


# ---------------------------------------------------------------------------
# MixUp augmentation
# ---------------------------------------------------------------------------

def mixup_batch(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple:
    """
    Apply MixUp augmentation to a batch of waveforms and labels.

    MixUp creates synthetic training examples by linearly interpolating
    between pairs of examples. This improves generalization and acts as
    a regularizer.

    Parameters
    ----------
    waveforms : torch.Tensor of shape (batch, samples)
    labels : torch.Tensor of shape (batch, n_classes)
    alpha : float — Beta distribution parameter

    Returns
    -------
    tuple of (mixed_waveforms, mixed_labels)
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(waveforms.size(0))
    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[idx]
    # Hard union labels (take max instead of weighted blend)
    mixed_labels = torch.max(labels, labels[idx])
    return mixed_waveforms, mixed_labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    mel_transform: MelSpectrogramTransform,
    device: torch.device,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    mixup_prob: float = 0.5,
) -> float:
    """
    Train model for one epoch.

    Parameters
    ----------
    model : DistilledEfficientNetSED
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    scheduler : LR scheduler
    mel_transform : MelSpectrogramTransform
    device : torch.device
    use_mixup : bool
    mixup_alpha : float
    mixup_prob : float

    Returns
    -------
    float — mean training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (waveforms, labels) in enumerate(loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # MixUp augmentation
        if use_mixup and np.random.random() < mixup_prob:
            waveforms, labels = mixup_batch(waveforms, labels, mixup_alpha)

        # Convert to mel spectrogram
        with torch.no_grad():
            mel = mel_transform(waveforms)

        # Forward pass
        out = model(mel)
        loss = focal_bce_loss(out["clip_logits"], labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    mel_transform: MelSpectrogramTransform,
    device: torch.device,
) -> tuple:
    """
    Validate model on a held-out fold.

    Returns
    -------
    tuple of (val_loss, val_auc)
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            mel = mel_transform(waveforms)
            out = model(mel)
            all_logits.append(out["clip_logits"].cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss = F.binary_cross_entropy_with_logits(
        torch.tensor(all_logits),
        torch.tensor(all_labels),
    ).item()

    val_auc = macro_auc_skip_empty(all_labels, all_logits)
    return val_loss, val_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(args.seed)

    comp_dir = Path(config["paths"]["competition_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = config["competition"]["num_classes"]
    sr = config["competition"]["sample_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg21 = config["model_21"]

    # Mel transform (shared across train/val)
    mel_transform = MelSpectrogramTransform(
        sample_rate=sr,
        n_fft=cfg21["n_fft"],
        hop_length=cfg21["hop_length"],
        n_mels=cfg21["n_mels"],
        fmin=cfg21["fmin"],
        fmax=cfg21["fmax"],
    ).to(device)

    print(f"\nTraining Model_21 — Distilled EfficientNet SED")
    print(f"Backbone: {cfg21['backbone']}")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {cfg21['epochs']}")
    print(f"Batch size: {cfg21['batch_size']}")

    # Load data
    sample_sub = pd.read_csv(comp_dir / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    # Load waveform cache metadata
    cache_dir = Path(config["paths"]["cache_dir"]) / "waveforms"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Waveform cache not found at {cache_dir}\n"
            "Run Perch preprocessing on Kaggle first and copy cache here."
        )

    cache_meta = pd.read_csv(cache_dir / "audio_cache_meta.csv")
    cache_meta = cache_meta[
        cache_meta["primary_label"].isin(label_to_idx)
    ].reset_index(drop=True)

    print(f"\nFocal recordings: {len(cache_meta)}")

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=cfg21["n_folds"],
        shuffle=True,
        random_state=args.seed,
    )

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(cache_meta, cache_meta["primary_label"])
    ):
        if fold not in args.folds:
            continue

        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{cfg21['n_folds']}")
        print(f"{'='*50}")
        print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

        train_meta = cache_meta.iloc[train_idx].reset_index(drop=True)
        val_meta = cache_meta.iloc[val_idx].reset_index(drop=True)

        train_ds = BirdCLEFDataset(
            train_meta, cache_dir, label_to_idx, n_classes,
            augment=True, aug_prob=0.5,
        )
        val_ds = BirdCLEFDataset(
            val_meta, cache_dir, label_to_idx, n_classes,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg21["batch_size"],
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg21["batch_size"] * 2,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        # Model
        model = DistilledEfficientNetSED(
            n_classes=n_classes,
            backbone_name=cfg21["backbone"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg21["lr"],
            weight_decay=cfg21["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg21["lr"],
            epochs=cfg21["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        best_auc = 0.0
        best_state = None
        patience_counter = 0
        patience = 5

        for epoch in range(cfg21["epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                mel_transform, device,
            )
            val_loss, val_auc = validate(
                model, val_loader, mel_transform, device
            )

            print(f"Epoch {epoch+1:3d}/{cfg21['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save best model for this fold
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_path = output_dir / f"sed_fold{fold}.pt"
        torch.save(model.state_dict(), fold_path)
        print(
            f"Saved fold {fold} model to {fold_path} (best AUC={best_auc:.4f})")
        fold_aucs.append(best_auc)

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    if fold_aucs:
        print(f"\nMean CV AUC: {np.mean(fold_aucs):.4f}")
        print(f"Model_21 training complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
