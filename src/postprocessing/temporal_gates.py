"""
src/postprocessing/temporal_gates.py
--------------------------------------
Temporal post-processing gates applied to Model_74 predictions.

These gates operate on the rank-blended ProtoSSM + SED predictions
before the final TAX_SMOOTHING pass. They handle edge cases that the
models alone cannot resolve:

  Gate 1 — Noise suppression
  Gate 2 — Temporal continuity
  Gate 3 — SED spike preservation
  Gate 4 — Sonotype mirroring
  Gate 5 — Adaptive rare-class thresholding

Each gate makes a small targeted adjustment and passes the result
to the next gate. The cumulative effect is a cleaner, more calibrated
prediction surface.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Gate 1: Noise Suppression
# ---------------------------------------------------------------------------

def gate_noise_suppression(
    pred: np.ndarray,
    rank_proto: np.ndarray,
    p_proto: np.ndarray,
    p_sed: np.ndarray,
    proto_weight: float = 0.08,
) -> tuple:
    """
    Suppress predictions where ProtoSSM is confident but SED strongly disagrees.

    Rationale: if ProtoSSM says a species is present (>0.50) but SED
    gives it almost zero probability (<0.05), it is likely a false positive
    from the ProtoSSM pipeline. Pull the prediction back toward ProtoSSM rank
    with a small weight to reduce but not eliminate the signal.

    Parameters
    ----------
    pred : np.ndarray of shape (n_windows, n_classes)
    rank_proto : np.ndarray of shape (n_windows, n_classes)
    p_proto : np.ndarray of shape (n_windows, n_classes)
    p_sed : np.ndarray of shape (n_windows, n_classes)
    proto_weight : float

    Returns
    -------
    tuple of (pred, fake_only_mask)
    """
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    pred = np.where(
        fake_only,
        (1 - proto_weight) * pred + proto_weight * rank_proto,
        pred
    )
    return pred, fake_only


# ---------------------------------------------------------------------------
# Gate 2: Temporal Continuity
# ---------------------------------------------------------------------------

def gate_temporal_continuity(
    pred: np.ndarray,
    rank_proto: np.ndarray,
    p_proto: np.ndarray,
    p_sed: np.ndarray,
    fake_only: np.ndarray,
    continuity_weight: float = 0.15,
) -> tuple:
    """
    Protect continuous calls that span multiple windows.

    Uses a fat-tailed t-distribution kernel (35s context window, ±3 windows)
    to compute a context-aware ProtoSSM score. Windows where the context score
    is high but SED is low get a boost from the contextual signal.

    Rationale: some species call continuously across multiple 5s windows.
    The SED model (which looks at each window independently) may miss the
    middle windows of a long call. The temporal context fixes this.

    Parameters
    ----------
    pred : np.ndarray of shape (n_windows, n_classes)
    rank_proto : np.ndarray
    p_proto : np.ndarray
    p_sed : np.ndarray
    fake_only : np.ndarray — boolean mask from Gate 1
    continuity_weight : float

    Returns
    -------
    tuple of (pred, proto_cont_mask)
    """
    # Fat-tailed t-distribution kernel (heavier tails than Gaussian)
    offs = np.arange(-3, 4, dtype=np.float32)
    kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    kernel = (kernel / kernel.sum()).astype(np.float32)

    # Apply kernel across windows
    xp = np.pad(p_proto, ((3, 3), (0, 0)), mode="edge")
    pa_ctx = sum(kernel[i] * xp[i:i + len(p_proto)] for i in range(7))

    xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (
        (xctx > 0.88) & (rank_proto > 0.75) &
        (p_sed < 0.12) & (~fake_only)
    )

    pred = np.where(
        proto_cont,
        (1 - continuity_weight) * pred +
        continuity_weight * np.maximum(rank_proto, xctx),
        pred
    )
    return pred, proto_cont


# ---------------------------------------------------------------------------
# Gate 3: SED Spike Preservation
# ---------------------------------------------------------------------------

def gate_sed_spike_preservation(
    pred: np.ndarray,
    rank_proto: np.ndarray,
    rank_sed: np.ndarray,
    fake_only: np.ndarray,
    proto_cont: np.ndarray,
    sed_weight: float = 0.12,
) -> np.ndarray:
    """
    Preserve brief high-confidence SED detections that ProtoSSM missed.

    Rationale: the SED model excels at detecting short, sharp vocalizations
    (single calls, brief songs). ProtoSSM may smooth these out. When SED
    is very confident (top 5% rank) and ProtoSSM is uncertain, trust SED.

    Parameters
    ----------
    pred : np.ndarray
    rank_proto : np.ndarray
    rank_sed : np.ndarray
    fake_only : np.ndarray
    proto_cont : np.ndarray
    sed_weight : float

    Returns
    -------
    np.ndarray
    """
    sed_only = (
        (rank_sed > 0.95) & (rank_proto < 0.80) &
        (~fake_only) & (~proto_cont)
    )
    pred = np.where(
        sed_only,
        (1 - sed_weight) * pred + sed_weight * rank_sed,
        pred
    )
    return pred


# ---------------------------------------------------------------------------
# Gate 4: Sonotype Mirroring
# ---------------------------------------------------------------------------

# Species groups that are acoustically identical or nearly identical.
# When one is detected, all members of the group should receive the
# same (maximum) prediction score.
SONOTYPE_MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)


def gate_sonotype_mirroring(
    pred: np.ndarray,
    cols: list,
) -> np.ndarray:
    """
    Max-pool predictions across acoustically identical species groups.

    Rationale: some species in the Pantanal dataset share nearly identical
    vocalizations (sonotypes). The model cannot distinguish between them,
    so assigning the maximum confidence to all members of the group
    avoids penalizing the model for picking the wrong member.

    Parameters
    ----------
    pred : np.ndarray of shape (n_windows, n_classes)
    cols : list of str — species label columns

    Returns
    -------
    np.ndarray
    """
    col_to_idx = {l: i for i, l in enumerate(cols)}
    pred = pred.copy()

    for group in SONOTYPE_MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = pred[:, valid_idx].max(axis=1, keepdims=True)
            pred[:, valid_idx] = group_max

    return pred


# ---------------------------------------------------------------------------
# Gate 5: Adaptive Rare-Class Thresholding
# ---------------------------------------------------------------------------

def gate_rare_class_thresholding(
    pred: np.ndarray,
    cols: list,
    taxonomy_path,
    rare_taxa: set = None,
    suppression_factor: float = 0.9,
    threshold_offset: float = 0.05,
) -> np.ndarray:
    """
    Suppress low-confidence predictions for rare taxonomic classes.

    Rationale: Amphibia, Mammalia, and Reptilia are underrepresented in
    the training data. The model tends to produce spurious low-confidence
    detections for these classes. Suppressing predictions below a
    per-class adaptive threshold reduces false positives.

    Parameters
    ----------
    pred : np.ndarray of shape (n_windows, n_classes)
    cols : list of str
    taxonomy_path : str or Path
    rare_taxa : set of str — taxonomic classes to suppress
    suppression_factor : float — multiply suppressed values by this
    threshold_offset : float — threshold = mean + offset

    Returns
    -------
    np.ndarray
    """
    if rare_taxa is None:
        rare_taxa = {"Amphibia", "Mammalia", "Reptilia"}

    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        print(f"[Gate 5] taxonomy.csv not found at {taxonomy_path} — skipping")
        return pred

    try:
        tax_df = pd.read_csv(taxonomy_path).set_index("primary_label")
        pred = pred.copy()
        for ci, species in enumerate(cols):
            if (species in tax_df.index and
                    tax_df.loc[species, "class_name"] in rare_taxa):
                vals = pred[:, ci]
                thr = vals.mean() + threshold_offset
                pred[:, ci] = np.where(
                    vals < thr,
                    vals * suppression_factor,
                    vals
                )
    except Exception as e:
        print(f"[Gate 5] Skipped: {e}")

    return pred


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def apply_temporal_gates(
    p_proto: np.ndarray,
    p_sed: np.ndarray,
    cols: list,
    proto_w: float = 0.60,
    sed_w: float = 0.40,
    taxonomy_path=None,
) -> np.ndarray:
    """
    Run all 5 gates in sequence on ProtoSSM and SED predictions.

    Parameters
    ----------
    p_proto : np.ndarray of shape (n_windows, n_classes)
    p_sed : np.ndarray of shape (n_windows, n_classes)
    cols : list of str
    proto_w : float — ProtoSSM rank blend weight
    sed_w : float — SED rank blend weight
    taxonomy_path : str or Path, optional

    Returns
    -------
    np.ndarray of shape (n_windows, n_classes)
    """
    EPS = 1e-5
    p_proto = np.clip(p_proto, EPS, 1 - EPS)
    p_sed = np.clip(p_sed, EPS, 1 - EPS)

    rank_proto = pd.DataFrame(p_proto).rank(
        axis=0, pct=True).to_numpy(np.float32)
    rank_sed = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)

    # Base rank blend
    pred = proto_w * rank_proto + sed_w * rank_sed

    # Gate 1
    pred, fake_only = gate_noise_suppression(pred, rank_proto, p_proto, p_sed)

    # Gate 2
    pred, proto_cont = gate_temporal_continuity(
        pred, rank_proto, p_proto, p_sed, fake_only
    )

    # Gate 3
    pred = gate_sed_spike_preservation(
        pred, rank_proto, rank_sed, fake_only, proto_cont
    )

    # Gate 4
    pred = gate_sonotype_mirroring(pred, cols)

    # Gate 5
    if taxonomy_path is not None:
        pred = gate_rare_class_thresholding(pred, cols, taxonomy_path)

    return pred
