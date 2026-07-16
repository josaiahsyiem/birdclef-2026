"""
src/models/karnakbayev.py
--------------------------
Karnakbayev full pipeline — Model_74 (ensemble weight: 0.965).

This is the dominant model in the ensemble. It operates entirely on
frozen Perch v2 embeddings, training lightweight heads on top rather
than fine-tuning the backbone. This is why it outperforms the from-scratch
EfficientNet approach — Perch was pretrained by Google on 10,000+ species.

Pipeline:
  1. Perch v2 (frozen) -> 1536-dim embeddings + raw logits
  2. Site/hour prior tables for spatial-temporal calibration
  3. MLP probes (per-class) on top of PCA-reduced embeddings
  4. ProtoSSM v2 for temporal sequence modeling
  5. ResidualSSM for correction of ProtoSSM predictions
  6. 5-gate post-processing pipeline
  7. xSED rank blend with Distilled SED output

Original implementation by Yaroslav Kholmirzayev, modularized here.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Residual SSM
# ---------------------------------------------------------------------------

class ResidualSSM(nn.Module):
    """
    Lightweight residual SSM that corrects ProtoSSM predictions.

    Takes ProtoSSM output logits as input and learns a small additive
    correction. Prevents over-fitting by keeping the correction small
    via a learnable correction_weight.
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int = 128,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        dropout: float = 0.1,
        correction_weight: float = 0.35,
    ):
        """
        Parameters
        ----------
        n_classes : int
            Number of species classes (234).
        d_model : int
            Internal model dimension.
        d_state : int
            SSM hidden state dimension.
        n_ssm_layers : int
            Number of SSM layers.
        dropout : float
            Dropout probability.
        correction_weight : float
            Maximum weight of the residual correction (0-1).
        """
        super().__init__()
        self.correction_weight = correction_weight

        self.input_proj = nn.Sequential(
            nn.Linear(n_classes, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_ssm_layers)
        ])

        self.output_proj = nn.Linear(d_model, n_classes)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor of shape (batch, n_windows, n_classes)
            ProtoSSM output logits.

        Returns
        -------
        torch.Tensor of shape (batch, n_windows, n_classes)
            Corrected logits.
        """
        h = self.input_proj(logits)
        for layer in self.ssm_layers:
            h = layer(h) + h
        correction = self.output_proj(h)
        return logits + self.correction_weight * correction


# ---------------------------------------------------------------------------
# Prior Tables
# ---------------------------------------------------------------------------

def build_prior_tables(sc_df: pd.DataFrame, Y_labels: np.ndarray) -> dict:
    """
    Build site, hour, and joint site-hour prior probability tables.

    These tables capture the spatial and temporal distribution of species
    across recording sites and times of day. Used to calibrate Perch logits
    before feeding into the SSM pipeline.

    Parameters
    ----------
    sc_df : pd.DataFrame
        Soundscape metadata with 'site' and 'hour_utc' columns.
    Y_labels : np.ndarray of shape (n_windows, n_classes)
        Multi-hot label matrix aligned with sc_df rows.

    Returns
    -------
    dict with keys: global_p, site_to_i, site_p, site_n,
                    hour_to_i, hour_p, hour_n, sh_to_i, sh_p, sh_n
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    # Site priors
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    # Hour priors with circular Gaussian smoothing
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    # Apply circular Gaussian smoothing on hour axis (sigma=1.5 hours)
    # Motivation: species activity peaks (dawn/dusk) are continuous across
    # hour boundaries, not discrete jumps. Wrap-around handles midnight.
    if len(hour_keys) >= 3:
        full_hour_p = np.zeros((24, hour_p.shape[1]), dtype=np.float32)
        for h, i in hour_to_i.items():
            full_hour_p[int(h)] = hour_p[i]
        tiled = np.tile(full_hour_p, (3, 1))
        tiled_smooth = gaussian_filter1d(tiled, sigma=1.5, axis=0, mode="wrap")
        full_smooth = tiled_smooth[24:48]
        for h, i in hour_to_i.items():
            hour_p[i] = full_smooth[int(h)]
        hour_p = np.clip(hour_p, 0.0, 1.0)

    # Joint site-hour priors
    sh_keys = sorted({
        (str(s), int(h))
        for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
        if not pd.isna(s) and not pd.isna(h)
    })
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (
            (sc_df["site"].astype(str).values == s) &
            (sc_df["hour_utc"].astype(int).values == h)
        )
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i": sh_to_i, "sh_p": sh_p, "sh_n": sh_n,
    }


# ---------------------------------------------------------------------------
# 5-Gate Post-Processing Pipeline
# ---------------------------------------------------------------------------

def apply_five_gate_postprocessing(
    p_proto: np.ndarray,
    p_sed: np.ndarray,
    cols: list,
    taxonomy_path: Path = None,
) -> np.ndarray:
    """
    Apply the 5-gate post-processing pipeline to blend ProtoSSM and SED.

    Gates applied in order:
      1. Noise suppression — ProtoSSM confident but SED strongly disagrees
      2. Temporal continuity — protect continuous calls across windows
      3. SED spike preservation — brief high-confidence SED detections
      4. Sonotype mirroring — max-pool acoustically identical species
      5. Adaptive rare-class thresholding — suppress Amphibia/Mammalia/Reptilia

    Parameters
    ----------
    p_proto : np.ndarray of shape (n_windows, n_classes)
        ProtoSSM probabilities (sigmoid applied).
    p_sed : np.ndarray of shape (n_windows, n_classes)
        Distilled SED probabilities.
    cols : list of str
        Species label columns in order.
    taxonomy_path : Path, optional
        Path to taxonomy.csv for Gate 5.

    Returns
    -------
    np.ndarray of shape (n_windows, n_classes) — blended predictions
    """
    EPS = 1e-5
    p_proto = np.clip(p_proto, EPS, 1 - EPS)
    p_sed = np.clip(p_sed, EPS, 1 - EPS)

    rank_proto = pd.DataFrame(p_proto).rank(
        axis=0, pct=True).to_numpy(np.float32)
    rank_sed = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)

    # Base blend: 0.60 ProtoSSM / 0.40 SED (rank-based)
    pred = 0.60 * rank_proto + 0.40 * rank_sed

    # Gate 1: Noise suppression
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    pred = np.where(fake_only, 0.92 * pred + 0.08 * rank_proto, pred)

    # Gate 2: Temporal continuity (fat-tailed t-distribution kernel)
    row_ids = np.arange(len(p_proto))
    offs = np.arange(-3, 4, dtype=np.float32)
    proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

    pa_ctx = p_proto.copy()
    xp = np.pad(p_proto, ((3, 3), (0, 0)), mode="edge")
    pa_ctx = sum(proto_kernel[i] * xp[i:i + len(p_proto)] for i in range(7))

    xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (
        (xctx > 0.88) & (rank_proto > 0.75) &
        (p_sed < 0.12) & (~fake_only)
    )
    pred = np.where(
        proto_cont,
        0.85 * pred + 0.15 * np.maximum(rank_proto, xctx),
        pred
    )

    # Gate 3: SED spike preservation
    sed_only = (
        (rank_sed > 0.95) & (rank_proto < 0.80) &
        (~fake_only) & (~proto_cont)
    )
    pred = np.where(sed_only, 0.88 * pred + 0.12 * rank_sed, pred)

    # Gate 4: Sonotype mirroring
    MIRROR_PAIRS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )
    col_to_idx = {l: i for i, l in enumerate(cols)}
    pred_df = pd.DataFrame(pred, columns=cols)
    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = pred_df.iloc[:, valid_idx].max(
                axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                pred_df.iloc[:, idx] = group_max
    pred = pred_df.to_numpy(np.float32)

    # Gate 5: Adaptive rare-class thresholding
    if taxonomy_path is not None and Path(taxonomy_path).exists():
        try:
            tax_df = pd.read_csv(taxonomy_path).set_index("primary_label")
            rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
            for ci, species in enumerate(cols):
                if (species in tax_df.index and
                        tax_df.loc[species, "class_name"] in rare_classes):
                    vals = pred[:, ci]
                    thr = vals.mean() + 0.05
                    pred[:, ci] = np.where(vals < thr, vals * 0.9, vals)
        except Exception as e:
            print(f"[Gate 5] Skipped: {e}")

    return pred
