"""
src/inference.py
----------------
Inference pipeline for BirdCLEF+ 2026 deployment.

Takes a raw audio file and returns top species predictions.

Pipeline:
  1. Load and segment audio into 12 x 5s windows
  2. Run Perch v2 ONNX -> 1536-dim embeddings + logits
  3. Run ProtoSSM -> species logits
  4. Run ResidualSSM -> corrected logits
  5. Apply sigmoid + rank -> top species
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
FILE_SAMPLES = 60 * SR
N_CLASSES = 234


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path, target_sr: int = SR) -> np.ndarray:
    """
    Load an audio file and return a 60-second mono waveform at target_sr.

    Parameters
    ----------
    path : str or Path
    target_sr : int

    Returns
    -------
    np.ndarray of shape (FILE_SAMPLES,)
    """
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=True, duration=60.0)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32)


def segment_audio(y: np.ndarray) -> np.ndarray:
    """
    Split a 60-second waveform into 12 x 5-second windows.

    Parameters
    ----------
    y : np.ndarray of shape (FILE_SAMPLES,)

    Returns
    -------
    np.ndarray of shape (N_WINDOWS, WINDOW_SAMPLES)
    """
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)


# ---------------------------------------------------------------------------
# Perch ONNX inference
# ---------------------------------------------------------------------------

def load_perch_session(onnx_path: str):
    """
    Load the Perch v2 ONNX model.

    Parameters
    ----------
    onnx_path : str

    Returns
    -------
    tuple of (session, input_name, output_map)
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    return session, input_name, output_map


def run_perch(
    windows: np.ndarray,
    session,
    input_name: str,
    output_map: dict,
) -> tuple:
    """
    Run Perch v2 on audio windows.

    Parameters
    ----------
    windows : np.ndarray of shape (N_WINDOWS, WINDOW_SAMPLES)
    session : onnxruntime.InferenceSession
    input_name : str
    output_map : dict

    Returns
    -------
    tuple of (logits, embeddings)
      logits: (N_WINDOWS, n_perch_classes)
      embeddings: (N_WINDOWS, 1536)
    """
    outs = session.run(None, {input_name: windows})
    logits = outs[output_map["label"]].astype(np.float32)
    embeddings = outs[output_map["embedding"]].astype(np.float32)
    return logits, embeddings


# ---------------------------------------------------------------------------
# ProtoSSM model (matches LightProtoSSM from competition)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv,
                                padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(
            1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))
                        [:, :, :T].transpose(1, 2))
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16, n_classes=234,
                 n_windows=12, dropout=0.15, n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, cross_attn_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(2)
            ])
            self.cross_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([
                self.site_emb(site_ids),
                self.hour_emb(hours)
            ], dim=-1))
            h = h + meta[:, None, :]
        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            res = h
            hf = fwd(h)
            hb = bwd(h.flip(1)).flip(1)
            h = self.drop(merge(torch.cat([hf, hb], dim=-1)))
            h = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (torch.matmul(h_n, p_n.T) *
               F.softplus(self.proto_temp) + self.class_bias[None, None, :])
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out


# ---------------------------------------------------------------------------
# ResidualSSM model
# ---------------------------------------------------------------------------

class ResidualSSM(nn.Module):
    def __init__(self, d_input=1536, d_scores=234, d_model=128, d_state=16,
                 n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([
                self.site_emb(site_ids.clamp(
                    0, self.site_emb.num_embeddings - 1)),
                self.hour_emb(hours.clamp(0, 23))
            ], dim=-1))
            h = h + meta.unsqueeze(1)
        res = h
        hf = self.ssm_fwd(h)
        hb = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([hf, hb], dim=-1)))
        h = self.ssm_norm(h + res)
        return self.output_head(h)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_models(weights_dir: str):
    """
    Load all model weights from a directory.

    Parameters
    ----------
    weights_dir : str or Path

    Returns
    -------
    dict with keys: proto_ssm, res_ssm, site2i
    """
    weights_dir = Path(weights_dir)

    # Load site mapping
    site2i_path = weights_dir / "site2i_74.json"
    with open(site2i_path) as f:
        site2i = json.load(f)

    # Load ProtoSSM
    proto_model = LightProtoSSM(n_classes=N_CLASSES, n_sites=20)
    proto_state = torch.load(
        weights_dir / "proto_ssm_74.pt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(proto_state, dict) and "model_state_dict" in proto_state:
        proto_state = proto_state["model_state_dict"]
    proto_model.load_state_dict(proto_state, strict=False)
    proto_model.eval()

    # Load ResidualSSM
    res_model = ResidualSSM(n_classes=N_CLASSES)
    res_state = torch.load(
        weights_dir / "residual_ssm_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(res_state, dict) and "model_state_dict" in res_state:
        res_state = res_state["model_state_dict"]
    res_model.load_state_dict(res_state, strict=False)
    res_model.eval()

    # Load taxonomy for common names
    taxonomy_path = weights_dir / "taxonomy.csv"
    taxonomy = {}
    if taxonomy_path.exists():
        import pandas as pd
        tax_df = pd.read_csv(taxonomy_path)
        for _, row in tax_df.iterrows():
            taxonomy[str(row["primary_label"])] = {
                "common_name": str(row.get("common_name", row["primary_label"])),
                "scientific_name": str(row.get("scientific_name", "")),
            }

    return {
        "proto_ssm": proto_model,
        "res_ssm": res_model,
        "site2i": site2i,
        "taxonomy": taxonomy,
    }


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def predict(
    audio_path: str,
    perch_session,
    perch_input_name: str,
    perch_output_map: dict,
    models: dict,
    primary_labels: list,
    top_k: int = 10,
) -> list:
    """
    Run full inference pipeline on an audio file.

    Parameters
    ----------
    audio_path : str
    perch_session : onnxruntime.InferenceSession
    perch_input_name : str
    perch_output_map : dict
    models : dict — output of load_models()
    primary_labels : list of str
    top_k : int — number of top species to return

    Returns
    -------
    list of dicts with keys: species, confidence, window
    """
    # Load and segment audio
    y = load_audio(audio_path)
    windows = segment_audio(y)

    # Perch inference
    perch_logits, embeddings = run_perch(
        windows, perch_session, perch_input_name, perch_output_map
    )

    # Convert to tensors — shape (1, N_WINDOWS, dim)
    emb_t = torch.tensor(embeddings[None], dtype=torch.float32)
    log_t = torch.tensor(
        perch_logits[None, :, :N_CLASSES], dtype=torch.float32)

    # ProtoSSM
    with torch.no_grad():
        proto_out = models["proto_ssm"](
            emb_t, log_t,
            site_ids=torch.zeros(1, dtype=torch.long),
            hours=torch.zeros(1, dtype=torch.long),
        )

    # ResidualSSM
    with torch.no_grad():
        res_correction = models["res_ssm"](
            emb_t, proto_out,
            site_ids=torch.zeros(1, dtype=torch.long),
            hours=torch.zeros(1, dtype=torch.long),
        )

    # Final scores
    final_scores = (proto_out + 0.30 * res_correction).squeeze(0).numpy()
    probs = sigmoid(final_scores)  # (N_WINDOWS, N_CLASSES)

    # Aggregate across windows — take max per species
    species_probs = probs.max(axis=0)  # (N_CLASSES,)

    # Get top-k species
    top_indices = np.argsort(species_probs)[::-1][:top_k]
    taxonomy = models.get("taxonomy", {})
    results = [
        {
            "species": taxonomy.get(primary_labels[i], {}).get("common_name", primary_labels[i]),
            "code": primary_labels[i],
            "scientific": taxonomy.get(primary_labels[i], {}).get("scientific_name", ""),
            "confidence": float(round(species_probs[i], 4)),
        }
        for i in top_indices
        if species_probs[i] > 0.01
    ]
    return results
