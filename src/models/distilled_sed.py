"""
src/models/distilled_sed.py
----------------------------
Distilled EfficientNet Sound Event Detection (SED) model.

Architecture:
  - Backbone: EfficientNet-B0 pretrained on JFT (via timm)
  - Input: Mel spectrogram (256 mels, 32kHz, 5s windows)
  - Knowledge distillation from frozen Perch v2 embeddings
  - Clip-level and frame-level predictions averaged for final output

Used as Model_21 in the ensemble (weight=0.014) and contributes
to the xSED blend inside Model_74 (weight=0.40).

Original implementation by Tucker Arrants, modularized here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


# ---------------------------------------------------------------------------
# Mel Spectrogram Transform
# ---------------------------------------------------------------------------

class MelSpectrogramTransform(nn.Module):
    """
    Convert raw waveform to log-mel spectrogram.

    Parameters match competition settings:
      - 256 mel bins
      - 32kHz sample rate
      - 5 second windows
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 256,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            top_db=top_db
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor of shape (batch, samples)

        Returns
        -------
        torch.Tensor of shape (batch, 1, n_mels, time_frames)
            Normalized log-mel spectrogram.
        """
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        # Normalize per spectrogram
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        mel = (mel - mean) / std
        return mel.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Attention-based pooling over time frames for clip-level prediction.
    Learns to weight frames by their importance for each class.
    """

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.attention = nn.Linear(in_dim, n_classes)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, time, features)

        Returns
        -------
        tuple of (clip_logits, frame_logits)
          clip_logits: (batch, n_classes)
          frame_logits: (batch, time, n_classes)
        """
        attn_weights = torch.softmax(self.attention(x), dim=1)
        frame_logits = self.classifier(x)
        clip_logits = (attn_weights * frame_logits).sum(dim=1)
        return clip_logits, frame_logits


# ---------------------------------------------------------------------------
# Distilled EfficientNet SED
# ---------------------------------------------------------------------------

class DistilledEfficientNetSED(nn.Module):
    """
    EfficientNet-B0 based Sound Event Detection model with Perch distillation.

    Training uses two losses:
      1. Focal BCE against ground truth labels
      2. MSE distillation loss against Perch v2 embeddings

    Inference combines clip-level and frame-level predictions.
    """

    def __init__(
        self,
        n_classes: int = 234,
        backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k",
        perch_embed_dim: int = 1536,
        alpha_distill: float = 1.0,
        dropout: float = 0.3,
    ):
        """
        Parameters
        ----------
        n_classes : int
            Number of species classes.
        backbone_name : str
            timm model name for EfficientNet backbone.
        perch_embed_dim : int
            Dimension of Perch v2 embeddings for distillation head.
        alpha_distill : float
            Weight of distillation MSE loss during training.
        dropout : float
            Dropout before classification head.
        """
        super().__init__()
        import timm
        self.alpha_distill = alpha_distill

        # EfficientNet backbone (pretrained)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,        # Remove classification head
            global_pool="",       # Remove global pooling
        )
        backbone_out_dim = self.backbone.num_features

        # Temporal pooling over frequency axis
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Projection to model dimension
        self.proj = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention pooling for clip/frame prediction
        self.attn_pool = AttentionPool(512, n_classes)

        # Perch distillation head
        self.distill_head = nn.Linear(512, perch_embed_dim)

    def forward(
        self,
        mel: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        mel : torch.Tensor of shape (batch, 1, n_mels, time_frames)
            Log-mel spectrogram input.
        return_embeddings : bool
            If True, also return projected embeddings.

        Returns
        -------
        dict with keys:
          'clip_logits': (batch, n_classes)
          'frame_logits': (batch, time, n_classes)
          'embeddings': (batch, time, 512) if return_embeddings=True
        """
        # Backbone feature extraction
        feat = self.backbone.forward_features(mel)  # (B, C, H, W)

        # Pool over frequency axis
        feat = self.freq_pool(feat).squeeze(2)       # (B, C, W)
        feat = feat.permute(0, 2, 1)                 # (B, W, C)

        # Project to model dimension
        h = self.proj(feat)                          # (B, W, 512)

        # Clip and frame predictions
        clip_logits, frame_logits = self.attn_pool(h)

        out = {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
        }

        if return_embeddings:
            out["embeddings"] = h

        return out

    def predict_proba(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Get final inference probabilities combining clip and frame predictions.

        Parameters
        ----------
        mel : torch.Tensor of shape (batch, 1, n_mels, time_frames)

        Returns
        -------
        torch.Tensor of shape (batch, n_classes) — sigmoid probabilities
        """
        with torch.no_grad():
            out = self.forward(mel)
            # Average clip and max-pooled frame predictions
            clip_prob = torch.sigmoid(out["clip_logits"])
            frame_prob = torch.sigmoid(out["frame_logits"]).max(dim=1).values
            return 0.5 * clip_prob + 0.5 * frame_prob
