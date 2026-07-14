"""
src/models/proto_ssm.py
-----------------------
Prototype State Space Model (ProtoSSM) architecture.

ProtoSSM is a temporal sequence model that combines:
  - State Space Model (SSM) layers for temporal modeling
  - Prototype-based classification heads
  - Optional cross-attention between windows
  - Site and time-of-day metadata embeddings

Used in both Model_21 (v5, larger) and Model_74 (v2, standard).

Original implementation by yukiZ (hideyukizushi) and Yaroslav Kholmirzayev,
adapted and modularized here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# SSM Layer
# ---------------------------------------------------------------------------

class SSMLayer(nn.Module):
    """
    Single State Space Model layer with linear recurrence.

    Models temporal dependencies across the 12 windows of a 60s soundscape.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        d_model : int
            Input/output feature dimension.
        d_state : int
            Hidden state dimension of the SSM.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor of shape (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        A = torch.tanh(self.A)
        for t in range(T):
            h = h @ A.T + x[:, t, :] @ self.B
            y = h @ self.C + x[:, t, :] * self.D
            outputs.append(y)

        out = torch.stack(outputs, dim=1)
        out = self.dropout(self.proj(out))
        return self.norm(x + out)


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention between windows within a file.
    Allows each window to attend to all other windows in the same recording.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# ProtoSSM v2
# ---------------------------------------------------------------------------

class ProtoSSMv2(nn.Module):
    """
    ProtoSSM v2 — used in Model_74 (Karnakbayev pipeline).

    Architecture:
      1. Input projection: d_input -> d_model
      2. Site + hour metadata embeddings
      3. Stack of SSM layers
      4. Optional cross-attention
      5. Prototype similarity head -> species logits
      6. Optional taxonomic family auxiliary head
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_state: int,
        n_ssm_layers: int,
        n_classes: int,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
    ):
        """
        Parameters
        ----------
        d_input : int
            Input feature dimension (Perch embedding dim, typically 1536).
        d_model : int
            Internal model dimension.
        d_state : int
            SSM hidden state dimension.
        n_ssm_layers : int
            Number of stacked SSM layers.
        n_classes : int
            Number of output species classes (234).
        n_windows : int
            Number of 5s windows per file (12 for 60s files).
        dropout : float
            Dropout probability.
        n_sites : int
            Number of unique recording sites for metadata embedding.
        meta_dim : int
            Dimension of site/hour metadata embeddings.
        use_cross_attn : bool
            Whether to apply cross-attention after SSM layers.
        cross_attn_heads : int
            Number of attention heads in cross-attention.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Metadata embeddings
        self.site_emb = nn.Embedding(n_sites + 1, meta_dim, padding_idx=0)
        self.hour_emb = nn.Embedding(25, meta_dim, padding_idx=0)
        self.meta_proj = nn.Linear(meta_dim * 2, d_model)

        # SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(d_model, d_state, dropout)
            for _ in range(n_ssm_layers)
        ])

        # Cross-attention
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                d_model, cross_attn_heads, dropout)

        # Learnable fusion weight between SSM output and input logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # Prototype temperature
        self.proto_temp = nn.Parameter(torch.zeros(1))

        # Prototypes (initialized from data)
        self.prototypes = nn.Parameter(
            torch.randn(n_classes, d_model) * 0.01
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Family auxiliary head (initialized later)
        self.family_head = None

    def init_prototypes_from_data(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Initialize prototype vectors as class-conditional mean embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor of shape (n_samples, d_input)
        labels : torch.Tensor of shape (n_samples, n_classes), multi-hot
        """
        with torch.no_grad():
            proj = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0
                if mask.sum() > 0:
                    self.prototypes.data[c] = proj[mask].mean(0)

    def init_family_head(
        self,
        n_families: int,
        class_to_family: dict,
    ) -> None:
        """
        Initialize the taxonomic family auxiliary classification head.

        Parameters
        ----------
        n_families : int
            Number of taxonomic families.
        class_to_family : dict
            Mapping from class index to family index.
        """
        self.family_head = nn.Linear(self.d_model, n_families)
        self.class_to_family = class_to_family

    def forward(
        self,
        x: torch.Tensor,
        logits_in: torch.Tensor,
        site_ids: torch.Tensor = None,
        hours: torch.Tensor = None,
    ) -> tuple:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_windows, d_input)
            Perch embeddings.
        logits_in : torch.Tensor of shape (batch, n_windows, n_classes)
            Raw Perch logits for knowledge distillation.
        site_ids : torch.Tensor of shape (batch,), optional
        hours : torch.Tensor of shape (batch,), optional

        Returns
        -------
        tuple of (species_logits, family_logits, proto_sim)
          species_logits: (batch, n_windows, n_classes)
          family_logits: (batch, n_windows, n_families) or None
          proto_sim: (batch, n_windows, n_classes)
        """
        B, T, _ = x.shape

        # Project input
        h = self.input_proj(x)

        # Add metadata
        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(
                0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(
                torch.cat([site_e, hour_e], dim=-1)
            ).unsqueeze(1).expand(-1, T, -1)
            h = h + meta

        # SSM layers
        for layer in self.ssm_layers:
            h = layer(h)

        # Cross-attention
        if self.use_cross_attn:
            h = self.cross_attn(h)

        # Prototype similarity
        proto = F.normalize(self.prototypes, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        temp = F.softplus(self.proto_temp) + 0.5
        proto_sim = torch.einsum("btd,cd->btc", h_norm, proto) / temp

        # Direct output head
        direct_out = self.output_head(h)

        # Fused output: learnable blend of prototype sim and direct head
        alpha = torch.sigmoid(self.fusion_alpha)
        species_out = alpha * proto_sim + (1 - alpha) * direct_out

        # Fusion with input logits
        species_out = species_out + \
            torch.sigmoid(self.fusion_alpha) * logits_in

        # Family auxiliary head
        family_out = None
        if self.family_head is not None:
            family_out = self.family_head(h)

        return species_out, family_out, proto_sim
