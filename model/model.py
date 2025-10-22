"""
Transformer model for predicting handover events from Rodrigues sequences.
"""

import torch
import torch.nn as nn
import math

# ---------- Positional encoding (adds sense of time to transformer input) ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # store as non-trainable tensor

    def forward(self, x):
        # Add positional encodings to the input sequence
        return x + self.pe[:, :x.size(1)]

# ---------- Transformer Encoder ----------
class HandoverTransformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, nlayers, ffdim, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)  # project input features to d_model
        self.pos = PositionalEncoding(d_model)

        # stack multiple transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffdim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        # final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: [B, L, D_in]
        returns: [B] logits (before sigmoid)
        """
        h = self.embed(x)
        h = self.pos(h)
        h = self.encoder(h)
        out = self.head(h[:, -1, :]).squeeze(-1)  # predict using last frame token
        return out
