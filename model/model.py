"""
Minimal Transformer encoder for sequence classification.

We project input features -> d_model, add sinusoidal positions,
encode with Transformer layers, then classify using the final token.
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (classic "Attention Is All You Need" style).
    Gives the model a sense of order without training extra parameters.
    """
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                       # [L, D_model]
        pos = torch.arange(0, max_len).unsqueeze(1).float()      # [L, 1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, L, D_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D_model] -> add PE for the first L positions
        return x + self.pe[:, :x.size(1), :]

class HandoverTransformer(nn.Module):
    """
    Transformer encoder with a final linear head that emits one logit per sequence.
    """
    def __init__(self, in_dim: int, d_model: int, nhead: int, nlayers: int, ffdim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)  # project raw features to model dimension
        self.pos  = PositionalEncoding(d_model) # add positional information

        # One encoder layer configuration (we'll stack 'nlayers' of these)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffdim,
            dropout=dropout,
            batch_first=True,          # (B, L, D) interface
            activation="gelu"          # smoother than ReLU for many NLP-ish tasks
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        # Head: take the final token embedding and map to a single logit
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]  -> logit: [B]
        """
        h = self.proj(x)            # [B, L, D_model]
        h = self.pos(h)             # add sinusoidal positions
        h = self.encoder(h)         # [B, L, D_model]
        last = h[:, -1, :]          # final token corresponds to "now"
        logit = self.head(last).squeeze(-1)
        return logit                # raw scores (use sigmoid at inference)
