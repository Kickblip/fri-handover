"""
Transformer encoder that emits one logit per sequence.
The final token represents "now"; classify using that embedding.
"""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class HandoverTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, nlayers: int, ffdim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffdim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))  # one logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)  # [B]
