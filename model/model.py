"""
Transformer encoder-decoder that predicts future frames of receiving hand.
Uses encoder to process input sequence, then decoder to predict future frames.
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
    def __init__(self, in_dim: int, out_dim: int, d_model: int, nhead: int, nlayers: int, ffdim: int, dropout: float, future_frames: int = 5):
        super().__init__()
        self.future_frames = future_frames
        self.proj_in = nn.Linear(in_dim, d_model)
        self.proj_out = nn.Linear(d_model, out_dim)
        self.pos = PositionalEncoding(d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffdim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        
        # Decoder for future frames
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffdim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        
        # Learnable query embeddings for future frames
        self.query_emb = nn.Parameter(torch.randn(1, future_frames, d_model))
        
        # Layer norm before output projection
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, in_dim] input sequence
        Returns:
            [B, future_frames, out_dim] predicted future frames
        """
        B = x.size(0)
        # Encode input sequence
        h = self.proj_in(x)  # [B, L, d_model]
        h = self.pos(h)
        encoded = self.encoder(h)  # [B, L, d_model]
        
        # Use last frame as context for decoder
        memory = encoded  # [B, L, d_model]
        
        # Create query embeddings for future frames
        queries = self.query_emb.expand(B, -1, -1)  # [B, future_frames, d_model]
        # Add positional encoding for future frames (offset by sequence length)
        seq_len = encoded.size(1)
        d_model = encoded.size(2)
        future_pos = torch.arange(seq_len, seq_len + self.future_frames, device=encoded.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=encoded.device).float() * 
                       (-math.log(10000.0) / d_model))
        pe_future = torch.zeros(self.future_frames, d_model, device=encoded.device)
        pe_future[:, 0::2] = torch.sin(future_pos * div)
        if d_model % 2 == 0:
            pe_future[:, 1::2] = torch.cos(future_pos * div)
        else:
            # Handle odd d_model
            pe_future[:, 1::2] = torch.cos(future_pos * div[:-1]) if len(div) > 1 else torch.zeros_like(future_pos)
        queries = queries + pe_future.unsqueeze(0)
        
        # Decode future frames
        decoded = self.decoder(queries, memory)  # [B, future_frames, d_model]
        decoded = self.ln_out(decoded)
        
        # Project to output dimension
        output = self.proj_out(decoded)  # [B, future_frames, out_dim]
        
        return output
