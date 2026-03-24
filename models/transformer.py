"""Transformer encoder stack with padding-aware self-attention."""
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """Self-attention and FFN sublayers with post-LayerNorm and residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, need_weights: bool = False
    ):
        """
        Args:
            x: [batch, n_max, d_model]
            mask: [batch, n_max], 1 = real token, 0 = padding (ignored in attention)
            need_weights: if True, return (output, attn_weights)

        Returns:
            output, or (output, attn_weights) with shape [batch, n_heads, n_max, n_max]
            when need_weights=True.
        """
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        attn_output, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        if need_weights:
            return x, attn_weights
        return x


class TransformerEncoder(nn.Module):
    """Stack of `TransformerEncoderLayer`."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_last_attention: bool = False
    ):
        """
        Args:
            x: [batch, n_max, d_model]
            mask: [batch, n_max]
            return_last_attention: if True, also return last layer attention weights

        Returns:
            x, or (x, last_attn_weights) with last_attn_weights [batch, n_heads, n_max, n_max].
        """
        last_attn_weights = None
        n_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            need_weights = return_last_attention and (i == n_layers - 1)
            if need_weights:
                x, last_attn_weights = layer(x, mask, need_weights=True)
            else:
                x = layer(x, mask, need_weights=False)

        if return_last_attention:
            return x, last_attn_weights
        return x
