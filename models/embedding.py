"""Element, fraction (sin/cos), and composition fusion embeddings."""
import math
import torch
import torch.nn as nn


class ElementEmbedding(nn.Module):
    """
    Learnable element embeddings (shape inferred from `mat2vec_embeddings`), projected to d_model.

    The passed tensor only sets vocabulary size and feature width; weights are Xavier-initialized,
    not copied from mat2vec.
    """

    def __init__(self, mat2vec_embeddings: torch.Tensor, d_model: int, projection_type: str = "linear"):
        super().__init__()
        self.d_model = d_model
        self.d_mat2vec = mat2vec_embeddings.shape[1]
        self.projection_type = projection_type
        n_elements = mat2vec_embeddings.shape[0]
        # Indices are 1-based in data; 0 is padding.
        self.embedding = nn.Embedding(n_elements + 1, self.d_mat2vec, padding_idx=0)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.embedding.weight[1:n_elements + 1])
            self.embedding.weight[0].zero_()

        if projection_type == "linear":
            self.projection = nn.Linear(self.d_mat2vec, d_model)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(self.d_mat2vec, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

    def forward(self, element_indices: torch.Tensor) -> torch.Tensor:
        """element_indices: [batch, n_max], 0 = padding."""
        embeddings = self.embedding(element_indices)
        return self.projection(embeddings)


class FractionEmbedding(nn.Module):
    """
    Sinusoidal encoding of scaled mole fractions (treated as continuous positions).

    Expects even `d_model` (standard sin/cos layout on pairs of dimensions).
    """

    def __init__(self, d_f: int, d_model: int, epsilon: float = 0.05):
        super().__init__()
        self.d_f = d_f
        self.d_model = d_model
        self.epsilon = epsilon
        self.scale_factor = 100.0

    def forward(self, fractions: torch.Tensor) -> torch.Tensor:
        """fractions: [batch, n_max], zero on padding slots."""
        batch_size, n_max = fractions.shape
        device = fractions.device
        scaled_pos = fractions * self.scale_factor

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(batch_size, n_max, self.d_model, device=device, dtype=torch.float32)
        pos_expanded = scaled_pos.unsqueeze(-1)
        pe[:, :, 0::2] = torch.sin(pos_expanded * div_term)
        pe[:, :, 1::2] = torch.cos(pos_expanded * div_term)
        return pe


class CompositionEmbedding(nn.Module):
    """Fuse element and fraction embeddings (concat + linear + norm, or add + norm)."""

    def __init__(self, element_embedding: ElementEmbedding, fraction_embedding: FractionEmbedding,
                 d_model: int, zero_padding: bool = True, fusion_type: str = "concat"):
        super().__init__()
        self.element_embedding = element_embedding
        self.fraction_embedding = fraction_embedding
        self.zero_padding = zero_padding
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = nn.Linear(2 * d_model, d_model)
            self.post_fusion_norm = nn.LayerNorm(d_model)
        elif fusion_type == "add":
            self.element_norm = nn.LayerNorm(d_model)
            self.fraction_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be 'concat' or 'add'")

    def forward(self, element_indices: torch.Tensor, fractions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        element_emb = self.element_embedding(element_indices)
        fraction_emb = self.fraction_embedding(fractions)

        if self.fusion_type == "concat":
            concatenated = torch.cat([element_emb, fraction_emb], dim=-1)
            tokens = self.post_fusion_norm(self.fusion(concatenated))
        else:
            element_emb = self.element_norm(element_emb)
            fraction_emb = self.fraction_norm(fraction_emb)
            tokens = element_emb + fraction_emb

        if self.zero_padding and mask is not None:
            tokens = tokens * mask.unsqueeze(-1)

        return tokens
