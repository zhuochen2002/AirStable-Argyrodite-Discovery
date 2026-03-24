"""Full formation-energy model: composition embedding, Transformer, per-atom head, fraction-weighted sum."""
import torch
import torch.nn as nn

from models.embedding import ElementEmbedding, FractionEmbedding, CompositionEmbedding
from models.transformer import TransformerEncoder


class FormationEnergyModel(nn.Module):
    """
    Encoder over composition tokens; per-position scalar head; prediction = sum(fraction * contribution).
    """

    def __init__(self, config: dict, mat2vec_embeddings: torch.Tensor):
        super().__init__()
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.d_ff = config.get('d_ff') or (4 * config['d_model'])
        self.dropout = config['dropout']
        self.d_f = config['d_f']
        self.epsilon = config['epsilon']
        self.n_max = config['n_max']

        self.element_embedding = ElementEmbedding(
            mat2vec_embeddings,
            self.d_model,
            projection_type="linear"
        )
        self.fraction_embedding = FractionEmbedding(
            self.d_f,
            self.d_model,
            self.epsilon
        )
        self.composition_embedding = CompositionEmbedding(
            self.element_embedding,
            self.fraction_embedding,
            d_model=self.d_model,
            zero_padding=True,
            fusion_type="concat"
        )
        self.encoder = TransformerEncoder(
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.dropout
        )
        self.pre_head_norm = nn.LayerNorm(self.d_model)

        element_head_n_layers = config.get('element_head_n_layers', 1)
        d_hidden = config.get('element_head_hidden') or self.d_model

        element_head_layers = []
        element_head_layers.append(nn.Linear(self.d_model, d_hidden))
        element_head_layers.append(nn.GELU())
        if element_head_n_layers > 1:
            element_head_layers.append(nn.Dropout(self.dropout))
        for _ in range(element_head_n_layers - 1):
            element_head_layers.append(nn.Linear(d_hidden, d_hidden))
            element_head_layers.append(nn.GELU())
            element_head_layers.append(nn.Dropout(self.dropout))
        element_head_layers.append(nn.Linear(d_hidden, 1))
        self.element_head = nn.Sequential(*element_head_layers)

    def forward(self, element_indices: torch.Tensor, fractions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            element_indices: [batch, n_max], 0 = padding
            fractions: [batch, n_max]
            mask: [batch, n_max], 1 = real atom slot

        Returns:
            formation_energy: [batch]
        """
        tokens = self.composition_embedding(element_indices, fractions, mask)
        encoded = self.encoder(tokens, mask)
        encoded_norm = self.pre_head_norm(encoded)
        element_contributions = self.element_head(encoded_norm).squeeze(-1)
        formation_energy = (fractions * element_contributions).sum(dim=1)
        return formation_energy

    def get_attention_weights(
        self, element_indices: torch.Tensor, fractions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Last-layer attention averaged over heads: [batch, n_max, n_max]."""
        tokens = self.composition_embedding(element_indices, fractions, mask)
        _, last_attn = self.encoder(tokens, mask, return_last_attention=True)
        return last_attn.mean(dim=1)

    def get_attention_weights_per_head(
        self, element_indices: torch.Tensor, fractions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Last-layer attention per head: [batch, n_heads, n_max, n_max]."""
        tokens = self.composition_embedding(element_indices, fractions, mask)
        _, last_attn = self.encoder(tokens, mask, return_last_attention=True)
        return last_attn

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
