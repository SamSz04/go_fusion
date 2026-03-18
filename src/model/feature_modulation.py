"""
Feature modulation via parameter superposition (Paper Section 4.3).

Equation:
    x^(l+1) = a^(l)(m(h_G) odot x^(l))

where:
    m(h_G) is a gating function producing per-dimension modulation weights
    from the graph-level embedding h_G, and a^(l) is a normalization layer.

This allows the global graph context to modulate per-node features at each
Transformer layer, implementing a form of parameter superposition.
"""

import torch
import torch.nn as nn


class FeatureModulation(nn.Module):
    """Modulates node features using a graph-level gating signal.

    Computes a sigmoid gate from the graph embedding h_G and applies it
    element-wise to each node's feature vector, followed by layer
    normalization.

    Args:
        dim: Feature dimension (default: 128).
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h_G: torch.Tensor) -> torch.Tensor:
        """Apply feature modulation.

        Args:
            x: Node feature matrix of shape [N, dim].
            h_G: Graph-level embedding of shape [dim].

        Returns:
            Modulated node features of shape [N, dim].
        """
        gate = torch.sigmoid(self.gate_proj(h_G))  # [dim]
        return self.norm(x * gate.unsqueeze(0))  # [N, dim]
