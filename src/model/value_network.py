"""
Value network (critic) for PPO training.

Takes node embeddings produced by the policy network, aggregates them into
a graph-level embedding via mean pooling, and predicts a scalar state value.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Value head for PPO that estimates the state value.

    Architecture:
        Mean pool node embeddings -> Linear(hidden_dim, 64) -> ReLU -> Linear(64, 1)

    Args:
        hidden_dim: Dimension of node embeddings from the policy network (default: 128).
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the scalar state value from node embeddings.

        Args:
            node_embeddings: Node embeddings from the policy network,
                             shape [N, hidden_dim].

        Returns:
            Scalar state value of shape [1].
        """
        graph_embedding = node_embeddings.mean(dim=0)  # [hidden_dim]
        return self.value_head(graph_embedding)  # [1]
