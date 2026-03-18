"""
GraphSAGE with max-pool aggregation (Hamilton et al., 2017 Section 4.1).

Equations:
    h_N(v)^(l) = max(sigma(W_pool^(l) * h_u^(l) + b_pool^(l)), for all u in N(v))
    h_v^(l+1) = sigma(W^(l+1) * concat(h_v^(l), h_N(v)^(l)))

This is a custom implementation using PyTorch Geometric's MessagePassing base
class with aggr='max', NOT the built-in SAGEConv (which defaults to mean).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class GraphSAGEMaxPoolLayer(MessagePassing):
    """Single GraphSAGE layer with max-pool aggregation.

    Each neighbor embedding is transformed through a linear + ReLU before
    element-wise max aggregation, then the self-embedding is concatenated
    with the aggregated neighborhood and projected to the output dimension.

    Args:
        in_channels: Dimension of input node features.
        out_channels: Dimension of output node features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="max")
        # Transform neighbors before max-pool: W_pool * h_u + b_pool
        self.lin_pool = nn.Linear(in_channels, in_channels)
        # Projection after concat(self, aggregated_neighbors)
        self.lin_update = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature matrix of shape [N, in_channels].
            edge_index: Graph connectivity in COO format of shape [2, E].

        Returns:
            Updated node features of shape [N, out_channels].
        """
        # Aggregate neighbor messages via max-pool
        neigh_agg = self.propagate(edge_index, x=x)  # [N, in_channels]
        # Concatenate self-features with aggregated neighbor features
        out = torch.cat([x, neigh_agg], dim=-1)  # [N, 2 * in_channels]
        return F.relu(self.lin_update(out))  # [N, out_channels]

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Construct messages from neighbor nodes.

        Applies the pooling transform sigma(W_pool * h_u + b_pool) to each
        neighbor embedding before the max aggregation.

        Args:
            x_j: Features of source (neighbor) nodes, shape [E, in_channels].

        Returns:
            Transformed neighbor features of shape [E, in_channels].
        """
        return F.relu(self.lin_pool(x_j))


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE model with max-pool aggregation.

    Architecture:
        Input projection: Linear(num_node_features, hidden_dim)
        Layer 1: GraphSAGEMaxPoolLayer(hidden_dim, hidden_dim)
        Layer 2: GraphSAGEMaxPoolLayer(hidden_dim, hidden_dim)

    Args:
        num_node_features: Dimension of raw input node features.
        hidden_dim: Hidden dimension used throughout the GNN layers.
        num_layers: Number of GraphSAGE layers (default: 2).
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphSAGEMaxPoolLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GraphSAGE model.

        Args:
            x: Node feature matrix of shape [N, num_node_features].
            edge_index: Graph connectivity in COO format of shape [2, E].

        Returns:
            Node embeddings of shape [N, hidden_dim].
        """
        x = F.relu(self.input_proj(x))  # [N, hidden_dim]
        for layer in self.layers:
            x = layer(x, edge_index)  # [N, hidden_dim]
        return x
