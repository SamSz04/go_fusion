"""
GO Fusion Policy Network with non-autoregressive iterative refinement.

Complete forward pass:
    For each refinement iteration t = 1..T:
        1. Look up opcode embeddings via nn.Embedding(132, opcode_embed_dim)
        2. Concatenate opcode embeddings, continuous features, and prev actions
        3. Project to hidden dimension
        4. Apply GraphSAGE (GNN) to capture local structure
        5. Compute graph-level embedding h_G = mean(node embeddings)
        6. Apply Segmented Transformer with per-layer feature modulation
        7. Produce action logits via policy head
        8. Detach action probabilities for next iteration

The network outputs per-node action probabilities over num_priorities classes
and node embeddings (for the value network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.model.feature_modulation import FeatureModulation
from src.model.graphsage import GraphSAGE
from src.model.segmented_transformer import SegmentedTransformer


class GOFusionPolicy(nn.Module):
    """Full GO policy network with iterative non-autoregressive refinement.

    Args:
        num_node_features: Dimension of continuous node features (default: 19).
        hidden_dim: Hidden dimension used throughout (default: 128).
        num_priorities: Number of priority classes / action space size (default: 20).
        num_opcodes: Size of the opcode vocabulary for embedding (default: 132).
        opcode_embed_dim: Dimension of the learnable opcode embedding (default: 32).
        num_gnn_layers: Number of GraphSAGE layers (default: 2).
        num_transformer_layers: Number of Segmented Transformer layers (default: 3).
        num_heads: Number of attention heads (default: 8).
        segment_size: Segment size for Transformer-XL recurrence (default: 256).
        d_ff: Feed-forward inner dimension in Transformer (default: 512).
        num_iterations: Number of refinement iterations (default: 3).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        num_node_features: int = 19,
        hidden_dim: int = 128,
        num_priorities: int = 20,
        num_opcodes: int = 132,
        opcode_embed_dim: int = 32,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        segment_size: int = 256,
        d_ff: int = 512,
        num_iterations: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_priorities = num_priorities

        # Learnable opcode embedding (replaces one-hot encoding)
        self.opcode_embed = nn.Embedding(num_opcodes, opcode_embed_dim)

        # Input projection: opcode_embed + continuous features + prev actions -> hidden_dim
        input_dim = opcode_embed_dim + num_node_features + num_priorities
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GraphSAGE with max-pool aggregation
        self.gnn = GraphSAGE(hidden_dim, hidden_dim, num_layers=num_gnn_layers)

        # Per-Transformer-layer feature modulation
        self.modulations = nn.ModuleList(
            [FeatureModulation(hidden_dim) for _ in range(num_transformer_layers)]
        )

        # Segmented Transformer-XL
        self.transformer = SegmentedTransformer(
            num_layers=num_transformer_layers,
            d_model=hidden_dim,
            nhead=num_heads,
            d_ff=d_ff,
            segment_size=segment_size,
            dropout=dropout,
        )

        # Policy head: produces logits over priority classes
        self.policy_head = nn.Linear(hidden_dim, num_priorities)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        opcode_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with iterative non-autoregressive refinement.

        Args:
            x: Continuous node feature matrix of shape [N, num_node_features].
            edge_index: Graph connectivity in COO format of shape [2, E].
            opcode_ids: Integer opcode indices of shape [N], values in [0, num_opcodes).

        Returns:
            probs: Action probabilities of shape [N, num_priorities].
            h: Node embeddings of shape [N, hidden_dim] from the final
               iteration (for the value network).
        """
        N = x.size(0)
        prev_actions = torch.zeros(N, self.num_priorities, device=x.device)

        # Look up opcode embeddings (same across iterations)
        opcode_emb = self.opcode_embed(opcode_ids)  # [N, opcode_embed_dim]

        probs = None
        h = None

        for t in range(self.num_iterations):
            # Concatenate opcode embedding, continuous features, and prev actions
            h = torch.cat([opcode_emb, x, prev_actions], dim=-1)
            h = F.relu(self.input_proj(h))  # [N, hidden_dim]

            # GNN: capture local graph structure
            h = self.gnn(h, edge_index)  # [N, hidden_dim]

            # Graph-level embedding for feature modulation
            h_G = h.mean(dim=0)  # [hidden_dim]

            # Segmented Transformer with per-layer feature modulation
            h = self.transformer.forward_with_modulation(
                h, h_G, self.modulations
            )  # [N, hidden_dim]

            # Policy head
            logits = self.policy_head(h)  # [N, num_priorities]
            probs = F.softmax(logits, dim=-1)  # [N, num_priorities]

            # Detach for next iteration
            prev_actions = probs.detach()

        return probs, h

    def get_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        opcode_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities for PPO.

        Args:
            x: Continuous node feature matrix of shape [N, num_node_features].
            edge_index: Graph connectivity in COO format of shape [2, E].
            opcode_ids: Integer opcode indices of shape [N].

        Returns:
            actions: Sampled actions of shape [N], each in [0, num_priorities).
            log_probs: Log probabilities of sampled actions, shape [N].
            entropy: Per-node entropy of the action distribution, shape [N].
            embeddings: Node embeddings of shape [N, hidden_dim].
        """
        probs, embeddings = self.forward(x, edge_index, opcode_ids)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy, embeddings
