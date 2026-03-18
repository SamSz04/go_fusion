"""
Segmented Transformer-XL for processing computation graphs (Paper Section 4.2).

Key design choices:
    - NO positional encoding: graph topology is already captured by the GNN.
    - Segment-level recurrence from Transformer-XL: previous segment's hidden
      states are cached (detached from the computation graph) and used as
      extended context for the current segment's attention computation.
    - Feature modulation is applied per layer between attention and the next
      layer, allowing graph-level context to influence per-node representations.

Default configuration: 3 layers, 8 heads, d_model=128, d_ff=512, segment_size=256.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.model.feature_modulation import FeatureModulation


class SegmentedTransformerLayer(nn.Module):
    """Single Transformer layer with segment-level recurrence.

    Implements pre-norm Transformer with extended key-value context from the
    previous segment (Transformer-XL style recurrence).

    Args:
        d_model: Model dimension (default: 128).
        nhead: Number of attention heads (default: 8).
        d_ff: Feed-forward inner dimension (default: 512).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional segment-level recurrence.

        Args:
            x: Current segment features of shape [S, d_model].
            memory: Previous segment's hidden states of shape [S_prev, d_model],
                    detached from the computation graph. None for the first segment.

        Returns:
            Updated features of shape [S, d_model].
        """
        # Build extended key-value context by prepending memory
        if memory is not None:
            kv = torch.cat([memory, x], dim=0)  # [S_prev + S, d_model]
        else:
            kv = x  # [S, d_model]

        # Self-attention: Q = current segment, K = V = extended context
        # Add batch dimension for nn.MultiheadAttention (batch_first=True)
        attn_out, _ = self.self_attn(
            x.unsqueeze(0), kv.unsqueeze(0), kv.unsqueeze(0)
        )  # [1, S, d_model]
        x = self.norm1(x + self.dropout(attn_out.squeeze(0)))  # [S, d_model]
        x = self.norm2(x + self.dropout(self.ff(x)))  # [S, d_model]
        return x


class SegmentedTransformer(nn.Module):
    """Segmented Transformer-XL with per-layer feature modulation.

    Processes nodes in fixed-size segments with recurrence: each segment
    attends to both itself and the cached hidden states from the previous
    segment. This enables long-range dependencies without quadratic cost
    in the full sequence length.

    The last segment may be shorter than segment_size when N is not evenly
    divisible; this is handled transparently.

    Args:
        num_layers: Number of Transformer layers (default: 3).
        d_model: Model dimension (default: 128).
        nhead: Number of attention heads (default: 8).
        d_ff: Feed-forward inner dimension (default: 512).
        segment_size: Number of nodes per segment (default: 256).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        d_ff: int = 512,
        segment_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.segment_size = segment_size
        self.layers = nn.ModuleList(
            [
                SegmentedTransformerLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without feature modulation.

        Args:
            x: Node features of shape [N, d_model].

        Returns:
            Transformed node features of shape [N, d_model].
        """
        return self._process_segments(x, modulations=None, h_G=None)

    def forward_with_modulation(
        self,
        x: torch.Tensor,
        h_G: torch.Tensor,
        modulations: nn.ModuleList,
    ) -> torch.Tensor:
        """Forward pass with per-layer feature modulation.

        Feature modulation (parameter superposition) is applied after each
        Transformer layer, using the graph-level embedding h_G as the
        modulation signal.

        Args:
            x: Node features of shape [N, d_model].
            h_G: Graph-level embedding of shape [d_model].
            modulations: ModuleList of FeatureModulation modules, one per layer.

        Returns:
            Transformed node features of shape [N, d_model].
        """
        return self._process_segments(x, modulations=modulations, h_G=h_G)

    def _process_segments(
        self,
        x: torch.Tensor,
        modulations: Optional[nn.ModuleList],
        h_G: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Core segmented processing with Transformer-XL recurrence.

        Splits the input into segments of size segment_size, processes each
        segment through all layers while maintaining per-layer memory from
        the previous segment.

        Args:
            x: Node features of shape [N, d_model].
            modulations: Optional ModuleList of FeatureModulation modules.
            h_G: Optional graph-level embedding of shape [d_model].

        Returns:
            Transformed node features of shape [N, d_model].
        """
        N = x.size(0)
        S = self.segment_size

        # Split input into segments; last segment may be shorter
        segments = [x[i : i + S] for i in range(0, N, S)]
        num_segments = len(segments)

        # Per-layer memory from the previous segment (initially None)
        memories: List[Optional[torch.Tensor]] = [None] * self.num_layers

        output_segments = []

        for seg_idx in range(num_segments):
            h = segments[seg_idx]  # [S_cur, d_model] (S_cur <= S)

            new_memories: List[Optional[torch.Tensor]] = []

            for layer_idx in range(self.num_layers):
                layer = self.layers[layer_idx]
                memory = memories[layer_idx]

                h = layer(h, memory=memory)  # [S_cur, d_model]

                # Apply feature modulation after each layer if provided
                if modulations is not None and h_G is not None:
                    h = modulations[layer_idx](h, h_G)  # [S_cur, d_model]

                # Cache this segment's output (detached) as memory for the
                # next segment at this layer
                new_memories.append(h.detach())

            memories = new_memories
            output_segments.append(h)

        # Concatenate all segment outputs back into a single tensor
        return torch.cat(output_segments, dim=0)  # [N, d_model]
