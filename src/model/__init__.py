"""GO Fusion neural network components.

Modules:
    graphsage: GraphSAGE with max-pool aggregation (Paper Section 4.1)
    feature_modulation: Parameter superposition via gated modulation (Paper Section 4.3)
    segmented_transformer: Segmented Transformer-XL with recurrence (Paper Section 4.2)
    policy_network: Full GO policy network with iterative refinement
    value_network: Value head (critic) for PPO
"""

from src.model.feature_modulation import FeatureModulation
from src.model.graphsage import GraphSAGE, GraphSAGEMaxPoolLayer
from src.model.policy_network import GOFusionPolicy
from src.model.segmented_transformer import (
    SegmentedTransformer,
    SegmentedTransformerLayer,
)
from src.model.value_network import ValueNetwork

__all__ = [
    "GraphSAGEMaxPoolLayer",
    "GraphSAGE",
    "FeatureModulation",
    "SegmentedTransformerLayer",
    "SegmentedTransformer",
    "GOFusionPolicy",
    "ValueNetwork",
]
