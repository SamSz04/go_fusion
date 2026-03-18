"""Environment modules for the GO fusion system (Phases 2-3)."""

from src.env.performance_model import (
    compute_flops,
    compute_bytes,
    estimate_cluster_runtime,
    estimate_total_runtime,
)
from src.env.fusion_rules import FusionRules
from src.env.fusion_simulator import (
    UnionFind,
    FusionCluster,
    simulate_fusion,
)
from src.env.fusion_env import FusionEnv

__all__ = [
    # Performance model
    "compute_flops",
    "compute_bytes",
    "estimate_cluster_runtime",
    "estimate_total_runtime",
    # Fusion rules
    "FusionRules",
    # Fusion simulator
    "UnionFind",
    "FusionCluster",
    "simulate_fusion",
    # RL environment
    "FusionEnv",
]
