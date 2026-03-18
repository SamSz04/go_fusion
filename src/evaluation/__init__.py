"""Evaluation module: baselines and metrics for GO fusion comparison."""

from src.evaluation.baselines import (
    no_fusion_baseline,
    random_priority_baseline,
    greedy_memory_baseline,
    xla_default_fusion_baseline,
    simulated_annealing_baseline,
    run_all_baselines,
)
from src.evaluation.metrics import FusionMetrics

__all__ = [
    "no_fusion_baseline",
    "random_priority_baseline",
    "greedy_memory_baseline",
    "xla_default_fusion_baseline",
    "simulated_annealing_baseline",
    "run_all_baselines",
    "FusionMetrics",
]
