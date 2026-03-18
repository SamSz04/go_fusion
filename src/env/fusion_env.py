"""
Gymnasium-compatible RL environment for HLO operator fusion.

The agent observes a PyG graph representation of an HLO module and outputs
a priority score for every fusable node.  The environment simulates
priority-based fusion, estimates runtime via the roofline model, and
returns a reward that encourages shorter total execution time.

Reward design::

    reward = -sqrt(runtime / baseline_runtime)

A penalty of ``-10`` is applied for invalid fusion configurations (e.g.
cycles or constraint violations caught during simulation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # allow import even without PyG installed
    PyGData = Any  # type: ignore[assignment,misc]

from src.hlo_parser.hlo_ir import HloInstruction, HloComputation
from src.utils.gpu_specs import GPUSpecs, a100_specs
from src.env.fusion_rules import FusionRules
from src.env.fusion_simulator import simulate_fusion, FusionCluster
from src.env.performance_model import estimate_total_runtime


# ======================================================================
# Constants
# ======================================================================

_INVALID_PENALTY: float = -10.0
_DEFAULT_NUM_PRIORITIES: int = 20


# ======================================================================
# Environment
# ======================================================================

class FusionEnv:
    """RL environment for learning fusion priority assignments.

    Follows the Gymnasium ``reset`` / ``step`` contract (without subclassing
    ``gymnasium.Env`` so that the dependency is optional).

    Lifecycle::

        env = FusionEnv(graph_data, instructions, edges, gpu_specs)
        obs = env.reset()
        obs, reward, done, info = env.step(priority_actions)

    Attributes:
        graph_data: PyG ``Data`` object encoding the HLO graph.
        instructions: List of ``HloInstruction`` objects.
        instruction_map: Dict mapping instruction name to object.
        edges: List of ``(producer_name, consumer_name)`` tuples.
        gpu_specs: Target GPU hardware parameters.
        num_priorities: Number of discrete priority buckets.
        baseline_runtime: Estimated runtime with no fusion (each node is its
            own kernel).
        fusable_mask: Boolean tensor indicating which nodes are fusable.
        computation_map: Optional computation map for FLOP estimation.
    """

    def __init__(
        self,
        hlo_graph_data: PyGData,
        instructions: List[HloInstruction],
        edges: List[Tuple[str, str]],
        gpu_specs: Optional[GPUSpecs] = None,
        num_priorities: int = _DEFAULT_NUM_PRIORITIES,
        computation_map: Optional[Dict[str, HloComputation]] = None,
    ) -> None:
        self.graph_data = hlo_graph_data
        self.instructions = instructions
        self.instruction_map: Dict[str, HloInstruction] = {
            inst.name: inst for inst in instructions
        }
        self.edges = edges
        self.gpu_specs = gpu_specs or a100_specs()
        self.num_priorities = num_priorities
        self.computation_map = computation_map

        # Build fusable mask (aligned with self.instructions ordering).
        self.fusable_indices: List[int] = []
        fusable_flags: List[bool] = []
        for idx, inst in enumerate(self.instructions):
            is_f = FusionRules.is_fusable(inst)
            fusable_flags.append(is_f)
            if is_f:
                self.fusable_indices.append(idx)
        self.fusable_mask = torch.tensor(fusable_flags, dtype=torch.bool)
        self.num_fusable = int(self.fusable_mask.sum().item())

        # Compute baseline runtime (no fusion: every node is its own cluster).
        self.baseline_runtime = self._compute_baseline_runtime()

        # Episode state.
        self._done = False
        self._last_runtime: Optional[float] = None

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------

    def reset(self) -> PyGData:
        """Reset the environment and return the initial observation.

        The observation is the PyG graph data object (unchanged across
        resets since the graph topology is fixed for a given HLO module).

        Returns:
            The PyG ``Data`` object representing the HLO graph.
        """
        self._done = False
        self._last_runtime = None
        return self.graph_data

    def step(
        self,
        priority_actions: Tensor,
    ) -> Tuple[PyGData, float, bool, Dict[str, Any]]:
        """Execute one step: assign priorities, simulate fusion, compute reward.

        Args:
            priority_actions: A 1-D tensor of length ``num_fusable`` (or
                ``len(instructions)``) containing priority scores.  If the
                tensor has length ``len(instructions)``, only the fusable
                entries (as indicated by ``fusable_mask``) are used.

        Returns:
            A 4-tuple ``(observation, reward, done, info)`` where:

            * *observation* is the (unchanged) PyG graph.
            * *reward* is ``-sqrt(runtime / baseline_runtime)`` on success,
              or ``_INVALID_PENALTY`` on failure.
            * *done* is always ``True`` (single-step episode).
            * *info* is a dict with diagnostic data.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )

        self._done = True
        info: Dict[str, Any] = {}

        # -- Unpack priorities --
        priorities = self._unpack_priorities(priority_actions)
        info["priorities"] = priorities

        # -- Simulate fusion --
        try:
            clusters = simulate_fusion(
                instructions=self.instructions,
                edges=self.edges,
                priorities=priorities,
            )
        except Exception as exc:
            info["error"] = str(exc)
            return self.graph_data, _INVALID_PENALTY, True, info

        # -- Estimate runtime --
        cluster_sets: List[Set[str]] = [c.members for c in clusters]
        try:
            runtime = estimate_total_runtime(
                clusters=cluster_sets,
                instruction_map=self.instruction_map,
                gpu_specs=self.gpu_specs,
                computation_map=self.computation_map,
            )
        except Exception as exc:
            info["error"] = str(exc)
            return self.graph_data, _INVALID_PENALTY, True, info

        self._last_runtime = runtime
        info["runtime"] = runtime
        info["baseline_runtime"] = self.baseline_runtime
        info["num_clusters"] = len(clusters)
        info["cluster_sizes"] = [c.size for c in clusters]

        # -- Compute reward --
        if self.baseline_runtime > 0:
            ratio = runtime / self.baseline_runtime
            reward = -math.sqrt(ratio)
        else:
            # Degenerate case: baseline is zero (empty graph).
            reward = 0.0

        info["reward"] = reward
        return self.graph_data, reward, True, info

    # ------------------------------------------------------------------
    # Observation / action space metadata (Gymnasium-compatible)
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the HLO graph."""
        return len(self.instructions)

    @property
    def action_dim(self) -> int:
        """Dimensionality of the priority action vector."""
        return self.num_fusable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_priorities(self, actions: Tensor) -> Dict[str, float]:
        """Convert a priority tensor into a name->priority dict.

        Accepts either:
        * A tensor of length ``num_fusable`` (only fusable nodes).
        * A tensor of length ``num_nodes`` (priorities for all nodes;
          non-fusable entries are ignored).

        Non-fusable nodes receive a priority of ``-inf`` so that they are
        never selected for fusion.
        """
        actions_flat = actions.detach().cpu().float().flatten()
        priorities: Dict[str, float] = {}

        if actions_flat.shape[0] == self.num_fusable:
            # Compact form: one entry per fusable node.
            for i, idx in enumerate(self.fusable_indices):
                inst = self.instructions[idx]
                priorities[inst.name] = float(actions_flat[i].item())
        elif actions_flat.shape[0] == len(self.instructions):
            # Full form: one entry per node.
            for idx, inst in enumerate(self.instructions):
                if FusionRules.is_fusable(inst):
                    priorities[inst.name] = float(actions_flat[idx].item())
        else:
            raise ValueError(
                f"priority_actions has length {actions_flat.shape[0]}, "
                f"expected {self.num_fusable} (fusable nodes) or "
                f"{len(self.instructions)} (all nodes)."
            )

        # Assign -inf to non-fusable nodes.
        for inst in self.instructions:
            if inst.name not in priorities:
                priorities[inst.name] = float("-inf")

        return priorities

    def _compute_baseline_runtime(self) -> float:
        """Estimate runtime when every instruction is its own kernel (no fusion)."""
        singleton_clusters: List[Set[str]] = [
            {inst.name} for inst in self.instructions
            if FusionRules.is_fusable(inst)
        ]
        # Non-fusable instructions (parameter, constant) don't generate
        # kernels, so they are excluded from the baseline.
        if not singleton_clusters:
            return 0.0
        return estimate_total_runtime(
            clusters=singleton_clusters,
            instruction_map=self.instruction_map,
            gpu_specs=self.gpu_specs,
            computation_map=self.computation_map,
        )
