"""
Baseline fusion strategies for comparison against the GO learned policy.

Each baseline takes parsed HLO instructions, name-based edges, and GPU specs,
then returns an estimated total runtime. Baselines implement different
priority-assignment strategies that feed into the same fusion simulator
used by the RL environment.

Baselines from the paper:
- No fusion: each op is its own kernel (normalization reference)
- XLA default: greedy fusion in reverse post-order
- Random priority: random assignment, averaged over seeds
- Greedy (memory): priority proportional to memory savings
- Simulated annealing: search with random priority swaps
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from src.hlo_parser.hlo_ir import HloInstruction, HloComputation
from src.env.fusion_simulator import simulate_fusion, FusionCluster
from src.env.fusion_rules import FusionRules
from src.env.performance_model import estimate_total_runtime
from src.utils.gpu_specs import GPUSpecs


def _compute_runtime(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    priorities: Dict[str, float],
    max_cluster_size: int = 64,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Shared helper: given priorities, simulate fusion and compute runtime.

    Args:
        instructions: List of HloInstruction objects from the parsed graph.
        edges: List of (producer_name, consumer_name) string tuples.
        gpu_specs: GPU hardware parameters.
        priorities: Dict mapping instruction name to priority score.
        max_cluster_size: Maximum number of ops in a fusion cluster.
        computation_map: Optional computation map for FLOP estimation.

    Returns:
        Estimated total runtime in seconds.
    """
    clusters = simulate_fusion(
        instructions=instructions,
        edges=edges,
        priorities=priorities,
        max_cluster_size=max_cluster_size,
    )

    instruction_map = {inst.name: inst for inst in instructions}
    cluster_sets: List[Set[str]] = [c.members for c in clusters]

    total_runtime = estimate_total_runtime(
        clusters=cluster_sets,
        instruction_map=instruction_map,
        gpu_specs=gpu_specs,
        computation_map=computation_map,
    )

    return total_runtime


def no_fusion_baseline(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """No-fusion baseline: each fusable op is its own kernel.

    This is the normalization reference used in the reward function.
    Every instruction gets a unique priority with minimal score,
    and since no merging occurs, each fusable op stays as a singleton cluster.

    Returns:
        Total runtime with no fusion applied.
    """
    instruction_map = {inst.name: inst for inst in instructions}
    singleton_clusters: List[Set[str]] = [
        {inst.name} for inst in instructions
        if FusionRules.is_fusable(inst)
    ]

    if not singleton_clusters:
        return 0.0

    return estimate_total_runtime(
        clusters=singleton_clusters,
        instruction_map=instruction_map,
        gpu_specs=gpu_specs,
        computation_map=computation_map,
    )


def random_priority_baseline(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    num_priorities: int = 20,
    num_seeds: int = 10,
    max_cluster_size: int = 64,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Random priority assignment, averaged over multiple seeds.

    Each fusable node gets a uniformly random priority from [0, num_priorities).
    Results are averaged over num_seeds trials to reduce variance.

    Args:
        num_priorities: Number of distinct priority levels.
        num_seeds: Number of random trials to average over.
        max_cluster_size: Maximum cluster size for fusion legality.

    Returns:
        Average runtime across all random seeds.
    """
    fusable_names = [inst.name for inst in instructions if FusionRules.is_fusable(inst)]
    runtimes = []

    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)
        priorities: Dict[str, float] = {}
        for name in fusable_names:
            priorities[name] = float(rng.randint(0, num_priorities))
        # Non-fusable get -inf
        for inst in instructions:
            if inst.name not in priorities:
                priorities[inst.name] = float("-inf")

        runtime = _compute_runtime(
            instructions, edges, gpu_specs, priorities,
            max_cluster_size, computation_map,
        )
        runtimes.append(runtime)

    return float(np.mean(runtimes))


def greedy_memory_baseline(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    num_priorities: int = 20,
    max_cluster_size: int = 64,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Priority proportional to memory savings from fusing with consumers.

    For each fusable instruction, compute the total bytes of output tensors
    that could be saved by fusing with its consumers (those outputs would stay
    in registers/shared memory instead of going through HBM). Higher savings
    get higher priority (processed first in the fusion algorithm).

    Returns:
        Estimated runtime using memory-greedy priorities.
    """
    # Build consumer count per instruction
    consumer_count: Dict[str, int] = {inst.name: 0 for inst in instructions}
    for src, dst in edges:
        if src in consumer_count:
            consumer_count[src] += 1

    # Compute memory savings score per fusable instruction
    fusable_instructions = [inst for inst in instructions if FusionRules.is_fusable(inst)]
    savings: Dict[str, float] = {}
    for inst in fusable_instructions:
        output_bytes = inst.shape.total_bytes
        num_consumers = consumer_count.get(inst.name, 0)
        # Savings = output_bytes * (1 write + num_consumer reads avoided)
        savings[inst.name] = output_bytes * (1 + num_consumers)

    # Quantize savings into priority bins
    if not savings:
        return no_fusion_baseline(instructions, edges, gpu_specs, computation_map)

    values = list(savings.values())
    min_val, max_val = min(values), max(values)

    priorities: Dict[str, float] = {}
    for name, score in savings.items():
        if max_val > min_val:
            normalized = (score - min_val) / (max_val - min_val)
        else:
            normalized = 0.0
        priorities[name] = normalized * (num_priorities - 1)

    # Non-fusable get -inf
    for inst in instructions:
        if inst.name not in priorities:
            priorities[inst.name] = float("-inf")

    return _compute_runtime(
        instructions, edges, gpu_specs, priorities,
        max_cluster_size, computation_map,
    )


def xla_default_fusion_baseline(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    num_priorities: int = 20,
    max_cluster_size: int = 64,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Simulate XLA's default greedy fusion behavior.

    XLA processes instructions in reverse post-order (topological order from
    outputs to inputs) and greedily fuses each producer with its consumer.
    We approximate this by assigning priorities in reverse topological order:
    nodes closer to outputs get higher priority (processed first).

    Returns:
        Estimated runtime simulating XLA default fusion.
    """
    # Build adjacency for topological sort (using names)
    name_to_inst = {inst.name: inst for inst in instructions}
    in_degree: Dict[str, int] = {inst.name: 0 for inst in instructions}
    adj: Dict[str, List[str]] = {inst.name: [] for inst in instructions}

    for src, dst in edges:
        if src in adj and dst in in_degree:
            adj[src].append(dst)
            in_degree[dst] += 1

    # Kahn's algorithm for topological sort
    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    topo_order: List[str] = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes were visited (unlikely in valid HLO), add remaining
    visited = set(topo_order)
    for inst in instructions:
        if inst.name not in visited:
            topo_order.append(inst.name)

    num_ops = len(topo_order)

    # Reverse topological order: nodes closer to outputs get higher priority
    priorities: Dict[str, float] = {}
    for rank, name in enumerate(reversed(topo_order)):
        if name in name_to_inst and FusionRules.is_fusable(name_to_inst[name]):
            bin_idx = int(rank / max(num_ops, 1) * (num_priorities - 1))
            priorities[name] = float(min(bin_idx, num_priorities - 1))

    # Non-fusable get -inf
    for inst in instructions:
        if inst.name not in priorities:
            priorities[inst.name] = float("-inf")

    return _compute_runtime(
        instructions, edges, gpu_specs, priorities,
        max_cluster_size, computation_map,
    )


def simulated_annealing_baseline(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    num_priorities: int = 20,
    num_iterations: int = 10000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.9995,
    max_cluster_size: int = 64,
    seed: int = 42,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Simulated annealing with random priority swaps.

    Starts from a random priority assignment and iteratively proposes
    single-node priority changes. Accepts improvements always; accepts
    degradations with probability exp(-delta/temperature). Uses geometric
    cooling schedule.

    Returns:
        Best runtime found during the annealing process.
    """
    rng = np.random.RandomState(seed)
    fusable_names = [inst.name for inst in instructions if FusionRules.is_fusable(inst)]
    num_fusable = len(fusable_names)

    if num_fusable == 0:
        return no_fusion_baseline(instructions, edges, gpu_specs, computation_map)

    # Initialize with random priorities
    current_prio_values = rng.randint(0, num_priorities, size=num_fusable)
    current_priorities: Dict[str, float] = {
        name: float(val) for name, val in zip(fusable_names, current_prio_values)
    }
    for inst in instructions:
        if inst.name not in current_priorities:
            current_priorities[inst.name] = float("-inf")

    current_runtime = _compute_runtime(
        instructions, edges, gpu_specs, current_priorities,
        max_cluster_size, computation_map,
    )

    best_runtime = current_runtime
    best_priorities = dict(current_priorities)

    temperature = initial_temp

    for iteration in range(num_iterations):
        # Propose a neighbor: change one random fusable node's priority
        node_idx = rng.randint(0, num_fusable)
        node_name = fusable_names[node_idx]
        old_val = current_priorities[node_name]
        new_val = float(rng.randint(0, num_priorities))

        new_priorities = dict(current_priorities)
        new_priorities[node_name] = new_val

        new_runtime = _compute_runtime(
            instructions, edges, gpu_specs, new_priorities,
            max_cluster_size, computation_map,
        )

        # Acceptance criterion
        delta = new_runtime - current_runtime
        if delta < 0:
            accept = True
        else:
            if temperature > 1e-10 and best_runtime > 0:
                accept = rng.random() < math.exp(-delta / (temperature * best_runtime))
            else:
                accept = False

        if accept:
            current_priorities = new_priorities
            current_runtime = new_runtime

        if current_runtime < best_runtime:
            best_runtime = current_runtime
            best_priorities = dict(current_priorities)

        # Cool down
        temperature *= cooling_rate

    return best_runtime


def run_all_baselines(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    gpu_specs: GPUSpecs,
    num_priorities: int = 20,
    max_cluster_size: int = 64,
    sa_iterations: int = 10000,
    computation_map: Optional[Dict[str, HloComputation]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run all baseline strategies and return their runtimes.

    Args:
        instructions: Parsed HLO instructions.
        edges: Edge list as (producer_name, consumer_name) string tuples.
        gpu_specs: GPU hardware parameters (GPUSpecs dataclass).
        num_priorities: Number of priority levels.
        max_cluster_size: Maximum fusion cluster size.
        sa_iterations: Number of simulated annealing iterations.
        computation_map: Optional computation map for FLOP estimation.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping baseline name to estimated runtime.
    """
    results = {}

    if verbose:
        print("Running baselines...")

    if verbose:
        print("  [1/5] No fusion...")
    results["no_fusion"] = no_fusion_baseline(
        instructions, edges, gpu_specs, computation_map
    )

    if verbose:
        print("  [2/5] Random priority...")
    results["random"] = random_priority_baseline(
        instructions, edges, gpu_specs, num_priorities,
        max_cluster_size=max_cluster_size, computation_map=computation_map,
    )

    if verbose:
        print("  [3/5] Greedy memory...")
    results["greedy_memory"] = greedy_memory_baseline(
        instructions, edges, gpu_specs, num_priorities,
        max_cluster_size=max_cluster_size, computation_map=computation_map,
    )

    if verbose:
        print("  [4/5] XLA default...")
    results["xla_default"] = xla_default_fusion_baseline(
        instructions, edges, gpu_specs, num_priorities,
        max_cluster_size=max_cluster_size, computation_map=computation_map,
    )

    if verbose:
        print("  [5/5] Simulated annealing...")
    results["simulated_annealing"] = simulated_annealing_baseline(
        instructions, edges, gpu_specs, num_priorities,
        num_iterations=sa_iterations, max_cluster_size=max_cluster_size,
        computation_map=computation_map,
    )

    if verbose:
        print("  Done.")

    return results
