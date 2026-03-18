"""
Fusion legality rules for the GO fusion system.

``FusionRules`` encodes the constraints that decide whether two HLO nodes
(or two clusters of nodes) may be legally merged into a single fused kernel.
The rules mirror the key restrictions from XLA's fusion passes:

1. Both sides must be fusable (not bare parameters / constants).
2. A dataflow edge must connect the producer to the consumer.
3. The merge must not introduce a cycle in the cluster-level DAG.
4. The merged cluster must not exceed a size cap (default 64 instructions).
5. ``custom-call`` nodes (e.g. cuBLAS GEMMs) act as *barriers*: they cannot
   be absorbed into another cluster, but their **outputs** may be consumed
   by a downstream fusion.
6. Reductions may fuse with their element-wise inputs.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

from src.hlo_parser.hlo_ir import HloInstruction


class FusionRules:
    """Static rule-set governing fusion legality.

    All public methods are either class methods or static methods so that the
    class can be used without instantiation.  The opcode category sets are
    class-level constants.
    """

    # ------------------------------------------------------------------
    # Opcode categories
    # ------------------------------------------------------------------

    ELEMENT_WISE_OPS: Set[str] = {
        "add", "subtract", "multiply", "divide",
        "maximum", "minimum",
        "tanh", "rsqrt", "sqrt", "exp", "log", "power",
        "compare", "select", "convert", "clamp",
        "negate", "abs", "sign",
        "xor", "or", "and", "not",
        "shift-left", "shift-right-logical", "shift-right-arithmetic",
        "floor", "ceil", "round-nearest-afz",
    }

    SHAPE_OPS: Set[str] = {
        "broadcast", "reshape", "bitcast", "slice",
        "concatenate", "transpose", "gather", "iota",
        "dynamic-slice", "dynamic-update-slice", "pad", "reverse",
    }

    REDUCTION_OPS: Set[str] = {
        "reduce",
    }

    BARRIER_OPS: Set[str] = {
        "custom-call",   # cuBLAS GEMM or other external calls
        "parameter",
        "constant",
        "get-tuple-element",
        "tuple",
    }

    # Maximum number of instructions in a single fused cluster.
    MAX_CLUSTER_SIZE: int = 64

    # ------------------------------------------------------------------
    # Fusability predicates
    # ------------------------------------------------------------------

    @classmethod
    def is_fusable(cls, instruction: HloInstruction) -> bool:
        """Return True if *instruction* can participate in fusion at all.

        Parameters and constants are never fused (they are implicitly
        available).  ``custom-call`` nodes are barriers and cannot be
        *absorbed* into a fusion, but their outputs can feed one.
        """
        return instruction.opcode not in cls.BARRIER_OPS

    @classmethod
    def is_barrier(cls, instruction: HloInstruction) -> bool:
        """Return True if *instruction* is a fusion barrier."""
        return instruction.opcode in cls.BARRIER_OPS

    # ------------------------------------------------------------------
    # Edge-level legality
    # ------------------------------------------------------------------

    @classmethod
    def can_fuse(
        cls,
        producer: HloInstruction,
        consumer: HloInstruction,
        cluster_of: Dict[str, int],
        cluster_members: Dict[int, Set[str]],
        instruction_map: Dict[str, HloInstruction],
        adjacency: Dict[str, List[str]],
        max_cluster_size: Optional[int] = None,
    ) -> bool:
        """Decide whether the clusters of *producer* and *consumer* may merge.

        Args:
            producer: The upstream instruction.
            consumer: The downstream instruction.
            cluster_of: Mapping ``instruction_name -> cluster_id``.
            cluster_members: Mapping ``cluster_id -> set of member names``.
            instruction_map: All instructions keyed by name.
            adjacency: Forward adjacency list (producer -> [consumers]).
            max_cluster_size: Override for ``MAX_CLUSTER_SIZE``.

        Returns:
            ``True`` if the merge is legal.
        """
        if max_cluster_size is None:
            max_cluster_size = cls.MAX_CLUSTER_SIZE

        # 1. Both must be individually fusable.
        if not cls.is_fusable(producer) or not cls.is_fusable(consumer):
            return False

        # 2. They must already be in *different* clusters.
        pid = cluster_of.get(producer.name)
        cid = cluster_of.get(consumer.name)
        if pid is None or cid is None:
            return False
        if pid == cid:
            return False  # already fused

        # 3. There must be a dataflow edge from producer to consumer.
        if producer.name not in consumer.operand_names:
            return False

        # 4. Merged cluster must not exceed the size limit.
        merged_size = len(cluster_members[pid]) + len(cluster_members[cid])
        if merged_size > max_cluster_size:
            return False

        # 5. Merging must not create a cycle in the cluster-level DAG.
        if cls.would_create_cycle(
            pid, cid, cluster_of, cluster_members, adjacency,
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    @classmethod
    def would_create_cycle(
        cls,
        cluster_a: int,
        cluster_b: int,
        cluster_of: Dict[str, int],
        cluster_members: Dict[int, Set[str]],
        adjacency: Dict[str, List[str]],
    ) -> bool:
        """Return True if merging *cluster_a* into *cluster_b* would create a
        cycle in the cluster-level DAG.

        The idea: if there exists *another* path from cluster_b to cluster_a
        that does not go through the direct edge we are trying to fuse, then
        merging would create a cycle.  We check this with a BFS on the
        **cluster-level** graph, starting from cluster_b and seeing if we can
        reach cluster_a without using the direct edge.

        More precisely, we build the cluster-level forward adjacency on the
        fly and do a BFS from cluster_b.  If cluster_a is reachable (ignoring
        the direct b->a edges that we intend to merge), the merge is illegal.
        """
        # Build cluster-level forward adjacency (excluding a->b direct).
        # We only care about reachability from b to a via other clusters.
        cluster_adj: Dict[int, Set[int]] = {}
        for src_name, dst_names in adjacency.items():
            src_cid = cluster_of.get(src_name)
            if src_cid is None:
                continue
            for dst_name in dst_names:
                dst_cid = cluster_of.get(dst_name)
                if dst_cid is None or dst_cid == src_cid:
                    continue
                # Skip the direct edge from a -> b (the one we want to merge).
                if src_cid == cluster_a and dst_cid == cluster_b:
                    continue
                cluster_adj.setdefault(src_cid, set()).add(dst_cid)

        # BFS from cluster_b; if we can reach cluster_a, merging is cyclic.
        visited: Set[int] = set()
        queue: deque[int] = deque()
        queue.append(cluster_b)
        visited.add(cluster_b)

        while queue:
            cur = queue.popleft()
            for neighbor in cluster_adj.get(cur, set()):
                if neighbor == cluster_a:
                    return True  # cycle detected
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False
