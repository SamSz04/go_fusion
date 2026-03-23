"""
Priority-based fusion simulation with Union-Find cluster management.

Given a set of HLO instructions, dataflow edges, and per-node priority
scores, this module simulates XLA-style operator fusion by greedily
merging each **producer into ALL its consumers** simultaneously (all-or-
nothing), in descending priority order, subject to the legality
constraints in ``src.env.fusion_rules.FusionRules``.

Key XLA-aligned behaviors:

* **All-or-nothing**: a producer is fused into ALL its non-barrier
  consumers or into none.  If any consumer fails the legality check,
  the entire merge is skipped.
* **Special-case priorities**: fusible bitcasts are fused first
  (priority = +inf, they're no-ops); constants are fused last in a
  separate pass.
* **Union-Find** (disjoint-set) for near-constant-time merges and
  lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.hlo_parser.hlo_ir import HloInstruction
from src.env.fusion_rules import FusionRules


# ======================================================================
# Union-Find (Disjoint Set Union)
# ======================================================================

class UnionFind:
    """Weighted quick-union with path compression.

    Each element is identified by an arbitrary hashable key (here, an
    instruction name string).
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def make_set(self, x: str) -> None:
        """Create a singleton set for *x* (no-op if it already exists)."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        """Return the canonical representative for the set containing *x*.

        Uses path compression for amortised near-O(1) performance.
        """
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        """Merge the sets containing *a* and *b*.

        Returns the representative of the merged set.  Uses union-by-rank
        to keep the tree shallow.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # Attach smaller tree under larger tree.
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return ra

    def connected(self, a: str, b: str) -> bool:
        """Return True if *a* and *b* are in the same set."""
        return self.find(a) == self.find(b)


# ======================================================================
# Fusion cluster
# ======================================================================

@dataclass
class FusionCluster:
    """A cluster of fused instructions that will execute as one kernel.

    Attributes:
        id: Unique cluster identifier.
        members: Set of instruction names belonging to this cluster.
    """
    id: int
    members: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)


# ======================================================================
# Simulation entry point
# ======================================================================

def simulate_fusion(
    instructions: List[HloInstruction],
    edges: List[Tuple[str, str]],
    priorities: Dict[str, float],
    fusion_rules: Optional[FusionRules] = None,
    max_cluster_size: int = FusionRules.MAX_CLUSTER_SIZE,
) -> List[FusionCluster]:
    """Simulate XLA-style priority-based operator fusion.

    Algorithm (aligned with XLA ``PriorityFusion``):

    1. Initialise every node in its own singleton cluster (Union-Find).
    2. **Pre-pass**: fuse all fusible bitcasts first (priority = +inf).
    3. Sort fusable producers by their priority score in descending order.
    4. For each producer in that order (**all-or-nothing**):
       a. Find ALL non-barrier consumer clusters.
       b. Check if the producer can fuse with ALL of them.
       c. If any check fails, skip this producer entirely.
       d. If all pass, merge the producer into every consumer cluster.
    5. **Post-pass**: fuse remaining small constants (1 element) into users.
    6. Collect and return the final set of clusters.

    Args:
        instructions: All HLO instructions in the graph.
        edges: Directed dataflow edges as ``(producer_name, consumer_name)``
            pairs.
        priorities: Mapping ``instruction_name -> priority_score``.  Higher
            priority means the node is considered for fusion earlier.
        fusion_rules: ``FusionRules`` instance (uses a default if ``None``).
        max_cluster_size: Maximum allowed cluster size (passed through to
            the legality checker).

    Returns:
        A list of ``FusionCluster`` objects representing the final fusion
        partitioning.
    """
    if fusion_rules is None:
        fusion_rules = FusionRules()

    # -- Build lookup structures --
    instruction_map: Dict[str, HloInstruction] = {
        inst.name: inst for inst in instructions
    }

    # Forward adjacency: producer -> list of consumer names.
    adjacency: Dict[str, List[str]] = {inst.name: [] for inst in instructions}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)

    # -- Initialise Union-Find and cluster bookkeeping --
    uf = UnionFind()
    cluster_of: Dict[str, int] = {}
    cluster_members: Dict[int, Set[str]] = {}

    next_cluster_id = 0
    for inst in instructions:
        uf.make_set(inst.name)
        cid = next_cluster_id
        next_cluster_id += 1
        cluster_of[inst.name] = cid
        cluster_members[cid] = {inst.name}

    # Helper: merge producer into a consumer cluster.
    def _merge(producer_name: str, consumer_name: str) -> None:
        nonlocal next_cluster_id

        pid = cluster_of[uf.find(producer_name)]
        cid = cluster_of[uf.find(consumer_name)]
        if pid == cid:
            return

        new_root = uf.union(producer_name, consumer_name)
        merged_members = cluster_members[pid] | cluster_members[cid]
        new_cid = cluster_of[new_root]

        for member in merged_members:
            cluster_of[uf.find(member)] = new_cid
            cluster_of[member] = new_cid

        cluster_members[new_cid] = merged_members

        old_cid = pid if new_cid == cid else cid
        if old_cid != new_cid and old_cid in cluster_members:
            del cluster_members[old_cid]

    # Helper: get current cluster_of snapshot for legality checks.
    def _current_cluster_of() -> Dict[str, int]:
        return {name: cluster_of[uf.find(name)] for name in instruction_map}

    # ==========================================================
    # Pre-pass: fuse fusible bitcasts first (priority = +inf)
    # ==========================================================
    for inst in instructions:
        if inst.opcode != "bitcast":
            continue
        if not FusionRules.is_fusable(inst):
            continue
        producer_name = inst.name
        consumers = adjacency.get(producer_name, [])
        for consumer_name in consumers:
            consumer = instruction_map.get(consumer_name)
            if consumer is None or not FusionRules.is_fusable(consumer):
                continue
            if uf.connected(producer_name, consumer_name):
                continue
            _merge(producer_name, consumer_name)

    # ==========================================================
    # Main pass: all-or-nothing producer → all consumers
    # ==========================================================
    # Collect fusable producers (exclude bitcasts already handled and constants).
    fusable_producers = [
        inst.name for inst in instructions
        if FusionRules.is_fusable(inst)
        and inst.opcode != "bitcast"
        and inst.opcode != "constant"
    ]
    fusable_producers.sort(key=lambda n: priorities.get(n, 0.0), reverse=True)

    for producer_name in fusable_producers:
        producer = instruction_map[producer_name]

        # Skip if producer is root.
        if getattr(producer, "is_root", False):
            continue

        # Find all non-barrier consumers.
        consumer_names = []
        for cn in adjacency.get(producer_name, []):
            c = instruction_map.get(cn)
            if c is None:
                continue
            if FusionRules.is_barrier(c):
                continue
            # Skip if already in the same cluster.
            if uf.connected(producer_name, cn):
                continue
            consumer_names.append(cn)

        if not consumer_names:
            continue

        # All-or-nothing: check ALL consumers.
        current_co = _current_cluster_of()
        all_legal = True
        for cn in consumer_names:
            consumer = instruction_map[cn]
            can_merge = FusionRules.can_fuse(
                producer=producer,
                consumer=consumer,
                cluster_of=current_co,
                cluster_members=cluster_members,
                instruction_map=instruction_map,
                adjacency=adjacency,
                max_cluster_size=max_cluster_size,
            )
            if not can_merge:
                all_legal = False
                break

        if not all_legal:
            continue

        # Merge producer into ALL consumer clusters.
        for cn in consumer_names:
            if not uf.connected(producer_name, cn):
                _merge(producer_name, cn)

    # ==========================================================
    # Post-pass: fuse small constants into their users
    # ==========================================================
    for inst in instructions:
        if inst.opcode != "constant":
            continue
        if inst.shape.num_elements > 1:
            continue  # only small (scalar) constants
        producer_name = inst.name
        consumers = adjacency.get(producer_name, [])
        for cn in consumers:
            consumer = instruction_map.get(cn)
            if consumer is None or not FusionRules.is_fusable(consumer):
                continue
            if uf.connected(producer_name, cn):
                continue
            _merge(producer_name, cn)

    # -- Collect final clusters --
    root_to_members: Dict[str, Set[str]] = {}
    for inst in instructions:
        root = uf.find(inst.name)
        root_to_members.setdefault(root, set()).add(inst.name)

    clusters: List[FusionCluster] = []
    for idx, (root, members) in enumerate(sorted(root_to_members.items())):
        clusters.append(FusionCluster(id=idx, members=members))

    return clusters
