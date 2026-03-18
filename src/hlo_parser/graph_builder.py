"""
Graph builder: convert an HloModule into a PyTorch Geometric Data object.

Builds a directed dataflow graph from the ENTRY computation where:
* Each instruction is a node.
* Edges go from producer (operand) to consumer (the instruction that uses it).
* Reverse edges are also stored for bidirectional GNN message passing.
* Non-fusable nodes (parameter, constant, get-tuple-element, tuple) are flagged.
* Nodes are topologically sorted.

The resulting ``torch_geometric.data.Data`` object stores:
    - ``x``:               placeholder node features (num_nodes, 1), to be
                           replaced by ``FeatureEncoder``.
    - ``edge_index``:      (2, num_edges) forward edges (producer -> consumer).
    - ``reverse_edge_index``: (2, num_edges) reverse edges.
    - ``node_names``:      list of instruction names (len = num_nodes).
    - ``node_opcodes``:    list of opcode strings.
    - ``is_fusable``:      boolean tensor (num_nodes,).
    - ``topo_order``:      LongTensor with the topological index of each node.
    - ``num_nodes``:       int.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from .hlo_ir import HloComputation, HloInstruction, HloModule


def build_graph(
    module: HloModule,
    *,
    computation: Optional[HloComputation] = None,
) -> Data:
    """Build a PyG ``Data`` graph from an HloModule's ENTRY computation.

    Args:
        module: Parsed HLO module.
        computation: If provided, build the graph from this computation
            instead of the ENTRY computation.  Defaults to
            ``module.entry_computation``.

    Returns:
        A ``torch_geometric.data.Data`` object ready for feature encoding.

    Raises:
        ValueError: If no ENTRY computation is found.
    """
    comp = computation or module.entry_computation
    if comp is None:
        raise ValueError(
            "No ENTRY computation found in the module.  "
            "Either pass an explicit computation or parse an HLO file "
            "that contains an ENTRY block."
        )

    instructions = comp.instructions
    if not instructions:
        raise ValueError(f"Computation '{comp.name}' has no instructions.")

    # --- Build name -> index mapping ---
    name_to_idx: Dict[str, int] = {
        inst.name: idx for idx, inst in enumerate(instructions)
    }

    # --- Build forward edges (producer -> consumer) ---
    src_nodes: List[int] = []
    dst_nodes: List[int] = []

    for inst in instructions:
        consumer_idx = name_to_idx[inst.name]
        for operand_name in inst.operand_names:
            producer_idx = name_to_idx.get(operand_name)
            if producer_idx is not None:
                src_nodes.append(producer_idx)
                dst_nodes.append(consumer_idx)

    # --- Topological sort ---
    topo_order = _topological_sort(instructions, name_to_idx)

    # --- Build tensors ---
    num_nodes = len(instructions)

    if src_nodes:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        reverse_edge_index = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        reverse_edge_index = torch.zeros((2, 0), dtype=torch.long)

    is_fusable = torch.tensor(
        [inst.is_fusable for inst in instructions], dtype=torch.bool
    )

    topo_tensor = torch.tensor(topo_order, dtype=torch.long)

    # Placeholder features (to be replaced by FeatureEncoder)
    x = torch.zeros((num_nodes, 1), dtype=torch.float32)

    node_names = [inst.name for inst in instructions]
    node_opcodes = [inst.opcode for inst in instructions]

    data = Data(
        x=x,
        edge_index=edge_index,
        reverse_edge_index=reverse_edge_index,
        is_fusable=is_fusable,
        topo_order=topo_tensor,
        num_nodes=num_nodes,
    )
    # Attach metadata as Python attributes (not tensors).
    data.node_names = node_names
    data.node_opcodes = node_opcodes
    # Store instructions for feature encoding downstream
    data.instructions = instructions
    # Store the module for accessing sub-computations if needed
    data.hlo_module = module

    return data


def _topological_sort(
    instructions: List[HloInstruction],
    name_to_idx: Dict[str, int],
) -> List[int]:
    """Kahn's algorithm for topological sort.

    Returns a list of length ``len(instructions)`` where ``result[i]``
    is the topological position of instruction ``i``.
    """
    n = len(instructions)

    # Build adjacency and in-degree from operand relationships
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    in_degree = [0] * n

    for inst in instructions:
        consumer = name_to_idx[inst.name]
        for operand_name in inst.operand_names:
            producer = name_to_idx.get(operand_name)
            if producer is not None:
                children[producer].append(consumer)
                in_degree[consumer] += 1

    # BFS
    queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_position = [0] * n
    order = 0
    while queue:
        node = queue.popleft()
        topo_position[node] = order
        order += 1
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # If there's a cycle (shouldn't happen in HLO), remaining nodes get
    # order values at the end.
    if order < n:
        for i in range(n):
            if topo_position[i] == 0 and in_degree[i] != 0:
                topo_position[i] = order
                order += 1

    return topo_position
