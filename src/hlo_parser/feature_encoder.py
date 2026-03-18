"""
Feature encoder: convert HLO instructions into numeric feature tensors
for GNN input.

Produces two tensors per graph:

* ``data.x``:          continuous features of shape ``(N, 19)``
* ``data.opcode_ids``: integer opcode indices of shape ``(N,)``

The opcode index is consumed by a learnable ``nn.Embedding(132, embed_dim)``
inside the policy network, following the TpuGraphs approach.  This replaces
the previous sparse 30-dim one-hot encoding with a dense, learnable
representation over all 132 XLA HLO opcodes.

=========================  ====  ==========================================
Feature                    Dims  Encoding
=========================  ====  ==========================================
*Opcode (separate)*         —    Integer index in [0, 131] → nn.Embedding
Element type                 6   One-hot (f32, s32, u32, u64, pred, bf16)
Num dimensions               1   Scalar (rank)
Dimension sizes              4   log2 of each dim, zero-padded to max_rank=4
Total elements               1   log2(product of dims)
Total bytes                  1   log2(elements * dtype_bytes)
Num inputs (fan-in)          1   Number of operand names
Num users (fan-out)          1   Number of consumers
Shape-compatible inputs      1   1.0 if all operands share the same shape
Is fusion node               1   1.0 if opcode == 'fusion'
Is custom-call (GEMM)        1   1.0 if custom-call with __cublas$gemm
Is fusable                   1   1.0 if the node is fusable
=========================  ====  ==========================================
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch_geometric.data import Data

from .hlo_ir import HloInstruction


# ---------------------------------------------------------------------------
# Opcode vocabulary — all 132 XLA HLO opcodes from hlo_opcode.h
# (https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_opcode.h)
# ---------------------------------------------------------------------------

OPCODE_VOCAB: List[str] = [
    "abs",
    "acos",
    "acosh",
    "add",
    "add-dependency",
    "after-all",
    "all-gather",
    "all-gather-done",
    "all-gather-start",
    "all-reduce",
    "all-reduce-done",
    "all-reduce-start",
    "all-to-all",
    "and",
    "asin",
    "asinh",
    "async-done",
    "async-start",
    "async-update",
    "atan2",
    "atanh",
    "batch-norm-grad",
    "batch-norm-inference",
    "batch-norm-training",
    "bitcast",
    "bitcast-convert",
    "broadcast",
    "call",
    "cbrt",
    "ceil",
    "cholesky",
    "clamp",
    "count-leading-zeros",
    "collective-broadcast",
    "collective-permute",
    "collective-permute-done",
    "collective-permute-start",
    "compare",
    "complex",
    "concatenate",
    "conditional",
    "constant",
    "convert",
    "convolution",
    "copy",
    "copy-done",
    "copy-start",
    "cosine",
    "cosh",
    "custom-call",
    "divide",
    "domain",
    "dot",
    "dynamic-reshape",
    "dynamic-slice",
    "dynamic-update-slice",
    "erf",
    "exponential",
    "exponential-minus-one",
    "fft",
    "floor",
    "fusion",
    "gather",
    "get-dimension-size",
    "get-tuple-element",
    "imag",
    "infeed",
    "iota",
    "is-finite",
    "log",
    "log-plus-one",
    "logistic",
    "map",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "not",
    "opt-barrier",
    "or",
    "outfeed",
    "pad",
    "parameter",
    "partition-id",
    "popcnt",
    "power",
    "ragged-all-to-all",
    "ragged-dot",
    "real",
    "recv",
    "recv-done",
    "reduce",
    "reduce-precision",
    "reduce-scatter",
    "reduce-window",
    "remainder",
    "replica-id",
    "reshape",
    "reverse",
    "rng",
    "rng-bit-generator",
    "rng-get-and-update-state",
    "round-nearest-afz",
    "round-nearest-even",
    "rsqrt",
    "scaled-dot",
    "scan",
    "scatter",
    "select",
    "select-and-scatter",
    "send",
    "send-done",
    "set-dimension-size",
    "shift-left",
    "shift-right-arithmetic",
    "shift-right-logical",
    "sign",
    "sine",
    "sinh",
    "slice",
    "sort",
    "sqrt",
    "stochastic-convert",
    "subtract",
    "tan",
    "tanh",
    "topk",
    "transpose",
    "triangular-solve",
    "tuple",
    "while",
    "xor",
]

OPCODE_TO_IDX: Dict[str, int] = {op: i for i, op in enumerate(OPCODE_VOCAB)}
NUM_OPCODES: int = len(OPCODE_VOCAB)  # 132

# ---------------------------------------------------------------------------
# Element-type vocabulary (6 types)
# ---------------------------------------------------------------------------

ETYPE_VOCAB: List[str] = ["f32", "s32", "u32", "u64", "pred", "bf16"]
ETYPE_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ETYPE_VOCAB)}
NUM_ETYPES: int = len(ETYPE_VOCAB)  # 6

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RANK: int = 4
# Continuous feature dim: etype(6) + rank(1) + dims(4) + elems(1) + bytes(1)
#   + fan_in(1) + fan_out(1) + shape_compat(1) + is_fusion(1) + is_gemm(1) + is_fusable(1) = 19
FEATURE_DIM: int = NUM_ETYPES + 1 + MAX_RANK + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # = 19


def _safe_log2(x: float) -> float:
    """Return log2(x) clamped to 0 for non-positive values."""
    if x <= 0:
        return 0.0
    return math.log2(x)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def encode_features(data: Data) -> Data:
    """Encode node features for all instructions stored in ``data``.

    Expects ``data.instructions`` to be a list of ``HloInstruction``
    objects (attached by ``build_graph``).

    Produces:
    * ``data.x``:          continuous features ``(num_nodes, 19)``
    * ``data.opcode_ids``: integer opcode indices ``(num_nodes,)``

    Fan-out (num_users) is computed from the edge structure in
    ``data.edge_index``.

    Args:
        data: A ``torch_geometric.data.Data`` object produced by
            ``build_graph``.

    Returns:
        The same ``Data`` object with ``data.x`` and ``data.opcode_ids``
        set.
    """
    instructions: List[HloInstruction] = data.instructions
    num_nodes = len(instructions)

    # Pre-compute fan-out from edge_index
    fan_out = _compute_fan_out(data.edge_index, num_nodes)

    features = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.float32)
    opcode_ids = torch.zeros(num_nodes, dtype=torch.long)

    for i, inst in enumerate(instructions):
        features[i] = _encode_single(inst, fan_out[i])
        opcode_ids[i] = OPCODE_TO_IDX.get(inst.opcode, 0)

    data.x = features
    data.opcode_ids = opcode_ids
    return data


def encode_instruction(
    inst: HloInstruction, fan_out: int = 0
) -> torch.Tensor:
    """Encode a single instruction into a continuous feature vector (19,)."""
    return _encode_single(inst, fan_out)


def get_opcode_index(opcode: str) -> int:
    """Return the integer index for an opcode string."""
    return OPCODE_TO_IDX.get(opcode, 0)


# ---------------------------------------------------------------------------
# Internal encoding
# ---------------------------------------------------------------------------

def _encode_single(inst: HloInstruction, fan_out: int) -> torch.Tensor:
    """Build the 19-dim continuous feature vector for one instruction.

    Opcode is handled separately via opcode_ids (integer index for
    nn.Embedding), so it is NOT included in this vector.
    """
    vec = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    offset = 0

    # 1. Element type one-hot (6 dims)
    etype = inst.shape.element_type
    if inst.shape.is_tuple and inst.shape.tuple_shapes:
        etype = inst.shape.tuple_shapes[0].element_type
    eidx = ETYPE_TO_IDX.get(etype, -1)
    if eidx >= 0:
        vec[offset + eidx] = 1.0
    offset += NUM_ETYPES  # 6

    # 2. Num dimensions / rank (1 dim)
    dims = inst.shape.dimensions
    if inst.shape.is_tuple and inst.shape.tuple_shapes:
        dims = inst.shape.tuple_shapes[0].dimensions
    vec[offset] = float(len(dims))
    offset += 1

    # 3. Dimension sizes — log2, padded to MAX_RANK (4 dims)
    for j in range(MAX_RANK):
        if j < len(dims):
            vec[offset + j] = _safe_log2(float(dims[j]))
    offset += MAX_RANK  # 4

    # 4. Total elements — log2 (1 dim)
    num_elements = inst.shape.num_elements
    vec[offset] = _safe_log2(float(num_elements))
    offset += 1

    # 5. Total bytes — log2 (1 dim)
    total_bytes = inst.shape.total_bytes
    vec[offset] = _safe_log2(float(total_bytes))
    offset += 1

    # 6. Num inputs / fan-in (1 dim)
    vec[offset] = float(len(inst.operand_names))
    offset += 1

    # 7. Num users / fan-out (1 dim)
    vec[offset] = float(fan_out)
    offset += 1

    # 8. Shape-compatible inputs (1 dim)
    vec[offset] = float(_all_operands_shape_compatible(inst))
    offset += 1

    # 9. Is fusion node (1 dim)
    vec[offset] = 1.0 if inst.is_fusion else 0.0
    offset += 1

    # 10. Is custom-call GEMM (1 dim)
    vec[offset] = 1.0 if inst.is_gemm else 0.0
    offset += 1

    # 11. Is fusable (1 dim)
    vec[offset] = 1.0 if inst.is_fusable else 0.0
    offset += 1

    return vec


def _all_operands_shape_compatible(inst: HloInstruction) -> bool:
    """Check if all operands would have the same shape dimensions as
    this instruction.

    We cannot directly look up operand shapes here (we only have names),
    so we use a heuristic: if the instruction has exactly 0 or 1 operand,
    it is trivially compatible.  For element-wise ops with 2+ operands
    where the opcode is known to be element-wise (add, multiply, ...),
    assume compatible because XLA broadcasting guarantees it at this
    stage of the pipeline.
    """
    if len(inst.operand_names) <= 1:
        return True
    elementwise = {
        "add", "subtract", "multiply", "divide", "maximum", "minimum",
        "select", "compare", "and", "or", "xor",
    }
    return inst.opcode in elementwise


def _compute_fan_out(
    edge_index: torch.Tensor, num_nodes: int
) -> List[int]:
    """Count the number of consumers (fan-out) for each node from
    the forward edge_index tensor (shape [2, E])."""
    fan_out = [0] * num_nodes
    if edge_index.numel() == 0:
        return fan_out
    src = edge_index[0]
    for s in src.tolist():
        fan_out[s] += 1
    return fan_out
