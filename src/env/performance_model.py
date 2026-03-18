"""
Roofline analytical cost model for HLO instruction clusters.

The roofline model estimates kernel runtime as::

    runtime = max(compute_time, memory_time) + kernel_launch_overhead

where:

* ``compute_time = total_FLOPs / peak_FLOPS``
* ``memory_time  = bytes_accessed / memory_bandwidth``

This module exposes helpers to compute FLOP counts and byte traffic for
individual instructions and whole fusion clusters, then combines them
via the roofline formula using GPU hardware specs from
``src.utils.gpu_specs.GPUSpecs``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from src.hlo_parser.hlo_ir import HloInstruction, HloShape, HloComputation, HloModule
from src.utils.gpu_specs import GPUSpecs


# ======================================================================
# FLOP estimation
# ======================================================================

# Opcodes that are free (pure data movement / metadata).
_ZERO_FLOP_OPS = frozenset({
    "broadcast", "reshape", "bitcast", "slice", "parameter", "constant",
    "tuple", "get-tuple-element", "copy",
})

# 1 FLOP per element.
_SIMPLE_OPS = frozenset({
    "add", "multiply", "subtract", "compare", "select",
    "xor", "or", "and", "not",
    "shift-left", "shift-right-logical", "shift-right-arithmetic",
    "negate", "abs", "sign", "clamp", "convert", "maximum", "minimum",
    "floor", "ceil", "round-nearest-afz",
})


def compute_flops(
    instruction: HloInstruction,
    computation_map: Optional[Dict[str, HloComputation]] = None,
    _visited: Optional[Set[str]] = None,
) -> int:
    """Estimate the number of floating-point operations for *instruction*.

    Args:
        instruction: The HLO instruction to analyze.
        computation_map: Mapping of computation names to ``HloComputation``
            objects.  Required when *instruction* is a ``fusion`` or
            ``custom-call`` that references a sub-computation.
        _visited: Internal set tracking visited computation names to prevent
            infinite recursion.

    Returns:
        Estimated FLOPs (integer).
    """
    opcode = instruction.opcode
    n_elements = instruction.shape.num_elements

    # ---- Free ops (zero FLOPs) ----
    if opcode in _ZERO_FLOP_OPS:
        return 0

    # ---- Simple element-wise (1 FLOP / element) ----
    if opcode in _SIMPLE_OPS:
        return n_elements

    # ---- More expensive element-wise ----
    if opcode in ("divide", "rsqrt", "sqrt", "exp", "log"):
        return 4 * n_elements

    if opcode == "tanh":
        return 8 * n_elements

    if opcode == "power":
        return 8 * n_elements  # conservative: exp(y * log(x))

    # ---- Reduction ----
    if opcode == "reduce":
        # FLOPs ~ number of input elements (one reduction op per element).
        # The input shape is typically the first operand; we approximate via
        # attributes or fall back to output * reduction factor.
        reduce_dims = instruction.attributes.get("dimensions", [])
        input_dims = list(instruction.shape.dimensions)
        # Reconstruct input size: output dims + reduced dims.
        # If we have the reduce_dims sizes in attributes, use them.
        reduce_sizes = instruction.attributes.get("reduce_dim_sizes", [])
        if reduce_sizes:
            factor = 1
            for s in reduce_sizes:
                factor *= s
            return n_elements * factor
        # Fallback: assume reduction is over the last dimension and guess
        # a moderate factor.  This is intentionally conservative.
        return n_elements

    # ---- Dot / matmul ----
    if opcode == "dot":
        # 2 * M * N * K for standard GEMM.
        dot_dims = instruction.attributes.get("dot_dimension_numbers", {})
        contracting = dot_dims.get("lhs_contracting_dimensions", [])
        # K is the contracting dimension size.
        k_size = 1
        lhs_dims = instruction.attributes.get("lhs_shape_dims")
        if lhs_dims and contracting:
            for d in contracting:
                if d < len(lhs_dims):
                    k_size *= lhs_dims[d]
        else:
            # Heuristic: assume last dim of output shape is N, and K == N.
            k_size = instruction.shape.dimensions[-1] if instruction.shape.dimensions else 1
        return 2 * n_elements * k_size

    # ---- Convolution ----
    if opcode == "convolution":
        # Approximate: 2 * output_elements * kernel_volume * input_features.
        window = instruction.attributes.get("window", {})
        kernel_size = 1
        for dim_info in window.get("dimensions", []):
            kernel_size *= dim_info.get("size", 1)
        input_features = instruction.attributes.get("feature_group_count", 1)
        return 2 * n_elements * kernel_size * input_features

    # ---- Fusion / custom-call: recurse into sub-computation ----
    if opcode in ("fusion", "custom-call"):
        if computation_map:
            # Use called_computation (from "calls"/"to_apply" attributes),
            # NOT computation_name (which is the *parent* computation).
            comp_name = instruction.called_computation
            if comp_name:
                # Guard against infinite recursion via visited set
                if _visited is None:
                    _visited = set()
                if comp_name in _visited:
                    return 0
                _visited.add(comp_name)
                comp = computation_map.get(comp_name)
                if comp is not None:
                    total = 0
                    for sub_inst in comp.instructions:
                        total += compute_flops(sub_inst, computation_map, _visited)
                    return total
        # custom-call without a known computation: treat as opaque barrier.
        return 0

    # ---- Fallback: treat as element-wise 1 FLOP/element ----
    return n_elements


# ======================================================================
# Byte-traffic estimation
# ======================================================================

def _tensor_bytes(shape: HloShape) -> int:
    """Total bytes for a tensor described by *shape*."""
    return shape.total_bytes


def compute_bytes(
    instruction: HloInstruction,
    instruction_map: Dict[str, HloInstruction],
    fused_intermediates: Optional[Set[str]] = None,
    fused_cluster_members: Optional[Set[str]] = None,
) -> int:
    """Estimate the bytes read + written by *instruction*.

    Operands whose names appear in *fused_intermediates* are assumed to reside
    in registers / shared memory and therefore do **not** incur an HBM read.

    If all consumers of this instruction are inside *fused_cluster_members*,
    the write is also elided (the output stays in registers).

    Args:
        instruction: The instruction to analyze.
        instruction_map: All instructions keyed by name (for consumer lookup).
        fused_intermediates: Names of operands produced within the same
            fusion cluster (reads skipped).
        fused_cluster_members: Names of all instructions in the same fusion
            cluster (writes skipped if all consumers are members).

    Returns:
        Estimated bytes accessed from/to global memory.
    """
    if fused_intermediates is None:
        fused_intermediates = set()
    if fused_cluster_members is None:
        fused_cluster_members = set()

    opcode = instruction.opcode

    # Parameters and constants are read-only; their cost is accounted for
    # by their *consumers*.
    if opcode in ("parameter", "constant"):
        return 0

    total_bytes = 0

    # ---- Reads: operand tensors not already in registers ----
    for op_name in instruction.operand_names:
        if op_name in fused_intermediates:
            continue  # produced in-register within the cluster
        op_inst = instruction_map.get(op_name)
        if op_inst is not None:
            total_bytes += _tensor_bytes(op_inst.shape)

    # ---- Write: output tensor ----
    # Check whether *every* consumer is inside the same cluster.  If so, the
    # output is consumed in-register and never written to HBM.
    all_consumers_fused = True
    if fused_cluster_members:
        for inst in instruction_map.values():
            if instruction.name in inst.operand_names:
                if inst.name not in fused_cluster_members:
                    all_consumers_fused = False
                    break
    else:
        all_consumers_fused = False

    if not all_consumers_fused or instruction.is_root:
        total_bytes += _tensor_bytes(instruction.shape)

    return total_bytes


# ======================================================================
# Cluster-level runtime estimation
# ======================================================================

def estimate_cluster_runtime(
    cluster_member_names: Set[str],
    instruction_map: Dict[str, HloInstruction],
    gpu_specs: GPUSpecs,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Estimate the runtime (in seconds) of a fused cluster via roofline.

    Args:
        cluster_member_names: Set of instruction names in this cluster.
        instruction_map: Global instruction-name -> instruction mapping.
        gpu_specs: Target GPU hardware parameters.
        computation_map: Optional computation map for recursive FLOP counting.

    Returns:
        Estimated runtime in seconds.
    """
    total_flops = 0
    total_bytes = 0

    for name in cluster_member_names:
        inst = instruction_map.get(name)
        if inst is None:
            continue

        total_flops += compute_flops(inst, computation_map)
        total_bytes += compute_bytes(
            inst,
            instruction_map,
            fused_intermediates=cluster_member_names - {name},
            fused_cluster_members=cluster_member_names,
        )

    # Choose peak FLOPS based on predominant dtype in the cluster.
    uses_fp16 = any(
        instruction_map[n].shape.element_type in ("f16", "bf16")
        for n in cluster_member_names
        if n in instruction_map
    )
    peak_flops = gpu_specs.peak_flops_fp16 if uses_fp16 else gpu_specs.peak_flops_fp32

    compute_time = total_flops / peak_flops if peak_flops > 0 else 0.0
    memory_time = total_bytes / gpu_specs.memory_bandwidth if gpu_specs.memory_bandwidth > 0 else 0.0

    runtime = max(compute_time, memory_time) + gpu_specs.kernel_launch_overhead
    return runtime


def estimate_total_runtime(
    clusters: List[Set[str]],
    instruction_map: Dict[str, HloInstruction],
    gpu_specs: GPUSpecs,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Estimate total runtime as the sum of per-cluster runtimes.

    This is a simplification that assumes sequential (non-overlapping) kernel
    execution, which is the common case for a single-stream GPU program.

    Args:
        clusters: List of clusters (each a set of instruction names).
        instruction_map: Global instruction-name -> instruction mapping.
        gpu_specs: Target GPU hardware parameters.
        computation_map: Optional computation map for recursive FLOP counting.

    Returns:
        Total estimated runtime in seconds.
    """
    total = 0.0
    for cluster in clusters:
        total += estimate_cluster_runtime(
            cluster, instruction_map, gpu_specs, computation_map
        )
    return total
