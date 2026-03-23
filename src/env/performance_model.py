"""
XLA-aligned analytical cost model for HLO instruction clusters.

Models kernel runtime using XLA's ``GpuPerformanceModel`` formulation::

    exec_time = max(compute_time, memory_time)
              + (1 - parallelism) * min(compute_time, memory_time)
              + kernel_launch_overhead

where ``parallelism = 0.95`` (XLA's ``kMemoryComputeParallelism``).

Key improvements over a simple roofline:

* **Compute-memory overlap**: 95% overlap, 5% serialization.
* **Operand utilization**: broadcast > 1 (re-reads), slice < 1 (subset).
* **Coalescing approximation**: strided access degrades effective bandwidth.
* **L1/L2 cache modeling**: small operands get bandwidth multipliers
  (L1: 8x, L2: 2.5x from ``kL1CacheSpeedup``/``kL2CacheSpeedup``).
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
# Operand utilization (XLA: GpuPerformanceModelBase)
# ======================================================================

def _operand_utilization(
    consumer: HloInstruction,
    operand: HloInstruction,
) -> float:
    """Estimate how much of the operand tensor is actually accessed.

    XLA's ``GpuPerformanceModelBase::GetOperandUtilization`` computes the
    ratio of bytes the consumer actually touches to the operand's total
    bytes.  A ``broadcast`` re-reads each element multiple times
    (utilization > 1), while a ``slice`` reads only a subset (utilization < 1).

    Returns:
        Utilization factor (1.0 = reads entire operand exactly once).
    """
    opcode = consumer.opcode

    if opcode == "broadcast":
        # Consumer reads every input element output_elements / input_elements times.
        if operand.shape.num_elements > 0:
            return consumer.shape.num_elements / operand.shape.num_elements
        return 1.0

    if opcode in ("slice", "dynamic-slice"):
        # Consumer reads a subset of the operand.
        if operand.shape.num_elements > 0:
            return consumer.shape.num_elements / operand.shape.num_elements
        return 1.0

    # Default: reads entire operand once.
    return 1.0


# ======================================================================
# Coalescing approximation
# ======================================================================

def _coalescing_factor(consumer: HloInstruction) -> float:
    """Approximate memory coalescing efficiency for the consumer instruction.

    XLA uses ``CoalescingAnalysis`` with symbolic tile analysis for accurate
    results.  We approximate with opcode-based heuristics:

    - ``transpose`` touching the innermost dimension: ~1/16 efficiency
      (each warp accesses 32 × 4B = 128B but across a 64B cache line
      only 4B/64B = 1/16 is useful for f32).
    - ``gather``: ~0.25 efficiency (scattered access pattern).
    - Default: 1.0 (fully coalesced).

    Returns:
        Coalescing factor in (0, 1].  Lower = worse coalescing = slower reads.
    """
    opcode = consumer.opcode

    if opcode == "transpose":
        # Check if the transpose involves the innermost dimension.
        perm = consumer.attributes.get("dimensions", [])
        if perm:
            ndim = len(perm)
            # If the last dimension in the permutation is not ndim-1,
            # the innermost dim changed → poor coalescing.
            if perm[-1] != ndim - 1:
                return 0.0625  # 1/16
        return 1.0

    if opcode == "gather":
        return 0.25

    return 1.0


# ======================================================================
# Cache bandwidth modeling (XLA: L1/L2 speedup)
# ======================================================================

def _cache_bandwidth_multiplier(
    operand_bytes: int,
    gpu_specs: GPUSpecs,
) -> float:
    """Determine the bandwidth multiplier based on operand cache residency.

    XLA models that small operands benefit from L1 or L2 cache bandwidth:
    - Fits in shared memory (L1): ``kL1CacheSpeedup = 8.0``
    - Fits in L2 cache: ``kL2CacheSpeedup = 2.5``
    - Otherwise: HBM bandwidth (1.0)

    Args:
        operand_bytes: Total bytes of the operand tensor.
        gpu_specs: GPU hardware specs with cache sizes and speedups.

    Returns:
        Bandwidth multiplier (>= 1.0).
    """
    if operand_bytes <= gpu_specs.shared_memory_per_sm:
        return gpu_specs.l1_cache_speedup  # 8.0
    elif operand_bytes <= gpu_specs.l2_cache_size:
        return gpu_specs.l2_cache_speedup  # 2.5
    else:
        return 1.0


# ======================================================================
# Byte-traffic estimation (with utilization, coalescing, caching)
# ======================================================================

def _tensor_bytes(shape: HloShape) -> int:
    """Total bytes for a tensor described by *shape*."""
    return shape.total_bytes


def compute_bytes_accessed(
    instruction: HloInstruction,
    instruction_map: Dict[str, HloInstruction],
    fused_intermediates: Optional[Set[str]] = None,
    fused_cluster_members: Optional[Set[str]] = None,
) -> int:
    """Estimate the bytes read + written by *instruction* (simple version).

    This is the legacy interface that returns raw byte counts without
    utilization / coalescing / caching adjustments.  Used for memory
    reduction metrics.  The full cost model uses ``_compute_read_time``
    and ``_compute_write_time`` directly.
    """
    if fused_intermediates is None:
        fused_intermediates = set()
    if fused_cluster_members is None:
        fused_cluster_members = set()

    opcode = instruction.opcode

    if opcode in ("parameter", "constant"):
        return 0

    total_bytes = 0

    for op_name in instruction.operand_names:
        if op_name in fused_intermediates:
            continue
        op_inst = instruction_map.get(op_name)
        if op_inst is not None:
            total_bytes += _tensor_bytes(op_inst.shape)

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


# Keep the old name for backward compatibility.
compute_bytes = compute_bytes_accessed


# ======================================================================
# Per-operand read time (with utilization, coalescing, cache)
# ======================================================================

def _compute_read_time(
    instruction: HloInstruction,
    instruction_map: Dict[str, HloInstruction],
    gpu_specs: GPUSpecs,
    fused_intermediates: Optional[Set[str]] = None,
) -> float:
    """Compute read time for all operands of *instruction* in seconds.

    Accounts for operand utilization, coalescing, and L1/L2 cache.
    Mirrors XLA's per-operand read time computation in
    ``GpuPerformanceModel::EstimateRunTimes``.
    """
    if fused_intermediates is None:
        fused_intermediates = set()

    if instruction.opcode in ("parameter", "constant"):
        return 0.0

    bandwidth = gpu_specs.memory_bandwidth
    if bandwidth <= 0:
        return 0.0

    coalescing = _coalescing_factor(instruction)
    total_read_time = 0.0

    for op_name in instruction.operand_names:
        if op_name in fused_intermediates:
            continue  # produced in-register within the cluster

        op_inst = instruction_map.get(op_name)
        if op_inst is None:
            continue

        raw_bytes = _tensor_bytes(op_inst.shape)
        utilization = _operand_utilization(instruction, op_inst)
        cache_mult = _cache_bandwidth_multiplier(raw_bytes, gpu_specs)

        effective_bandwidth = bandwidth * coalescing * cache_mult
        if effective_bandwidth <= 0:
            effective_bandwidth = bandwidth  # fallback

        total_read_time += (raw_bytes * utilization) / effective_bandwidth

    return total_read_time


def _compute_write_time(
    instruction: HloInstruction,
    instruction_map: Dict[str, HloInstruction],
    gpu_specs: GPUSpecs,
    fused_cluster_members: Optional[Set[str]] = None,
) -> float:
    """Compute write time for the output of *instruction* in seconds.

    If all consumers are inside the same fused cluster, the write is
    elided (the output stays in registers).
    """
    if instruction.opcode in ("parameter", "constant"):
        return 0.0

    bandwidth = gpu_specs.memory_bandwidth
    if bandwidth <= 0:
        return 0.0

    if fused_cluster_members is None:
        fused_cluster_members = set()

    # Check whether *every* consumer is inside the same cluster.
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
        return _tensor_bytes(instruction.shape) / bandwidth

    return 0.0


# ======================================================================
# Cluster-level runtime estimation (XLA-aligned)
# ======================================================================

def estimate_cluster_runtime(
    cluster_member_names: Set[str],
    instruction_map: Dict[str, HloInstruction],
    gpu_specs: GPUSpecs,
    computation_map: Optional[Dict[str, HloComputation]] = None,
) -> float:
    """Estimate the runtime (in seconds) of a fused cluster.

    Uses XLA's ``GpuPerformanceModel`` formula::

        exec_time = max(compute_time, memory_time)
                  + (1 - parallelism) * min(compute_time, memory_time)
                  + kernel_launch_overhead

    where ``parallelism = 0.95`` models 95% compute-memory overlap.

    Args:
        cluster_member_names: Set of instruction names in this cluster.
        instruction_map: Global instruction-name -> instruction mapping.
        gpu_specs: Target GPU hardware parameters.
        computation_map: Optional computation map for recursive FLOP counting.

    Returns:
        Estimated runtime in seconds.
    """
    total_flops = 0
    total_read_time = 0.0
    total_write_time = 0.0

    for name in cluster_member_names:
        inst = instruction_map.get(name)
        if inst is None:
            continue

        total_flops += compute_flops(inst, computation_map)

        total_read_time += _compute_read_time(
            inst,
            instruction_map,
            gpu_specs,
            fused_intermediates=cluster_member_names - {name},
        )

        total_write_time += _compute_write_time(
            inst,
            instruction_map,
            gpu_specs,
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
    memory_time = total_read_time + total_write_time

    # XLA overlap formula: 95% parallel, 5% serialized.
    p = gpu_specs.compute_memory_parallelism
    runtime = (
        max(compute_time, memory_time)
        + (1.0 - p) * min(compute_time, memory_time)
        + gpu_specs.kernel_launch_overhead
    )
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
