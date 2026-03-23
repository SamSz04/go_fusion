"""
GPU hardware specifications for roofline performance modeling.

Provides a ``GPUSpecs`` dataclass and factory presets for common NVIDIA GPUs
(V100, A100, H100).  The specs are consumed by the roofline cost model in
``src.env.performance_model`` to estimate kernel runtimes.
"""

from dataclasses import dataclass


@dataclass
class GPUSpecs:
    """Hardware parameters for a single GPU.

    All throughput numbers are *peak sustained* values.

    Attributes:
        name: Human-readable GPU name (e.g. "NVIDIA A100").
        peak_flops_fp32: Peak FP32 FLOPS (without tensor cores).
        peak_flops_fp16: Peak FP16 / BF16 FLOPS (tensor core path).
        memory_bandwidth: HBM bandwidth in bytes per second.
        l2_cache_size: L2 cache capacity in bytes.
        shared_memory_per_sm: Shared memory per SM in bytes.
        num_sms: Number of streaming multiprocessors.
        kernel_launch_overhead: Fixed cost in seconds per kernel launch
            (XLA uses 1 μs = ``kKernelLaunchOverhead``).
        compute_memory_parallelism: Fraction of compute-memory overlap
            (XLA's ``kMemoryComputeParallelism`` = 0.95).
        l1_cache_speedup: Bandwidth multiplier for L1-resident operands
            (XLA's ``kL1CacheSpeedup`` = 8.0).
        l2_cache_speedup: Bandwidth multiplier for L2-resident operands
            (XLA's ``kL2CacheSpeedup`` = 2.5).
    """
    name: str
    peak_flops_fp32: float       # FLOPS
    peak_flops_fp16: float       # FLOPS (tensor core)
    memory_bandwidth: float      # bytes/sec
    l2_cache_size: int           # bytes
    shared_memory_per_sm: int    # bytes
    num_sms: int
    kernel_launch_overhead: float  # seconds
    compute_memory_parallelism: float = 0.95  # XLA kMemoryComputeParallelism
    l1_cache_speedup: float = 8.0             # XLA kL1CacheSpeedup
    l2_cache_speedup: float = 2.5             # XLA kL2CacheSpeedup

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def arithmetic_intensity_fp32(self) -> float:
        """Roofline ridge point for FP32 (FLOP/byte)."""
        return self.peak_flops_fp32 / self.memory_bandwidth

    @property
    def arithmetic_intensity_fp16(self) -> float:
        """Roofline ridge point for FP16 / tensor-core (FLOP/byte)."""
        return self.peak_flops_fp16 / self.memory_bandwidth

    @property
    def total_shared_memory(self) -> int:
        """Aggregate shared memory across all SMs (bytes)."""
        return self.shared_memory_per_sm * self.num_sms


# ======================================================================
# GPU presets
# ======================================================================

def v100_specs() -> GPUSpecs:
    """NVIDIA Tesla V100 (SXM2, 16 GB HBM2).

    Reference: https://images.nvidia.com/content/volta-architecture/pdf/
    volta-architecture-whitepaper.pdf
    """
    return GPUSpecs(
        name="NVIDIA V100",
        peak_flops_fp32=15.7e12,       # 15.7 TFLOPS
        peak_flops_fp16=125.0e12,      # 125 TFLOPS (tensor core)
        memory_bandwidth=900e9,        # 900 GB/s
        l2_cache_size=6 * 1024 * 1024, # 6 MB
        shared_memory_per_sm=96 * 1024,  # 96 KB (configurable up to 96 KB)
        num_sms=80,
        kernel_launch_overhead=1e-6,   # 1 μs (XLA kKernelLaunchOverhead)
    )


def a100_specs() -> GPUSpecs:
    """NVIDIA A100 (SXM4, 80 GB HBM2e).

    Reference: https://www.nvidia.com/content/dam/en-zz/Solutions/
    Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
    """
    return GPUSpecs(
        name="NVIDIA A100",
        peak_flops_fp32=19.5e12,        # 19.5 TFLOPS
        peak_flops_fp16=312.0e12,       # 312 TFLOPS (tensor core, with sparsity: 624)
        memory_bandwidth=2.0e12,        # 2.0 TB/s
        l2_cache_size=40 * 1024 * 1024, # 40 MB
        shared_memory_per_sm=164 * 1024,  # 164 KB (configurable)
        num_sms=108,
        kernel_launch_overhead=1e-6,    # 1 μs (XLA kKernelLaunchOverhead)
    )


def h100_specs() -> GPUSpecs:
    """NVIDIA H100 (SXM5, 80 GB HBM3).

    Reference: https://resources.nvidia.com/en-us-tensor-core/
    nvidia-tensor-core-gpu-datasheet
    """
    return GPUSpecs(
        name="NVIDIA H100",
        peak_flops_fp32=67.0e12,        # 67 TFLOPS
        peak_flops_fp16=990.0e12,       # ~990 TFLOPS (tensor core, with sparsity: 1979)
        memory_bandwidth=3.35e12,       # 3.35 TB/s
        l2_cache_size=50 * 1024 * 1024, # 50 MB
        shared_memory_per_sm=228 * 1024,  # 228 KB
        num_sms=132,
        kernel_launch_overhead=1e-6,    # 1 μs (XLA kKernelLaunchOverhead)
    )


# Convenience mapping from short names to factory functions.
GPU_PRESETS = {
    "v100": v100_specs,
    "a100": a100_specs,
    "h100": h100_specs,
}


def get_gpu_specs(name: str) -> GPUSpecs:
    """Look up a GPU preset by short name (case-insensitive).

    Args:
        name: One of "v100", "a100", "h100".

    Returns:
        A freshly constructed ``GPUSpecs`` instance.

    Raises:
        ValueError: If the name is not in the preset registry.
    """
    key = name.lower()
    if key not in GPU_PRESETS:
        raise ValueError(
            f"Unknown GPU preset '{name}'. "
            f"Available: {sorted(GPU_PRESETS.keys())}"
        )
    return GPU_PRESETS[key]()
