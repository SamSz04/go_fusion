# Cost Model Alignment Analysis: Ours vs. XLA's `GpuPerformanceModel`

> **Date:** 2026-04-02
> **Scope:** Comparison of `src/env/performance_model.py` against XLA's `GpuPerformanceModel` (source: `xla/service/gpu/model/gpu_performance_model.cc`)

---

## 1. What Matches XLA

| Component | XLA | Ours | Verdict |
|-----------|-----|------|---------|
| Top-level formula | `max(C, M) + 0.05·min(C, M)` | Same (line 481–484) | Matches |
| `kMemoryComputeParallelism` | 0.95 | 0.95 | Matches |
| `kKernelLaunchOverhead` | 1 μs | 1 μs | Matches |
| `kL1CacheSpeedup` / `kL2CacheSpeedup` | 8.0 / 2.5 | 8.0 / 2.5 | Matches |
| Intermediate write elision | Elided when all consumers in-cluster | Same (`_compute_write_time`, line 398–408) | Matches |
| All-or-nothing fusion semantics | Producer → all consumers or skip | Same (`fusion_simulator.py`) | Matches |

---

## 2. What Does NOT Match — Ranked by Impact

### Gap 1: Compute Time (High Impact)

```
XLA:   compute_time = flop_count × per_op_clock_cycles / (num_threads × clock_rate)
Ours:  compute_time = heuristic_flops / spec_sheet_peak_FLOPS       (line 476)
```

XLA uses **empirically profiled per-opcode clock cycles** from `hlo_op_profiles_data.h` — a hardware-measured lookup table, not FLOP heuristics. XLA also computes **achievable parallelism** (`num_threads`) based on the emitter kind (loop emitter, reduction emitter with warp shuffles, input fusion emitter), each of which has different thread launch configurations.

We use handcoded FLOP multipliers (`add`=1, `tanh`=8, `dot`=2MNK in `compute_flops()`, lines 50–158) divided by the datasheet peak TFLOPS. This implicitly assumes all SMs are fully utilized with no occupancy limitations — a fundamentally different calculation.

### Gap 2: Coalescing Analysis (High Impact)

```
XLA:   CoalescingAnalysis with symbolic tile analysis → precise per-operand coalescing ratio
Ours:  _coalescing_factor(): transpose→1/16, gather→0.25, everything else→1.0  (lines 201–232)
```

XLA performs actual memory access pattern analysis using indexing maps to determine what fraction of each cache line is useful. Our 3-value opcode heuristic is a crude approximation. For example, a `transpose` that doesn't touch the innermost dimension has perfect coalescing in XLA's analysis but could be mislabeled by our permutation check; conversely, a complex `gather` might have near-perfect coalescing that we penalize to 0.25.

### Gap 3: Occupancy-Dependent Bandwidth (High Impact)

```
XLA:   Adjusts effective bandwidth based on SM occupancy (register/shmem pressure → fewer warps)
Ours:  Always uses full spec-sheet bandwidth
```

**Completely absent from our model.** If a fused kernel uses too many registers per thread, fewer warps can be resident per SM, reducing the GPU's ability to hide memory latency. XLA models this; we don't. This is directly related to the Register Spilling Cliff problem — our cost model is blind to it.

### Gap 4: Fused Kernel Estimation / Shared Operands (Medium Impact)

```
XLA:   fused_flops = producer_flops × utilization_by_consumer + consumer_flops
       fused_reads use GetSharedOperandBytesAccessed (shared operands counted once)

Ours:  Each instruction's operands counted independently per-cluster member  (lines 447–466)
       No shared-operand deduplication
```

XLA explicitly tracks when a producer and consumer share an operand so the fused kernel reads it once, not twice. We don't — each instruction in `estimate_cluster_runtime` has its operands counted independently via `_compute_read_time`. The `fused_intermediates` set (line 458) only handles intra-cluster intermediates (outputs produced and consumed within the cluster), not shared external operands.

### Gap 5: First-Access-Always-DRAM (`n_bytes_net`) (Medium Impact)

```
XLA:   First access from DRAM (n_bytes_net), subsequent accesses may hit L1/L2
Ours:  L1/L2 speedup applied uniformly to entire operand if it fits in cache  (lines 257–262)
```

XLA distinguishes between the initial DRAM fetch and subsequent cached accesses. We apply the L1/L2 multiplier to the entire operand read, overestimating cache benefit.

### Gap 6: Operand Utilization Coverage (Medium Impact)

```
XLA:   GetOperandUtilization() handles all opcodes, recursively for fusions
Ours:  Only broadcast and slice/dynamic-slice                                (lines 181–194)
```

We miss `reduce` (utilization depends on reduction dimensions), `pad`, `concatenate`, `gather` with specific indexing patterns, and recursive analysis inside fused sub-computations.

### Gap 7: Cache Residency Threshold (Low–Medium Impact)

```
XLA:   Considers per-SM occupancy and actual kernel memory footprint across SMs
Ours:  operand_bytes <= shared_memory_per_sm for L1                          (line 257)
```

Comparing against `shared_memory_per_sm` (a per-SM quantity) is incorrect when the operand is distributed across many SMs. A 100KB operand "fits" in 164KB shared memory per-SM, but in practice each SM only processes a tile of the kernel's work.

### Gap 8: Write Coalescing (Low Impact)

```
XLA:   Write time affected by output coalescing
Ours:  write_time = output_bytes / bandwidth, no coalescing applied          (line 409)
```

### Gap 9: Triton Fusion Path (N/A for current inputs)

```
XLA:   GpuPerformanceModelWithIndexingAnalysis + SymbolicTileAnalysis
Ours:  Not implemented
```

XLA has a separate tile-based cost model for Triton fusions. Not relevant for our current HLO inputs (pre-fusion stage), but a gap nonetheless.

### Gap 10: Priority Semantics (Architectural difference, not a bug)

```
XLA:   priority = time_unfused - time_fused  (analytical Duration, incrementally updated)
Ours:  priority = RL-learned integer ∈ {0..19} (one-shot assignment, no updates)
```

This is by design — the RL agent learns priorities that the cost model evaluates. Our cost model is used as a **reward signal**, not as a **priority computation**, so errors affect the quality of the reward rather than the fusion order directly.

---

## 3. Impact Summary

| Gap | Component | Impact | Effort to Fix |
|-----|-----------|--------|---------------|
| #1 | Compute time (profiled cycles vs. FLOP heuristic) | **High** | High — requires porting `hlo_op_profiles_data.h` and emitter-aware thread count |
| #2 | Coalescing (symbolic tile vs. 3-value heuristic) | **High** | Very High — requires indexing analysis infrastructure |
| #3 | Occupancy-dependent bandwidth | **High** | Medium — requires register/shmem usage estimation per kernel |
| #4 | Shared operand deduplication | **Medium** | Low — track shared operands across cluster members |
| #5 | First-access-always-DRAM | **Medium** | Low — split read into DRAM portion + cached portion |
| #6 | Operand utilization coverage | **Medium** | Medium — add cases for reduce, pad, concatenate, etc. |
| #7 | Cache residency threshold | **Low–Medium** | Low — use total cache / num_active_SMs instead of per-SM |
| #8 | Write coalescing | **Low** | Low — apply `_coalescing_factor` to writes |
| #9 | Triton path | **N/A** | Very High |
| #10 | Priority semantics | **By design** | N/A |

---

## 4. Conclusion

Our cost model is best described as an **XLA-inspired analytical approximation**: it shares the top-level formula structure and key constants but uses substantially simplified sub-components. The three most impactful gaps — compute time calculation (heuristic FLOPs vs. profiled clock cycles), coalescing analysis (3-value lookup vs. symbolic tile analysis), and absent occupancy modeling — mean that the absolute runtime values our model produces are unlikely to correlate tightly with XLA's own estimates for the same fusion configurations.

However, for the purpose of **relative comparison** (ranking fusion strategies against each other), the model may still provide a useful signal — the question is whether the ranking is preserved despite the absolute error. Validating this requires comparing our model's rankings against actual XLA kernel measurements, which is identified in the project's future work plan.
