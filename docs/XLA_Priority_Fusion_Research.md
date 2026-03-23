# XLA Priority-Based Fusion — Research Summary

> **Sources:**
> - [RFC: XLA:GPU Priority-based fusion pass (Discussion #6407)](https://github.com/openxla/xla/discussions/6407)
> - [Cost Models in XLA GPU — Present and Future (Discussion #10065)](https://github.com/openxla/xla/discussions/10065)
> - XLA source code: `xla/backends/gpu/transforms/priority_fusion.{h,cc}`, `xla/service/gpu/model/gpu_performance_model.{h,cc}`
>
> **Date:** 2026-03-23

---

## 1. Historical Context

The old fusion pipeline had two separate passes:

1. **`GpuInstructionFusion`** — greedy, heuristic-only, reverse post-order, no cost model. Ran **twice** (with/without duplication).
2. **`FusionMerger`** — cost-model-based but limited to `kLoop` fusions, couldn't undo bad decisions from pass 1.

Both files (`instruction_fusion.cc`, `fusion_merger.cc`) have been **fully removed** from the XLA codebase. They've been replaced by a single `PriorityFusion` pass, enabled by default since mid-January 2024.

---

## 2. Current Architecture (PriorityFusion)

### 2.1 Source File Locations

Files have been relocated from their historical paths:

| File | Current Path |
|------|-------------|
| PriorityFusion header | `xla/backends/gpu/transforms/priority_fusion.h` |
| PriorityFusion impl | `xla/backends/gpu/transforms/priority_fusion.cc` |
| Cost model header | `xla/service/gpu/model/gpu_performance_model.h` |
| Cost model impl | `xla/service/gpu/model/gpu_performance_model.cc` |
| Cost model base | `xla/service/gpu/model/gpu_performance_model_base.h` |
| Coalescing analysis | `xla/service/gpu/model/coalescing_analysis.cc` |
| Per-opcode profiles | `xla/service/gpu/model/hlo_op_profiles_data.h` |

### 2.2 Class Hierarchy

```
HloModulePass
  └── PriorityFusion                        (NOT derived from InstructionFusion)
        ├── RunImpl()                        Main loop
        ├── CanFuse(producer, consumer)      Legality checks
        └── Fuse(producer, consumer)         Merge instructions

PriorityFusionQueue                          (anonymous namespace in .cc)
  ├── CalculateProducerPriority(producer)    Cost-model-driven priority
  ├── DequeueNextProducer()                  Pop highest priority
  ├── ComputeAndSetPriorities(instructions)  Batch priority computation
  └── UpdatePriorities()                     Incremental after fusion

GpuPerformanceModelBase
  └── GpuPerformanceModel                   Analytical cost model
        └── EstimateRunTimes(producer, consumers)

GpuPerformanceModelWithIndexingAnalysis      Tile-based cost model (Triton path)
```

---

## 3. The Core Algorithm

```
1. Run GpuHloCostAnalysis on the entire computation

2. For each producer instruction, compute:
       priority = time_unfused - time_fused      (an absl::Duration)

3. Insert all producers with priority >= 0 into:
       std::map<(priority, unique_id), HloInstruction*>
   (ordered map; unique_id breaks ties deterministically)

4. Main loop:
     a. Pop producer with HIGHEST priority (last element in map)
     b. Fuse producer into ALL its consumers
     c. Invalidate caches for affected instructions
     d. Incrementally update priorities of affected operands/consumers
     e. Repeat until queue is empty

5. Post-processing: fuse remaining small constants (1 element) into users
```

### 3.1 Priority Computation

**Special cases (short-circuited):**
- **Fusible bitcasts** (no bit-width change): priority = `+∞` (always fuse first, they're no-ops)
- **Constants**: priority = `-∞` (fused at the very end in a separate pass)
- **Cannot fuse with ALL non-bitcast users**: priority = `-∞` (skip entirely)

**Normal case:**
1. Check `CanFuseWithAllNonBitcastUsers(producer)` — ALL users must be fusible
2. Call `gpu_performance_model_.EstimateRunTimes(producer, &cost_analysis_, fused_consumers)`
3. `priority = time_unfused - time_fused` (positive = beneficial)

**Incremental updates after fusion:**
```
new_priority = old_priority + (new_time_unfused - new_time_fused)
                            - (removed_time_unfused - removed_time_fused)
```
This avoids full recomputation — only the delta from changed consumers is applied.

### 3.2 Critical Design Choice: Producer → ALL Consumers

XLA's priority fusion evaluates whether fusing a **producer into ALL its consumers simultaneously** is beneficial. If it can't fuse with ALL non-bitcast users, the priority is set to `-∞`. This differs fundamentally from pairwise fusion. The rationale:

- If a producer is fused into only *some* consumers, the producer's kernel still needs to run for the remaining consumers → no savings.
- The cost model evaluates the *total* change: eliminating one producer kernel vs. potentially duplicating computation in each consumer.

---

## 4. Cost Model: `GpuPerformanceModel::EstimateRunTimes`

### 4.1 Per-Instruction Runtime

**NOT a simple roofline.** The formula includes 95% compute-memory overlap:

```
exec_time = max(compute_time, memory_time) + (1 - 0.95) × min(compute_time, memory_time)
```

This models 95% overlap between compute and memory pipelines, with 5% serialization.

#### Compute Time

```
compute_time = flop_count × per_op_clock_cycles / (num_threads × clock_rate)
```

- FLOP counts from `GpuHloCostAnalysis` (traverses HLO computations)
- Per-opcode clock cycle costs from `hlo_op_profiles_data.h` (NOT simple FLOP counting)
- Parallelism = total CUDA threads, dependent on emitter (e.g., reduction emitter uses warp shuffles)

#### Memory Time

```
read_time = Σ_operands (operand_size × utilization) / (bandwidth × coalescing_rate × cache_speedup)
write_time = output_size / bandwidth
memory_time = read_time + write_time
```

Key factors:
- **Operand utilization**: can be < 1 (e.g., `slice` reads subset) or > 1 (e.g., `broadcast` re-reads data)
- **Coalescing waste**: fully strided access → only 4B/64B = 1/16 bandwidth for f32. Determined by `CoalescingAnalysis`
- **L1/L2 caching**: small operands get bandwidth multiplier (L1: 8×, L2: 2.5×)
- **Occupancy degradation**: bandwidth reduced if too few threads for good occupancy
- **First access always DRAM**: `n_bytes_net` from DRAM, subsequent accesses may be cached

### 4.2 Unfused vs. Fused Estimation

**`time_unfused`:**
```
time_unfused = 1μs × (num_consumers + 1)        // kernel launch overhead
             + producer_exec_time
             + Σ consumer_exec_time
```

**`time_fused`:**
```
time_fused = 1μs × num_consumers                 // one fewer kernel launch
           + Σ fused_exec_time(producer, consumer_i)
```

**`fused_exec_time(producer, consumer)`:**
```
fused_flops = producer_flops × utilization_by_consumer + consumer_flops
fused_read_time = Σ_operands read_time (shared operands use GetSharedOperandBytesAccessed)
fused_write_time = consumer_write_time  (intermediate write eliminated)
fused_exec_time = max(fused_compute, fused_memory) + 0.05 × min(fused_compute, fused_memory)
```

Key fusion savings:
1. Eliminates 1 kernel launch (1 μs)
2. Eliminates producer's write to HBM + consumer's read of that intermediate
3. Potentially better memory coalescing in fused kernel
4. Shared operands counted once (not double-read)

---

## 5. Fusion Legality (`CanFuse`)

Ordered check sequence in `CanFuse(producer, consumer)`:

1. **Root check**: Cannot fuse if producer is the computation's root
2. **IsFusible**: Both must pass `IsFusible()` (supports elementwise, bitcast, copy, iota, constant, reduce, broadcast, concatenate, dynamic-slice, dynamic-update-slice, gather, pad, reduce-window, reshape, reverse, scatter, slice, transpose, and non-custom fusions)
3. **Triton path** (preferred): `CanFuseTriton()` — checks symbolic tile analysis via `SymbolicTileAnalysis`, parameter budget. If either instruction is already a Triton fusion, Triton is required (no fallback)
4. **Bitcast consumer**: Cannot fuse into a standalone bitcast consumer
5. **Scatter**: `CanEmitInputFusedScatter()` legality
6. **Reduce-into-reduce**: Forbid fusing producer containing significant reduce (reduction_size ≥ 16) into consumer also containing reduce (cost model doesn't handle this well)
7. **Reduction epilog**: Don't convert reduction fusion → loop fusion
8. **Budget**: `FusionFitsInBudget()` — shared memory, parameter count limits
9. **Code size**: `ProducerConsumerMergedTooLarge()` — prevents exponential code growth
10. **In-place ops**: `ShouldFuseInPlaceOp()` — DynamicUpdateSlice legality

---

## 6. Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `kKernelLaunchOverhead` | 1 μs | Per-kernel launch cost |
| `kNcclKernelLaunchOverhead` | 5 μs | NCCL collective kernel launch |
| `kL2CacheSpeedup` | 2.5× | L2 cache bandwidth multiplier |
| `kL1CacheSpeedup` | 8× | L1 cache bandwidth multiplier |
| `kMemoryComputeParallelism` | 0.95 | Compute-memory overlap factor |
| Reduction size threshold | 16 | Min elements for "significant" reduce |

---

## 7. Caching Architecture

The implementation uses extensive caching to avoid redundant computation:

| Cache | Key | Value | Purpose |
|-------|-----|-------|---------|
| `GpuPerformanceModelCache` | instruction / (producer, consumer) | `EstimateRunTimeData` / `Duration` | Runtime estimates |
| `can_fuse_cache_` | (producer, consumer) | `FusionDecision` | Legality results |
| `block_level_parameters_cache_` | (producer, consumer) | `BlockLevelParameters` | Triton tile sizes |
| `tiled_run_time_data_cache_` | `FusionId` | `TiledRunTimeDataOrError` | Triton symbolic tile analysis |
| `FusionDeduplicationCache` | structural hash | — | Avoid re-analyzing identical fusions |
| `HloFusionAnalysisCache` | instruction | `HloFusionAnalysis` | Emitter kind, launch dims |
| `FusionInfoCache` | instruction | shared memory usage etc. | Budget checks |

All caches are invalidated via `InvalidateCaches(instruction)` when an instruction is consumed by fusion.

---

## 8. Triton Fusion Path

The Triton path is the **preferred** fusion strategy when available:

1. Check `IsTritonSupportedInstruction()` for both operands
2. Call `TryFindBestTilingForFusion()` via `SymbolicTileAnalysis` to find optimal tile sizes
3. Store resulting `BlockLevelParameters` (tile sizes, block counts) for backend config
4. Runtime estimated by indexing-based performance model (not the analytical one)
5. Multi-output Triton fusion: gated by flag, limited to exactly one consumer

If either producer or consumer is already a Triton fusion, the code exclusively uses the Triton path (no fallback to normal emitters).

---

## 9. Comparison: Our GO Fusion vs. XLA Reality

| Aspect | GO Fusion (Current) | XLA PriorityFusion |
|--------|--------------------|--------------------|
| **Priority source** | RL-learned integer (0–19) | Analytical: `time_unfused - time_fused` |
| **Fusion granularity** | Pairwise: producer-consumer merge | Producer → ALL consumers simultaneously |
| **Cost model** | Simple roofline: `max(compute, memory) + 5μs` | Sophisticated: coalescing, utilization, L1/L2, 95% overlap |
| **Launch overhead** | 5 μs | 1 μs |
| **Compute time** | FLOP count / peak_FLOPS | Per-opcode clock cycles / achievable parallelism |
| **Memory time** | bytes / bandwidth | bytes × utilization / (bandwidth × coalescing × cache_speedup) |
| **Legality checks** | Category heuristics (element-wise, barrier) | 10-step emitter-aware checks (Triton, scatter, reduce-reduce, budget) |
| **Priority update** | None (one-shot assignment) | Incremental after each fusion |
| **Cycle detection** | BFS on cluster DAG | Structural (guaranteed by producer-consumer relationship in DAG) |

---

## 10. Key Takeaways for Our Project

1. **XLA's analytical priority = our RL objective.** XLA computes `time_unfused - time_fused` analytically. Our RL agent learns to assign priorities that determine fusion order. The RL approach could potentially discover orderings that the analytical model misses.

2. **Our fusion simulator should match XLA's algorithm.** The current pairwise greedy merge doesn't match XLA's "fuse producer into ALL consumers" approach. Aligning this is critical for transferability.

3. **Our cost model needs significant upgrades.** The current simple roofline (max + 5μs) misses: operand utilization, coalescing waste, L1/L2 caching, compute-memory overlap (95%), and per-opcode clock cycle costs.

4. **Fusion legality needs refinement.** XLA has 10 specific checks (Triton, scatter, reduce-into-reduce, budget, code size, in-place ops). Our category-based heuristics are too coarse.

5. **The "fuse with ALL consumers or none" constraint is fundamental.** A producer that can't fuse with all its consumers gets priority `-∞` in XLA. Our simulator should reflect this.
