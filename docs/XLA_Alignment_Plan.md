# Plan: Align Fusion Pipeline with XLA's Priority-Based Fusion

## Context

Our current GO Fusion implementation uses a simplified roofline cost model, pairwise greedy fusion, and coarse legality checks. After studying XLA's actual `PriorityFusion` source code (see `docs/XLA_Priority_Fusion_Research.md`), we identified critical gaps that reduce the fidelity of our fusion simulator. Aligning with XLA's real algorithm is essential so the RL agent learns priorities that transfer to XLA's actual behavior.

**Key gaps identified:**
- Kernel launch overhead: we use 5μs, XLA uses 1μs
- Cost model: we use `max(compute, memory) + overhead`, XLA uses `max(compute, memory) + 0.05 × min(compute, memory)` with operand utilization, coalescing, and L1/L2 cache modeling
- Fusion simulator: we merge pairwise (producer into one consumer at a time), XLA fuses producer into ALL consumers simultaneously (all-or-nothing)
- Fusion rules: we lack reduce-into-reduce prevention, root instruction exclusion, bitcast consumer exclusion, and budget/code-size checks

## Phase 1: Update GPU Specs & Constants

**File: `src/utils/gpu_specs.py`**

Add three new fields to `GPUSpecs`:
```python
compute_memory_parallelism: float = 0.95  # 95% overlap (XLA's kMemoryComputeParallelism)
l1_cache_speedup: float = 8.0             # XLA's kL1CacheSpeedup
l2_cache_speedup: float = 2.5             # XLA's kL2CacheSpeedup
```

Change `kernel_launch_overhead` from `5e-6` to `1e-6` in all three GPU presets (v100, a100, h100) to match XLA's `kKernelLaunchOverhead = 1μs`.

## Phase 2: Upgrade Performance Model

**File: `src/env/performance_model.py`**

### 2.1 Replace roofline formula with XLA's overlap model

Current:
```python
runtime = max(compute_time, memory_time) + gpu_specs.kernel_launch_overhead
```

New:
```python
p = gpu_specs.compute_memory_parallelism  # 0.95
runtime = (max(compute_time, memory_time)
           + (1 - p) * min(compute_time, memory_time)
           + gpu_specs.kernel_launch_overhead)
```

### 2.2 Add operand utilization

For each operand read, scale bytes by a utilization factor:
- `broadcast`: utilization = output_elements / input_elements (re-reads data)
- `slice`, `dynamic-slice`: utilization = output_elements / input_elements (reads subset, < 1)
- Default: utilization = 1.0

New function: `_operand_utilization(instruction: HloInstruction) -> float`

### 2.3 Add L1/L2 cache bandwidth modeling

When computing `memory_time` for a read, check operand size against cache capacities:
```python
def _cache_bandwidth_multiplier(operand_bytes, gpu_specs):
    if operand_bytes <= gpu_specs.shared_memory_per_sm:  # fits in L1/shared
        return gpu_specs.l1_cache_speedup   # 8.0
    elif operand_bytes <= gpu_specs.l2_cache_size:       # fits in L2
        return gpu_specs.l2_cache_speedup   # 2.5
    else:
        return 1.0  # HBM
```

Memory time becomes: `operand_bytes × utilization / (bandwidth × cache_multiplier)`

### 2.4 Add coalescing approximation

Simple heuristic since we don't have full address analysis:
- `transpose` with innermost dim change: coalescing_factor = 0.0625 (1/16, fully strided)
- `gather` with non-contiguous indices: coalescing_factor = 0.25
- Default: coalescing_factor = 1.0

Applied as: `read_time = bytes × utilization / (bandwidth × coalescing × cache_multiplier)`

### 2.5 Update `estimate_cluster_runtime`

Refactor to use the new formula. Add `utilization` and `cache_multiplier` to per-operand byte computation. No signature change needed — the new fields come from `gpu_specs`.

## Phase 3: Upgrade Fusion Simulator

**File: `src/env/fusion_simulator.py`**

### 3.1 Change from pairwise to producer-into-all-consumers

Current algorithm: iterate producers by priority, for each producer try merging with each consumer one at a time.

New algorithm (matching XLA):
```
1. Sort producers by priority (descending)
2. For each producer p:
   a. Find ALL consumers of p: consumers = [c for c in adjacency[p]]
   b. Check if p can fuse with ALL non-barrier consumers
   c. If any consumer fails the legality check → skip p entirely (priority = -∞ semantics)
   d. If all pass → merge p into all consumer clusters simultaneously
3. Post-processing: fuse remaining small constants (1-element) into their users
```

### 3.2 Special priority cases (short-circuited)

Before the main priority sort:
- **Fusible bitcasts** (no bit-width change): process first (priority = +∞, they're no-ops)
- **Constants**: process last in a separate pass (priority = -∞)

### 3.3 Keep Union-Find

The `UnionFind` class is reused as-is. The merge logic changes only in that we merge p into ALL consumer clusters in one step (multiple `uf.union()` calls for the same producer).

## Phase 4: Upgrade Fusion Rules

**File: `src/env/fusion_rules.py`**

### 4.1 Add root instruction check (XLA check #1)
```python
if instruction.is_root:
    return False  # cannot fuse the computation's root
```

### 4.2 Add bitcast consumer exclusion (XLA check #4)
```python
if consumer.opcode == "bitcast":
    return False  # cannot fuse into a standalone bitcast consumer
```

### 4.3 Add reduce-into-reduce prevention (XLA check #6)

Forbid fusing a producer that contains a significant reduction (≥16 elements reduced) into a consumer that also contains a reduction. This prevents cost model inaccuracies.

```python
REDUCTION_SIZE_THRESHOLD = 16

@classmethod
def _has_significant_reduce(cls, cluster_members, instruction_map):
    for name in cluster_members:
        inst = instruction_map.get(name)
        if inst and inst.opcode == "reduce":
            reduce_dims = inst.attributes.get("dimensions", [])
            # Check if reduction is over enough elements
            if inst.shape.num_elements >= REDUCTION_SIZE_THRESHOLD:
                return True
    return False
```

### 4.4 Add budget/code-size check (XLA checks #8, #9)

Add a parameter count limit:
```python
MAX_PARAMETERS = 64  # XLA's FusionFitsInBudget limit

@classmethod
def _parameter_count(cls, cluster_members, instruction_map, edges):
    """Count unique external inputs to the merged cluster."""
    # External inputs = operands whose producers are NOT in the cluster
    external_inputs = set()
    for name in cluster_members:
        inst = instruction_map.get(name)
        if inst:
            for op in inst.operand_names:
                if op not in cluster_members:
                    external_inputs.add(op)
    return len(external_inputs)
```

### 4.5 Update `can_fuse` method

Add the new checks in order (matching XLA's check sequence):
1. Root check (new)
2. Both fusable (existing)
3. Bitcast consumer exclusion (new)
4. Reduce-into-reduce (new)
5. Cluster size limit (existing)
6. Parameter budget (new)
7. Cycle detection (existing)

## Phase 5: Wire Everything Together in Environment

**File: `src/env/fusion_env.py`**

- No changes needed — it already passes `gpu_specs` through to `estimate_total_runtime()` and `simulate_fusion()`. The new `gpu_specs` fields are used automatically by the updated performance model.

**File: `src/env/fusion_simulator.py`** (already changed in Phase 3)

- The simulator's new all-or-nothing semantics are used automatically by the environment's `step()` method.

## Files Changed Summary

| # | File | Changes |
|---|------|---------|
| 1 | `src/utils/gpu_specs.py` | Add 3 fields, fix launch overhead to 1μs |
| 2 | `src/env/performance_model.py` | Overlap formula, utilization, coalescing, L1/L2 cache |
| 3 | `src/env/fusion_simulator.py` | All-or-nothing producer→all-consumers, special cases |
| 4 | `src/env/fusion_rules.py` | Root, bitcast, reduce-into-reduce, parameter budget |

## Verification

1. **Smoke test**: Run existing `scripts/smoke_test.py` — should still parse HLO, build graph, encode features, run model forward pass, and complete a PPO step without errors
2. **Unit tests**: After implementation, verify:
   - Performance model: cluster with 2 ops produces runtime using the overlap formula (not simple max)
   - Fusion simulator: producer with 2 consumers either fuses into both or neither
   - Fusion rules: reduce-into-reduce is blocked, root instruction can't be fused, parameter budget enforced
3. **Baseline comparison**: Run `scripts/evaluate.py` on the sample HLO and confirm the cost model produces different (more realistic) runtime estimates than before
