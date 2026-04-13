# Runtime Measurement Methodology — Technical Documentation

> **Scope:** All fusion optimization methods in the GO Fusion codebase
> **Reference graph:** `mlp_gpt` ENTRY computation (2,048 nodes, 3,616 edges)
> **Date:** 2026-04-02

---

## 1. Executive Summary

**Every runtime number reported in this project — across all six methods — is produced by a purely analytical cost model. No actual GPU kernels are compiled or executed; no hardware profiling tools are used.** The "runtime" is a mathematical estimate computed from statically-parsed HLO instruction metadata (tensor shapes, opcodes, data types) and tabulated GPU hardware specifications. All methods share the identical measurement pathway: priority assignment → graph partitioning via Union-Find → per-cluster analytical runtime estimation → summation.

| Aspect | Status |
|--------|--------|
| Actual hardware kernel execution | **No** |
| XLA compilation (`HloRunner`, `StreamExecutor`) | **No** |
| Cycle-accurate simulation | **No** |
| GPU profiling (`GpuTimer`, `nvprof`, `nsys`, `torch.cuda.Event`) | **No** |
| Analytical cost modeling | **Yes — sole method** |

---

## 2. The Shared Measurement Pipeline

Every method (GO policy, SA, Random, Greedy, XLA Default, No-Fusion) follows the same two-stage pipeline. They differ only in **Stage 0** (how priorities are assigned); Stages 1–2 are identical.

```
Stage 0: Assign priority p_i ∈ {0, ..., 19} to each fusable node    [method-specific]
                           │
                           ▼
Stage 1: simulate_fusion(instructions, edges, priorities)
         ├── Union-Find graph partitioning (all-or-nothing semantics)
         ├── 7 ordered legality checks per merge
         └── Returns List[FusionCluster]                              [pure CPU, no hardware]
                           │
                           ▼
Stage 2: estimate_total_runtime(clusters, instruction_map, gpu_specs)
         ├── For each cluster:
         │     compute_time  = Σ FLOPs(instr) / peak_FLOPS
         │     memory_time   = Σ read_time(operand) + Σ write_time(output)
         │     cluster_time  = max(C, M) + 0.05·min(C, M) + 1μs
         └── total_runtime   = Σ cluster_time                         [pure arithmetic]
```

### 2.1 Stage 1 — Fusion Simulation (`fusion_simulator.py`)

A **graph-partitioning algorithm**, not a hardware simulation:

1. Initialize each instruction as a singleton cluster (Union-Find with path compression + union-by-rank).
2. **Pre-pass:** Fuse all bitcasts into their consumers (priority = +∞; zero-cost ops).
3. **Main pass:** Iterate over fusable producers in descending priority order. For each producer, check if it can legally fuse into **all** its non-barrier consumers (all-or-nothing). If any consumer fails the 7-check legality sequence, skip the entire producer.
4. **Post-pass:** Fuse remaining scalar constants into their users (priority = −∞).
5. Collect final cluster sets from Union-Find.

Output: `List[Set[str]]` — each set is the instruction names comprising one fused kernel.

### 2.2 Stage 2 — Analytical Runtime Estimation (`performance_model.py`)

Each cluster's runtime is computed via XLA's `GpuPerformanceModel` formula:

$$T_{cluster} = \max(T_C,\; T_M) + 0.05 \times \min(T_C,\; T_M) + 1\mu s$$

**Compute time** ($T_C$):

$$T_C = \frac{\sum_{i \in cluster} \text{FLOPs}(i)}{\text{Peak\_FLOPS}}$$

FLOPs are estimated per-opcode: `add` → 1 FLOP/elem, `tanh` → 8, `dot` → 2MNK, etc. Peak FLOPS is selected from the GPU spec sheet (A100 FP32: 19.5 TFLOPS; FP16: 312 TFLOPS).

**Memory time** ($T_M$):

$$T_M = \sum_{op \in \text{operands}} \frac{\text{bytes}(op) \times U(op)}{BW \times C \times \alpha} + \sum_{out} \frac{\text{bytes}(out)}{BW}$$

| Factor | Symbol | Values | Source |
|--------|--------|--------|--------|
| Operand utilization | $U$ | broadcast: `out_elems/in_elems` (>1); slice: `out_elems/in_elems` (<1); default: 1.0 | Opcode heuristic |
| Coalescing coefficient | $C$ | transpose (innermost dim): 1/16; gather: 0.25; default: 1.0 | Opcode heuristic |
| Cache bandwidth speedup | $\alpha$ | Fits L1/shmem: 8×; fits L2: 2.5×; HBM: 1.0 | Size vs. cache capacity |
| Base bandwidth | $BW$ | A100: 2.0 TB/s | Spec sheet |

Write time is **elided** (set to 0) if all consumers of an output are within the same cluster — the intermediate stays in registers.

**Total runtime** = sum of all cluster runtimes, assuming sequential single-stream execution.

### 2.3 Hardware Specifications Used

All GPU parameters come from the `GPUSpecs` dataclass (`gpu_specs.py`) — static spec-sheet values, not measured at runtime:

| Parameter | V100 | A100 | H100 |
|-----------|------|------|------|
| FP32 Peak (TFLOPS) | 15.7 | 19.5 | 67.0 |
| FP16 Peak (TFLOPS) | 125.0 | 312.0 | 989.0 |
| Memory BW (TB/s) | 0.9 | 2.0 | 3.35 |
| L2 Cache (MB) | 6 | 40 | 50 |
| Shmem/SM (KB) | 96 | 164 | 228 |
| Launch overhead | 1 μs | 1 μs | 1 μs |

---

## 3. Method-Specific Workflows

### 3.1 GO Policy (Deterministic)

```
HLO graph → Policy network forward pass (GraphSAGE + Transformer, on GPU)
         → probs [N, 20]
         → argmax per node → priorities Dict[str, int]
         → simulate_fusion() → estimate_total_runtime()
         → reported as "GO Deterministic runtime"
```

- **One** forward pass through the 784K-parameter network.
- **One** fusion simulation + cost model evaluation.
- The GPU is used only for neural network inference; the fusion simulation and runtime estimation are CPU-only.

### 3.2 GO Policy (Stochastic Best-of-K)

```
For k = 1..K (K=10):
    Sample actions ~ Categorical(probs)     [stochastic]
    simulate_fusion() → estimate_total_runtime() → runtime_k
Report min(runtime_1, ..., runtime_K)
```

- **K** independent samples from the learned policy distribution.
- **K** fusion simulations + cost model evaluations.
- Reports the minimum runtime found across all K samples.

### 3.3 Simulated Annealing

```
Initialize: random priorities for all fusable nodes
For iter = 1..N_iters (typically 1000–10000):
    Randomly select one node, change its priority
    simulate_fusion() → estimate_total_runtime() → new_runtime
    If new_runtime < current_runtime: accept
    Else: accept with probability exp(-Δ / (T × best_runtime))
    T *= cooling_rate (geometric cooling)
Report best_runtime found across all iterations
```

- **N_iters** fusion simulations + cost model evaluations (the dominant computational cost).
- For `mlp_gpt` (2,048 nodes), each iteration runs the full Union-Find partitioning over 2,048 nodes and 3,616 edges, then estimates runtime for all resulting clusters.
- At 10,000 iterations on CPU, this takes approximately 1+ hours for the 2,048-node graph.

### 3.4 Random Priority

```
For seed = 1..10:
    Assign each fusable node a uniform random priority ∈ {0..19}
    simulate_fusion() → estimate_total_runtime() → runtime_seed
Report mean(runtime_1, ..., runtime_10)
```

- **10** fusion simulations + cost model evaluations.
- Reports the **mean** (not best) to characterize expected performance of random assignment.

### 3.5 Greedy Memory

```
For each fusable instruction:
    score = output_bytes × (1 + num_consumers)   [memory savings heuristic]
Normalize and quantize scores into 20 priority bins
simulate_fusion() → estimate_total_runtime()
Report runtime
```

- **One** fusion simulation + cost model evaluation.
- Deterministic — produces the same result every time for the same graph.

### 3.6 XLA Default (Reverse Topological Order)

```
Topological sort (Kahn's algorithm) over the HLO graph
Assign priorities: node at topo position i → priority ∝ (N - i)
Quantize into 20 bins
simulate_fusion() → estimate_total_runtime()
Report runtime
```

- **One** fusion simulation + cost model evaluation.
- Deterministic. Approximates XLA's default fusion order when no cost model is available.

### 3.7 No-Fusion Baseline

```
Create one singleton cluster per fusable instruction (no merging at all)
estimate_total_runtime(singleton_clusters) → report runtime
```

- **Zero** fusion simulation (skips `simulate_fusion()` entirely).
- **One** cost model evaluation on 2,048 singleton clusters.
- Serves as the normalization reference: all speedup ratios are computed against this value.

---

## 4. Measurement Characteristics and Limitations

### 4.1 What the Results Represent

| Property | Value |
|----------|-------|
| **Nature of "runtime"** | Analytical estimate (seconds), not measured latency |
| **Hardware dependency** | Parameterized by GPU spec sheet; no actual hardware in the loop |
| **Reproducibility** | Fully deterministic for fixed priorities (GO-det, Greedy, XLA Default, No-Fusion). Stochastic for Random (seed-dependent) and SA (random walk). GO-stochastic varies per sample. |
| **Unit** | Seconds (e.g., 4.941×10⁻³ s for GO-det on mlp_gpt) |

### 4.2 Optimization Overhead vs. Reported Runtime

The "runtime" is the estimated execution time of the fused HLO graph on the target GPU. The time spent finding the fusion strategy (the "optimization overhead") is a separate, orthogonal cost:

| Method | Optimization Overhead (mlp_gpt, 2048 nodes) | Cost Model Calls |
|--------|---------------------------------------------|-----------------|
| No-Fusion | ~0 (trivial) | 1 |
| Greedy Memory | < 1 sec | 1 |
| XLA Default | < 1 sec | 1 |
| Random (10 seeds) | ~5 sec | 10 |
| GO Deterministic | ~0.5 sec (single forward pass) | 1 |
| GO Stochastic (K=10) | ~2 sec (10 forward passes + 10 evaluations) | 10 |
| Simulated Annealing (1000 iters) | ~5–10 min on CPU | 1,000 |
| Simulated Annealing (10000 iters) | ~1+ hour on CPU | 10,000 |

The GO policy's inference-time advantage is significant: a single O(N) forward pass replaces thousands of iterative search evaluations.

### 4.3 Measurement Stability on the 2,048-Node Graph

Since the cost model is purely deterministic (for fixed priorities), there is **zero measurement variance** for deterministic methods (GO-det, Greedy, XLA Default, No-Fusion). Variance arises only from the optimization algorithm itself:

| Method | Source of Variance | Mitigation |
|--------|--------------------|------------|
| GO Deterministic | None (argmax is deterministic) | N/A |
| GO Stochastic | Sampling randomness | Best-of-K selection (K=10) |
| Random | Seed randomness | Mean over 10 seeds |
| SA | Random walk trajectory | Best-across-all-iterations |
| Greedy / XLA Default / No-Fusion | None | N/A |

The 2,048-node scale does not introduce measurement instability in the cost model itself, but it significantly impacts the optimization search difficulty: the action space is 20²⁰⁴⁸, making exhaustive search intractable and necessitating the RL approach.

### 4.4 Known Accuracy Gaps

The analytical cost model, while aligned with XLA's `GpuPerformanceModel` formulation, has inherent limitations compared to actual hardware execution:

| Gap | Description | Impact |
|-----|-------------|--------|
| **Register pressure** | No modeling of register spilling when clusters are too large | May overestimate benefits of aggressive fusion |
| **Instruction scheduling** | Assumes peak ILP; real GPU warp schedulers may achieve less | Compute time may be underestimated |
| **Occupancy effects** | No SM occupancy modeling (register/shmem limits → fewer warps) | Bandwidth utilization may be overestimated |
| **Kernel codegen** | Different fusion patterns may produce vastly different PTX/SASS | Two clusters with identical analytical estimates could have 2–5× real-time differences |
| **Multi-stream overlap** | Assumes sequential single-stream execution | May overestimate total runtime if independent kernels can overlap |
| **Per-opcode costs** | Uses heuristic FLOP multipliers, not XLA's `hlo_op_profiles_data.h` | Compute time accuracy varies by opcode |

---

## 5. Summary

All six methods in this project — GO (deterministic and stochastic), Simulated Annealing, Random, Greedy Memory, XLA Default, and No-Fusion — use an **identical analytical cost model** to produce their reported runtime numbers. No actual GPU compilation, kernel execution, or hardware profiling occurs at any point in the evaluation pipeline. The cost model implements XLA's `GpuPerformanceModel` formula with compute-memory overlap (95%), operand utilization, coalescing approximation, and L1/L2 cache bandwidth modeling, parameterized by static GPU spec-sheet values.

The reported runtimes (e.g., GO deterministic = 4.941×10⁻³ s on mlp_gpt) should be interpreted as **analytically estimated execution times under idealized conditions**, not as measured hardware latencies. They are internally consistent and valid for relative comparison between methods, but their absolute accuracy against real hardware execution remains unvalidated — a gap explicitly identified in the project's future work plan (cost model calibration via actual XLA kernel profiling).
