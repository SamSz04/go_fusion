# GO Fusion v1 — Code Layout & Architecture

> **Project:** Reproducing the operation fusion task from *"Transferable Graph Optimizers for ML Compilers"* (GO, NeurIPS 2020)
>
> **Scope:** Single-task fusion policy only — no device placement, scheduling, or multi-task policy network.
>
> **Date:** 2026-03-18

---

## 1. Directory Structure

```
priorityfusion/
├── hlos/                                    # HLO input files (not part of go_fusion)
│   └── mlp_gpt/
│       ├── 0000before_opt.hlo              # Pre-optimization (1,094 instructions)
│       └── 0060before_fusion.hlo           # Pass 60 before fusion (2,048 ENTRY instructions) ← PRIMARY INPUT
│
├── go_fusion/                               # Main project directory
│   ├── pyproject.toml                       # Python packaging config
│   ├── requirements.txt                     # pip dependencies
│   │
│   ├── configs/
│   │   └── default.yaml                     # All hyperparameters
│   │
│   ├── scripts/
│   │   ├── train.py                         # CLI: launch training
│   │   └── evaluate.py                      # CLI: evaluate against baselines
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   │
│   │   ├── hlo_parser/                      # Phase 1: HLO text → PyG graph
│   │   │   ├── __init__.py
│   │   │   ├── hlo_ir.py                    # Data classes: HloShape, HloInstruction, HloComputation, HloModule
│   │   │   ├── parser.py                    # Regex-based line-by-line HLO text parser
│   │   │   ├── graph_builder.py             # HloModule ENTRY → PyG Data (nodes, edges, topo sort)
│   │   │   └── feature_encoder.py           # Encode 19-dim continuous features + opcode_ids
│   │   │
│   │   ├── model/                           # Phase 4: Neural network components
│   │   │   ├── __init__.py
│   │   │   ├── graphsage.py                 # GraphSAGE with max-pool aggregation (2 layers)
│   │   │   ├── feature_modulation.py        # Graph-conditioned sigmoid gating + LayerNorm
│   │   │   ├── segmented_transformer.py     # Segmented Transformer-XL with recurrence (3 layers)
│   │   │   ├── policy_network.py            # Full policy: Embedding + GNN + Transformer → softmax
│   │   │   └── value_network.py             # Critic: mean pool → MLP → scalar
│   │   │
│   │   ├── env/                             # Phase 2–3: Fusion environment
│   │   │   ├── __init__.py
│   │   │   ├── performance_model.py         # Roofline cost model (FLOP + byte estimation)
│   │   │   ├── fusion_rules.py              # Fusion legality constraints (XLA-aligned)
│   │   │   ├── fusion_simulator.py          # Priority-based fusion via Union-Find
│   │   │   └── fusion_env.py                # Gymnasium RL environment (reset/step)
│   │   │
│   │   ├── training/                        # Phase 5: RL training
│   │   │   ├── __init__.py
│   │   │   ├── ppo.py                       # PPO (clipped surrogate + value + entropy)
│   │   │   └── trainer.py                   # Training loop with TensorBoard + checkpoints
│   │   │
│   │   ├── evaluation/                      # Phase 6: Baselines & metrics
│   │   │   ├── __init__.py
│   │   │   ├── baselines.py                 # 5 baselines: no-fusion, random, greedy, XLA-default, SA
│   │   │   └── metrics.py                   # Speedup, memory reduction, cluster stats, table formatting
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── gpu_specs.py                 # GPUSpecs dataclass + V100/A100/H100 presets
│   │
│   └── tests/                               # (empty — tests not yet written)
```

**Total: 22 Python source files + 1 YAML config + 2 packaging files = 25 files**

---

## 2. System Architecture

### 2.1 End-to-End Pipeline

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  HLO Text   │─────▶│    HLO Parser    │─────▶│   Graph Builder   │
│ (.hlo file) │      │    (parser.py)   │      │ (graph_builder.py)│
└─────────────┘      └──────────────────┘      └─────────┬─────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  Feature Encoder │
                                                │ (feature_encoder)│
                                                └────────┬─────────┘
                                                         │
                              ┌──────────────────────────┤
                              │                          │
                              ▼                          ▼
                     ┌────────────────┐          ┌────────────────┐
                     │ data.x [N,19]  │          │ data.opcode_ids│
                     │ (continuous)   │          │      [N]       │
                     └───────┬────────┘          └───────┬────────┘
                             │                           │
                             └─────────────┬─────────────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │ Policy Network │──── probs [N, 20]
                                  │ (GOFusionPolicy│──── embeddings [N, 128]
                                  │  + ValueNet)   │──── value (scalar)
                                  └───────┬────────┘
                                          │ actions [N]
                                          ▼
                                  ┌────────────────┐
                                  │   Fusion Env   │──── reward (scalar)
                                  │ (fusion_env.py)│
                                  └───────┬────────┘
                                          │
                                ┌─────────┴─────────┐
                                ▼                   ▼
                       ┌────────────────┐  ┌────────────────┐
                       │Fusion Simulator│  │Performance     │
                       │(Union-Find     │  │Model (Roofline)│
                       │ + FusionRules) │  │                │
                       └────────────────┘  └────────────────┘
```

### 2.2 Policy Network (GOFusionPolicy) — Detailed

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GOFusionPolicy (776,468 params)                       │
│                                                                             │
│   Inputs:  x [N, 19]          continuous node features                      │
│            opcode_ids [N]     integer opcode indices (0..131)               │
│            edge_index [2, E]  COO adjacency                                 │
│                                                                             │
│   ┌── Iteration t = 1..T (T=3 non-autoregressive refinement) ──────────┐    │
│   │                                                                    │    │
│   │  opcode_emb = nn.Embedding(132, 32)(opcode_ids)        → [N, 32]   │    │ 
│   │  h = cat(opcode_emb, x, prev_actions)                  → [N, 71]   │    │
│   │  h = ReLU(Linear(71, 128)(h))                          → [N, 128]  │    │
│   │                                                                    │    │
│   │  h = GraphSAGE(h, edge_index)          2 layers        → [N, 128]  │    │
│   │       └─ MaxPool aggregation: ReLU(W_pool · h_neighbor)            │    │
│   │       └─ ReLU(W · concat(h_self, max_pool(neighbors)))             │    │
│   │                                                                    │    │
│   │  h_G = mean(h, dim=0)                  graph embedding → [128]     │    │
│   │                                                                    │    │
│   │  h = SegmentedTransformer(h, h_G)      3 layers, 8 heads           │    │
│   │       └─ Segment-level recurrence (segment_size=256)               │    │
│   │       └─ Per-layer FeatureModulation: LN(h * σ(W · h_G))           │    │
│   │       └─ No positional encoding                                    │    │
│   │                                                                    │    │
│   │  logits = Linear(128, 20)(h)                           → [N, 20]   │    │
│   │  probs  = softmax(logits)                              → [N, 20]   │    │
│   │  prev_actions = probs.detach()   ◄── feed back to next iteration   │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   Outputs: probs [N, 20]       per-node priority distributions              │
│            h [N, 128]          node embeddings → ValueNetwork               │
│                                                                             │
│   ValueNetwork: mean_pool(h) → Linear(128,64) → ReLU → Linear(64,1)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 PPO Training Loop

```
For each update (1..1000):
  For each rollout (1..16):
    1. env.reset()  →  PyG Data (x, edge_index, opcode_ids)
    2. policy.forward(x, edge_index, opcode_ids)  →  probs [N, 20]
    3. Categorical(probs).sample()  →  actions [N]
    4. env.step(actions)  →  reward (scalar)
       ├── simulate_fusion(instructions, edges, priorities)  →  clusters
       └── estimate_total_runtime(clusters, gpu_specs)  →  runtime
           reward = -sqrt(runtime / baseline_runtime)

  PPO.update(rollouts):
    For each epoch (1..4):
      For each rollout:
        - Re-evaluate policy: new_probs, new_value
        - ratio = exp(new_log_prob - old_log_prob)   (graph-level: sum over nodes)
        - advantage = reward - old_value
        - policy_loss = -min(ratio * A, clip(ratio, 1±0.2) * A)
        - value_loss = MSE(new_value, reward)
        - loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        - Adam step (lr=1e-4, grad_clip=0.5)
```

---

## 3. Module Specifications

### 3.1 HLO Parser (`src/hlo_parser/`)

| File | Lines | Purpose |
|------|-------|---------|
| `hlo_ir.py` | 248 | Data classes: `HloShape` (tensor/tuple shapes with `num_elements`, `total_bytes`), `HloInstruction` (with `is_fusable`, `is_gemm`, `called_computation`), `HloComputation`, `HloModule` |
| `parser.py` | ~587 | Line-by-line regex parser for XLA HLO text format. Handles nested tuple shapes, JSON `backend_config` (brace-depth tracking), `parameter(N)`/`constant(val)` special opcodes, metadata tables |
| `graph_builder.py` | ~120 | ENTRY computation → PyG `Data`. Forward + reverse edges (COO), Kahn's topological sort, `is_fusable` mask, stores `instructions`, `node_names`, `node_opcodes` on Data |
| `feature_encoder.py` | ~250 | **132-opcode vocabulary** (full XLA `hlo_opcode.h`). Outputs `data.x [N, 19]` (continuous) + `data.opcode_ids [N]` (integer). No opcode one-hot — embedding is in the model |

**19 continuous features:**

| Feature | Dims | Encoding |
|---------|------|----------|
| Element type | 6 | One-hot (f32, s32, u32, u64, pred, bf16) |
| Rank | 1 | Scalar (number of dimensions) |
| Dimension sizes | 4 | log2 of each dim, zero-padded to max_rank=4 |
| Total elements | 1 | log2(product of dims) |
| Total bytes | 1 | log2(elements × dtype_size) |
| Fan-in | 1 | Number of operand inputs |
| Fan-out | 1 | Number of consumers (from edge_index) |
| Shape-compatible inputs | 1 | Boolean heuristic |
| Is fusion node | 1 | opcode == "fusion" |
| Is GEMM custom-call | 1 | custom-call with `__cublas$gemm` |
| Is fusable | 1 | Not parameter/constant/get-tuple-element/tuple |

### 3.2 Neural Network (`src/model/`)

| File | Lines | Purpose |
|------|-------|---------|
| `graphsage.py` | ~80 | Custom `GraphSAGEMaxPoolLayer(MessagePassing)` with `aggr='max'`. Pre-transforms neighbors via `lin_pool` before max. `GraphSAGE` wraps input projection + N layers |
| `feature_modulation.py` | 47 | `FeatureModulation`: `LayerNorm(x * σ(Linear(h_G)))` — graph-level gating per Transformer layer |
| `segmented_transformer.py` | ~180 | `SegmentedTransformerLayer`: pre-norm MHA with recurrence (cached KV from previous segment, detached). `SegmentedTransformer`: splits nodes into segments of 256, `forward_with_modulation()` applies per-layer gating |
| `policy_network.py` | ~155 | `GOFusionPolicy`: `nn.Embedding(132,32)` + T=3 iterative refinement. Forward: `cat(embed, x, prev_actions) → proj → GraphSAGE → Transformer(+modulation) → softmax`. `get_action()` for PPO sampling |
| `value_network.py` | 42 | `ValueNetwork`: `mean_pool → Linear(128,64) → ReLU → Linear(64,1)` |

**Model parameters:** 776,468 (policy) + 8,321 (value) = **784,789 total**

### 3.3 Fusion Environment (`src/env/`)

| File | Lines | Purpose |
|------|-------|---------|
| `performance_model.py` | ~305 | **Roofline model**: `runtime = max(FLOPs/peak_FLOPS, bytes/bandwidth) + launch_overhead`. Per-opcode FLOP estimation (recursive for fusion/custom-call with cycle guard). Byte traffic with fusion-aware elision of intermediate reads/writes |
| `fusion_rules.py` | ~160 | `FusionRules`: opcode categories, `can_fuse()` checks (fusability, data edge, size limit ≤64, cycle detection via BFS on cluster-level DAG) |
| `fusion_simulator.py` | ~150 | `UnionFind` (path compression + union-by-rank). `simulate_fusion()`: sort by priority descending, greedily merge producer→consumer clusters. Returns `List[FusionCluster]` |
| `fusion_env.py` | ~200 | `FusionEnv`: Gymnasium-compatible. `reset()` returns PyG Data, `step(actions)` → simulate fusion → roofline runtime → reward = `-√(runtime/baseline)`, penalty `-10` for invalid |

### 3.4 Training (`src/training/`)

| File | Lines | Purpose |
|------|-------|---------|
| `ppo.py` | ~197 | `PPO`: graph-level log prob = sum of per-node log probs. Clipped surrogate + MSE value loss + entropy bonus. Single Adam optimizer. 4 epochs per update |
| `trainer.py` | ~415 | `Trainer`: loads HLO files → environments, `collect_rollout()` (round-robin), `train()` loop, TensorBoard logging, checkpoint save/load (policy + value + optimizer state) |

### 3.5 Evaluation (`src/evaluation/`)

| File | Lines | Purpose |
|------|-------|---------|
| `baselines.py` | ~300 | 5 strategies: `no_fusion` (singleton clusters), `random_priority` (avg over seeds), `greedy_memory` (priority ∝ memory savings), `xla_default` (reverse topo order), `simulated_annealing` (geometric cooling) |
| `metrics.py` | ~252 | `FusionMetrics`: speedup ratios, percentage improvements, memory reduction, cluster statistics (count/avg/max/min/median size), formatted comparison tables |

### 3.6 Utilities (`src/utils/`)

| File | Lines | Purpose |
|------|-------|---------|
| `gpu_specs.py` | ~142 | `GPUSpecs` dataclass + presets: V100 (15.7 TFLOPS, 900 GB/s), A100 (19.5 TFLOPS, 2.0 TB/s), H100 (67 TFLOPS, 3.35 TB/s). Derived properties: `arithmetic_intensity`, `total_shared_memory` |

---

## 4. Data Flow & Representations

### 4.1 Graph Representation

The graph is stored as a PyTorch Geometric `Data` object with the following attributes:

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `x` | `[N, 19]` | float32 | Continuous node features |
| `opcode_ids` | `[N]` | int64 | Opcode indices for `nn.Embedding` |
| `edge_index` | `[2, E]` | int64 | Forward edges (producer → consumer), COO format |
| `reverse_edge_index` | `[2, E]` | int64 | Reverse edges (consumer → producer) |
| `topo_order` | `[N]` | int64 | Topological sort order (Kahn's algorithm) |
| `is_fusable` | `[N]` | bool | Whether each node can participate in fusion |
| `instructions` | list | HloInstruction | Raw instruction objects |
| `node_names` | list | str | Instruction names |
| `node_opcodes` | list | str | Opcode strings |
| `hlo_module` | - | HloModule | Reference to the parsed module |

For the MLP-GPT graph: **N=2,048 nodes, E=3,616 edges**, 24 unique opcodes, 1,913 fusable nodes.

### 4.2 Adjacency Representation

No dense adjacency matrix. Two sparse formats:

1. **COO `edge_index [2, E]`** — for the GNN (PyG standard). Memory: 2 × 3,616 × 8 bytes = ~58 KB
2. **Name-pair edge lists `List[Tuple[str, str]]`** — for fusion simulator and baselines. Converted to adjacency dicts (`Dict[str, List[str]]`) for BFS/topo sort as needed

### 4.3 Action Space

- **|F| = 20** discrete priority levels per node
- Each node is assigned a priority class independently (non-autoregressive)
- Higher priority → merged earlier in the greedy fusion pass
- Graph-level action: joint assignment of all N nodes → `[N]` integer tensor

---

## 5. Hyperparameters (configs/default.yaml)

### Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_dim` | 128 | Used throughout GNN, Transformer, projections |
| `num_node_features` | 19 | Continuous features (opcode via embedding) |
| `num_opcodes` | 132 | Full XLA opcode vocabulary |
| `opcode_embed_dim` | 32 | Learnable opcode embedding dimension |
| `num_gnn_layers` | 2 | GraphSAGE layers |
| `num_transformer_layers` | 3 | Segmented Transformer layers |
| `num_attention_heads` | 8 | Multi-head attention |
| `segment_size` | 256 | Transformer-XL segment length |
| `num_priorities` | 20 | Action space size |F| |
| `num_iterations` | 3 | Non-autoregressive refinement iterations T |
| `ff_dim` | 512 | Transformer feed-forward inner dim |
| `dropout` | 0.1 | Applied in Transformer layers |

### Training (PPO)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 1e-4 |
| `clip_ratio` | 0.2 |
| `entropy_coeff` | 0.01 |
| `value_loss_coeff` | 0.5 |
| `max_grad_norm` | 0.5 |
| `num_epochs_per_update` | 4 |
| `rollouts_per_update` | 16 |
| `num_updates` | 1000 |

### Environment

| Parameter | Value |
|-----------|-------|
| `gpu_target` | V100 |
| `max_cluster_size` | 64 |
| `invalid_reward` | -10.0 |

---

## 6. Dependency Graph

```
                         hlo_ir.py ◄─────────────────────────────────────────┐
                            ▲                                                 │
              ┌─────────────┼──────────────┐                                  │
              │             │              │                                  │
          parser.py   graph_builder.py  feature_encoder.py                    │
              │             │              │                                  │
              └─────────────┼──────────────┘                                  │
                            │                                                 │
                            ▼                                                 │
    gpu_specs.py ◄──── performance_model.py                                   │ 
         ▲                  ▲                                                 │
         │                  │                                                 │
         │            fusion_rules.py ◄───── fusion_simulator.py              │
         │                  ▲                      ▲                          │
         │                  │                      │                          │
         └──────────── fusion_env.py ──────────────┘                          │
                            │                                                 │
                            │                                                 │
                            │   graphsage.py ◄──┐                             │
                            │                    │                            │
                            │   feature_modulation.py ◄── segmented_transformer.py
                            │        ▲                          ▲             │
                            │        │                          │             │
                            │        └──── policy_network.py ───┘             │
                            │                    │                            │
                            │              value_network.py                   │
                            │                    │                            │
                            │               ppo.py                            │
                            │                    │                            │
                            └──────────── trainer.py ────────────────────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                                train.py                 evaluate.py
                                                              │
                                                         baselines.py
                                                         metrics.py
```

---

## 7. Verified Smoke Test Results (MLP-GPT Graph)

| Test | Result |
|------|--------|
| HLO Parse | 76 computations, 2,048 ENTRY nodes, 3,616 edges |
| Feature Encode | `x: [2048, 19]`, `opcode_ids: [2048]`, 24 unique opcodes mapped to [3, 131] |
| Model Forward | `probs: [2048, 20]`, `embeddings: [2048, 128]`, 776K params, gradients flow |
| PPO Update | reward -0.93 → -0.91, policy_loss decreasing, entropy ~2.84 |
| No-fusion Baseline | 16.6ms (reference) |
| Random Fusion | 13.6ms (1.22× speedup) |
| Greedy Memory | 15.2ms (1.09× speedup) |
| XLA Default | 16.3ms (1.02× speedup) |

---

## 8. Dependencies

```
torch>=2.1.0
torch-geometric>=2.4.0
gymnasium>=0.29.0
pyyaml>=6.0
matplotlib>=3.7.0
networkx>=3.0
tensorboard>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
pytest>=7.4.0
```

**Runtime environment:** conda env `go_fusion` (Python 3.11), macOS Darwin 23.6.0

---

## 9. How to Run

```bash
# Activate environment
conda activate go_fusion

# Train
cd priorityfusion/go_fusion
python3 scripts/train.py --config configs/default.yaml --hlo-dir ../hlos

# Evaluate
python3 scripts/evaluate.py --checkpoint checkpoints/best_model.pt --hlo ../hlos/mlp_gpt/0060before_fusion.hlo
```
