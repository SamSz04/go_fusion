# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GO Fusion — RL-based operator fusion optimizer for XLA HLO graphs. Reproduces the fusion task from "Transferable Graph Optimizers for ML Compilers" (GO, NeurIPS 2020) using PPO to learn priority assignments that control XLA-style operator fusion. Single-task fusion policy only (no device placement or scheduling).

## Commands

```bash
# Environment setup (conda)
conda activate go_fusion
pip install -e .

# Training
python scripts/train.py --config configs/default.yaml --hlo-dir ../hlos
python scripts/train.py --gpu a100 --num-updates 500 --lr 3e-4
python scripts/train_gpu.py   # standalone GPU training + eval

# Evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --hlo ../hlos/mlp_gpt/0060before_fusion.hlo
python scripts/quick_eval.py  # quick single-graph eval

# Tests (pytest, tests/ directory exists but is mostly empty)
pytest tests/
```

## Architecture

### Data Pipeline (end-to-end)

```
HLO text file
  → parser.py (parse_hlo_file) → HloModule
  → graph_builder.py (build_graph) → PyG Data (edges, topo sort)
  → feature_encoder.py (encode_features) → data.x [N,19] + data.opcode_ids [N]
  → GOFusionPolicy → probs [N,20] per-node priority distributions
  → FusionEnv.step() → fusion_simulator.simulate_fusion() → clusters
  → performance_model.estimate_total_runtime() → runtime → reward
```

### Module Responsibilities

- **`src/hlo_parser/`** — Parse XLA HLO text format into `HloModule` → build PyG graph → encode 19-dim continuous features + 132-opcode learnable embeddings
- **`src/model/`** — Policy network: `nn.Embedding(132,32)` + GraphSAGE(2 layers, max-pool) + SegmentedTransformer-XL(3 layers, 8 heads) with FeatureModulation gating, T=3 non-autoregressive refinement iterations. Value network: mean-pool + MLP
- **`src/env/`** — Gymnasium-compatible RL environment. Fusion simulator uses Union-Find with all-or-nothing producer→all-consumers semantics (XLA-aligned). Cost model: `max(C,M) + 0.05*min(C,M) + 1μs` with operand utilization, coalescing, L1/L2 cache modeling
- **`src/training/`** — PPO with clipped surrogate. Graph-level log prob = sum of per-node log probs. Single-step episodes. Reward = `-sqrt(runtime/baseline)`
- **`src/evaluation/`** — 5 baselines (no-fusion, random, greedy-memory, XLA-default, simulated annealing) + metrics/formatting
- **`src/utils/`** — GPU specs dataclass with V100/A100/H100 presets and XLA constants

### Import Convention

Source modules use absolute imports rooted at `src` (e.g., `from src.model.graphsage import GraphSAGE`). Exception: `src/hlo_parser/` uses relative imports internally. Scripts prepend the project root to `sys.path`.

### Configuration

All hyperparameters in `configs/default.yaml`: model (hidden_dim=128, 20 priorities, 132 opcodes), training (PPO clip=0.2, lr=1e-4, 4 epochs/update), environment (gpu_target, max_cluster_size=64).

### HLO Input

HLO files live at `../hlos/` (outside the repo). Primary: `mlp_gpt/0060before_fusion.hlo` (2048 nodes). Also: `vit/0064before_fusion.hlo` (271 nodes). On the GPU server (`ssh -p 1170 root@10.241.78.106`), HLO dumps are at `/root/hlo_dumps/`.

### Checkpoint Format

Dict with keys: `policy_state_dict`, `value_net_state_dict`, `optimizer_state_dict`, `mean_reward`, `best_reward`, `update`. Load with `weights_only=False` (PyTorch 2.6+).

## Key Design Decisions

- Fusion simulator uses **all-or-nothing** semantics: a producer fuses into ALL its non-barrier consumers or none, matching XLA's `PriorityFusion` pass
- 7 ordered legality checks mirror XLA: root, fusability, bitcast consumer, reduce-into-reduce (threshold=16), size cap (64), parameter budget (64), cycle detection
- Feature encoding: 19-dim continuous features + separate `nn.Embedding(132, 32)` for opcodes (inspired by TpuGraphs), concatenated before the GNN
- `FusionEnv` follows Gymnasium reset/step contract but does not subclass `gymnasium.Env`
