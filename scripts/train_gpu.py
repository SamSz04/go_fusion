"""
Full GPU-accelerated training pipeline for GO Fusion.

Trains the RL policy on all available HLO graphs using PPO with
GPU acceleration. Evaluates against baselines at the end.
"""
import os
import sys
import time
import glob as globmod
import random

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hlo_parser.parser import parse_hlo_file
from src.hlo_parser.graph_builder import build_graph
from src.hlo_parser.feature_encoder import encode_features
from src.env.fusion_env import FusionEnv
from src.env.fusion_simulator import simulate_fusion
from src.env.performance_model import estimate_total_runtime
from src.env.fusion_rules import FusionRules
from src.evaluation.baselines import run_all_baselines
from src.evaluation.metrics import FusionMetrics
from src.model.policy_network import GOFusionPolicy
from src.model.value_network import ValueNetwork
from src.training.ppo import PPO
from src.utils.gpu_specs import get_gpu_specs


# =====================================================================
# Configuration
# =====================================================================
NUM_UPDATES = 2000
ROLLOUTS_PER_UPDATE = 8
LR = 3e-4
GPU_TARGET = "a100"  # cost model target (doesn't need to match training GPU)
HLO_DIR = "../hlos"
CHECKPOINT_DIR = "./checkpoints"
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 200
SA_ITERATIONS = 1000  # for final evaluation

# Graphs with trivial fusion spaces (too few fusable nodes for RL to add value)
EXCLUDE_DIRS = {"mlp_down_proj", "gqa_out_proj"}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_specs = get_gpu_specs(GPU_TARGET)
print(f"Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# =====================================================================
# Load environments
# =====================================================================
print(f"\nLoading HLO files from {HLO_DIR}...")
hlo_files = sorted(globmod.glob(os.path.join(HLO_DIR, "**", "*before_fusion*.hlo"), recursive=True))
if not hlo_files:
    hlo_files = sorted(globmod.glob(os.path.join(HLO_DIR, "**", "*.hlo"), recursive=True))

envs = []
env_names = []

for hlo_file in hlo_files:
    # Skip graphs with trivially small fusion spaces
    parent_dir = os.path.basename(os.path.dirname(hlo_file))
    if parent_dir in EXCLUDE_DIRS:
        print(f"  Skipped (excluded): {parent_dir}/{os.path.basename(hlo_file)}")
        continue

    try:
        module = parse_hlo_file(hlo_file)
        graph_data = build_graph(module)
        graph_data = encode_features(graph_data)
        instructions = graph_data.instructions
        node_names = graph_data.node_names
        edge_index = graph_data.edge_index
        edges = []
        for i in range(edge_index.shape[1]):
            edges.append((node_names[edge_index[0, i].item()], node_names[edge_index[1, i].item()]))
        computation_map = module.computations
        env = FusionEnv(graph_data, instructions, edges, gpu_specs, computation_map=computation_map)
        envs.append(env)
        env_names.append(os.path.basename(os.path.dirname(hlo_file)) + "/" + os.path.basename(hlo_file))
        print(f"  Loaded: {env_names[-1]} ({graph_data.num_nodes} nodes, {env.num_fusable} fusable, baseline={env.baseline_runtime:.4e}s)")
    except Exception as e:
        print(f"  Failed: {hlo_file}: {e}")

if not envs:
    print("ERROR: No environments loaded!")
    sys.exit(1)

print(f"\nTotal environments: {len(envs)}")

# =====================================================================
# Build model
# =====================================================================
policy = GOFusionPolicy(
    num_node_features=19, hidden_dim=128, num_priorities=20,
    num_opcodes=132, opcode_embed_dim=32,
    num_gnn_layers=2, num_transformer_layers=3, num_heads=8,
    segment_size=256, num_iterations=3, d_ff=512, dropout=0.1,
).to(device)
value_net = ValueNetwork(hidden_dim=128).to(device)
ppo = PPO(policy, value_net, lr=LR, clip_ratio=0.2, entropy_coeff=0.01,
          value_loss_coeff=0.5, max_grad_norm=0.5, num_epochs=4)

total_params = sum(p.numel() for p in policy.parameters()) + sum(p.numel() for p in value_net.parameters())
print(f"Model parameters: {total_params:,}")
print(f"\nTraining: {NUM_UPDATES} updates x {ROLLOUTS_PER_UPDATE} rollouts, lr={LR}")
print("=" * 80)

# =====================================================================
# Training loop
# =====================================================================
best_reward = float("-inf")
reward_history = []
start_time = time.time()

for update in range(1, NUM_UPDATES + 1):
    update_start = time.time()
    rollouts = []
    episode_rewards = []
    episode_infos = []

    for r in range(ROLLOUTS_PER_UPDATE):
        env = envs[r % len(envs)]

        policy.eval()
        value_net.eval()
        with torch.no_grad():
            g = env.reset()
            obs_x = g.x.to(device)
            obs_ei = g.edge_index.to(device)
            obs_oc = g.opcode_ids.to(device)
            probs, emb = policy(obs_x, obs_ei, obs_oc)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            value = value_net(emb).item()
            _, reward, done, info = env.step(actions.cpu())

        policy.train()
        value_net.train()
        rollouts.append({
            "obs_x": obs_x, "obs_edge_index": obs_ei,
            "obs_opcode_ids": obs_oc, "actions": actions,
            "old_log_probs": log_probs, "reward": reward, "value": value,
        })
        episode_rewards.append(reward)
        episode_infos.append(info)

    metrics = ppo.update(rollouts)
    mean_reward = np.mean(episode_rewards)
    reward_history.append(mean_reward)

    if mean_reward > best_reward:
        best_reward = mean_reward
        torch.save({
            "update": update, "policy_state_dict": policy.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "mean_reward": mean_reward, "best_reward": best_reward,
        }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    elapsed = time.time() - update_start

    if update % LOG_INTERVAL == 0 or update == 1:
        runtimes = [info.get("runtime", 0) for info in episode_infos]
        n_clusters = [info.get("num_clusters", 0) for info in episode_infos]
        print(
            f"Update {update:4d}/{NUM_UPDATES} | "
            f"Reward: {mean_reward:+.4f} (best: {best_reward:+.4f}) | "
            f"PL: {metrics['policy_loss']:+.5f} | "
            f"VL: {metrics['value_loss']:.4f} | "
            f"Ent: {metrics['entropy']:.3f} | "
            f"Clusters: {np.mean(n_clusters):.0f} | "
            f"{elapsed:.1f}s"
        )
        sys.stdout.flush()

    if update % CHECKPOINT_INTERVAL == 0:
        torch.save({
            "update": update, "policy_state_dict": policy.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "mean_reward": mean_reward, "best_reward": best_reward,
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_{update:06d}.pt"))
        print(f"  Saved checkpoint_{update:06d}.pt")

total_time = time.time() - start_time
print("=" * 80)
print(f"Training complete in {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
print(f"Best reward: {best_reward:+.4f}")
print(f"Reward trend: {reward_history[0]:+.4f} -> {reward_history[-1]:+.4f}")

# Save final model
torch.save({
    "update": NUM_UPDATES, "policy_state_dict": policy.state_dict(),
    "value_net_state_dict": value_net.state_dict(),
    "optimizer_state_dict": ppo.optimizer.state_dict(),
    "mean_reward": mean_reward, "best_reward": best_reward,
}, os.path.join(CHECKPOINT_DIR, "final_model.pt"))
print("Saved final_model.pt")

# =====================================================================
# Evaluation
# =====================================================================
print("\n" + "=" * 80)
print("  EVALUATION: GO Policy vs Baselines")
print("=" * 80)

# Load best model for evaluation
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=device, weights_only=False)
policy.load_state_dict(ckpt["policy_state_dict"])
print(f"Loaded best_model.pt (update={ckpt['update']}, reward={ckpt['mean_reward']:+.4f})")

policy.eval()

for env_idx, (env, name) in enumerate(zip(envs, env_names)):
    print(f"\n--- {name} ({env.num_fusable} fusable nodes) ---")
    g = env.reset()
    instructions = g.instructions
    node_names_list = g.node_names
    ei = g.edge_index
    edges = [(node_names_list[ei[0, i].item()], node_names_list[ei[1, i].item()]) for i in range(ei.shape[1])]
    instruction_map = {inst.name: inst for inst in instructions}
    computation_map_env = env.computation_map if hasattr(env, "computation_map") else {}

    # GO policy (deterministic)
    with torch.no_grad():
        obs_x = g.x.to(device)
        obs_ei = g.edge_index.to(device)
        obs_oc = g.opcode_ids.to(device)
        probs, _ = policy(obs_x, obs_ei, obs_oc)
        actions_det = probs.argmax(dim=-1)

    priorities = {}
    for i, inst in enumerate(instructions):
        if FusionRules.is_fusable(inst):
            priorities[inst.name] = float(actions_det[i].item())
        else:
            priorities[inst.name] = float("-inf")

    go_clusters = simulate_fusion(instructions=instructions, edges=edges, priorities=priorities)
    go_cluster_sets = [c.members for c in go_clusters]
    go_runtime = estimate_total_runtime(go_cluster_sets, instruction_map, gpu_specs, computation_map_env)
    print(f"  GO policy:      {go_runtime:.6e} s  ({len(go_clusters)} clusters)")

    # Stochastic samples
    stoch_runtimes = []
    for _ in range(10):
        with torch.no_grad():
            probs, _ = policy(obs_x, obs_ei, obs_oc)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
        prios = {}
        for i, inst in enumerate(instructions):
            if FusionRules.is_fusable(inst):
                prios[inst.name] = float(actions[i].item())
            else:
                prios[inst.name] = float("-inf")
        clusters = simulate_fusion(instructions=instructions, edges=edges, priorities=prios)
        rt = estimate_total_runtime([c.members for c in clusters], instruction_map, gpu_specs, computation_map_env)
        stoch_runtimes.append(rt)
    best_stoch = min(stoch_runtimes)
    print(f"  GO stochastic:  mean={np.mean(stoch_runtimes):.6e}, best={best_stoch:.6e}")

    # Baselines
    baseline_runtimes = run_all_baselines(
        instructions=instructions, edges=edges, gpu_specs=gpu_specs,
        num_priorities=20, max_cluster_size=64, sa_iterations=SA_ITERATIONS,
        computation_map=computation_map_env, verbose=True,
    )

    # Results table
    go_cluster_members = [list(c.members) for c in go_clusters]
    cluster_stats = FusionMetrics.compute_cluster_stats(go_cluster_members)
    results_table = FusionMetrics.format_results_table(
        go_runtime=go_runtime, baselines=baseline_runtimes, go_cluster_stats=cluster_stats,
    )
    print(results_table)

# =====================================================================
# Reward history summary
# =====================================================================
print("\nReward History (sampled):")
indices = list(range(0, len(reward_history), max(1, len(reward_history) // 20)))
if len(reward_history) - 1 not in indices:
    indices.append(len(reward_history) - 1)
for i in indices:
    print(f"  Update {i + 1:4d}: {reward_history[i]:+.4f}")
