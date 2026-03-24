"""Quick evaluation of trained GO policy vs baselines."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

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
from src.utils.gpu_specs import get_gpu_specs

GPU_TARGET = 'a100'
SA_ITERATIONS = 500  # reduced for speed

device = torch.device('cpu')
gpu_specs = get_gpu_specs(GPU_TARGET)

# ===== Parse HLO =====
hlo_path = '../hlos/mlp_gpt/0060before_fusion.hlo'
print(f'Parsing {hlo_path}...')
module = parse_hlo_file(hlo_path)
graph_data = build_graph(module)
graph_data = encode_features(graph_data)

instructions = graph_data.instructions
node_names = graph_data.node_names
edge_index = graph_data.edge_index
edges = []
for i in range(edge_index.shape[1]):
    edges.append((node_names[edge_index[0,i].item()], node_names[edge_index[1,i].item()]))

computation_map = module.computations
instruction_map = {inst.name: inst for inst in instructions}

env = FusionEnv(graph_data, instructions, edges, gpu_specs, computation_map=computation_map)
print(f'Graph: {graph_data.num_nodes} nodes, {edge_index.shape[1]} edges, {env.num_fusable} fusable')
print(f'Baseline runtime (no fusion): {env.baseline_runtime:.4e} s')

# ===== Load trained model =====
print('\nLoading trained model...')
policy = GOFusionPolicy(
    num_node_features=19, hidden_dim=128, num_priorities=20,
    num_opcodes=132, opcode_embed_dim=32,
    num_gnn_layers=2, num_transformer_layers=3, num_heads=8,
    segment_size=256, num_iterations=3, d_ff=512, dropout=0.1,
).to(device)
value_net = ValueNetwork(hidden_dim=128).to(device)

ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
policy.load_state_dict(ckpt['policy_state_dict'])
value_net.load_state_dict(ckpt['value_net_state_dict'])
print(f'Loaded best_model.pt (update={ckpt.get("update","?")}, reward={ckpt.get("mean_reward","?"):.4f})')

total_params = sum(p.numel() for p in policy.parameters()) + sum(p.numel() for p in value_net.parameters())
print(f'Model parameters: {total_params:,}')

# ===== Run GO policy (deterministic) =====
print('\n' + '='*70)
print('  EVALUATION: GO Policy vs Baselines')
print('='*70)
print(f'\nTarget GPU: {GPU_TARGET.upper()}')
print(f'HLO: {hlo_path}')

policy.eval()
with torch.no_grad():
    obs_x = graph_data.x.to(device)
    obs_ei = graph_data.edge_index.to(device)
    obs_oc = graph_data.opcode_ids.to(device)
    probs, _ = policy(obs_x, obs_ei, obs_oc)
    actions_det = probs.argmax(dim=-1)

priorities = {}
for i, inst in enumerate(instructions):
    if FusionRules.is_fusable(inst):
        priorities[inst.name] = float(actions_det[i].item())
    else:
        priorities[inst.name] = float('-inf')

go_clusters = simulate_fusion(instructions=instructions, edges=edges, priorities=priorities)
go_cluster_sets = [c.members for c in go_clusters]
go_runtime = estimate_total_runtime(go_cluster_sets, instruction_map, gpu_specs, computation_map)
print(f'\nGO policy runtime: {go_runtime:.6e} s  ({len(go_clusters)} clusters)')

# Also run stochastic (5 samples)
print('Running 5 stochastic samples...')
stoch_runtimes = []
for _ in range(5):
    with torch.no_grad():
        probs, _ = policy(obs_x, obs_ei, obs_oc)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
    prios = {}
    for i, inst in enumerate(instructions):
        if FusionRules.is_fusable(inst):
            prios[inst.name] = float(actions[i].item())
        else:
            prios[inst.name] = float('-inf')
    clusters = simulate_fusion(instructions=instructions, edges=edges, priorities=prios)
    rt = estimate_total_runtime([c.members for c in clusters], instruction_map, gpu_specs, computation_map)
    stoch_runtimes.append(rt)

print(f'GO stochastic: mean={np.mean(stoch_runtimes):.6e}, '
      f'best={min(stoch_runtimes):.6e}, worst={max(stoch_runtimes):.6e}')

# ===== Run baselines =====
print('\nRunning baselines...')
t0 = time.time()
baseline_runtimes = run_all_baselines(
    instructions=instructions, edges=edges, gpu_specs=gpu_specs,
    num_priorities=20, max_cluster_size=64, sa_iterations=SA_ITERATIONS,
    computation_map=computation_map, verbose=True,
)
print(f'Baselines computed in {time.time()-t0:.1f}s')

# ===== Results table =====
go_cluster_members = [list(c.members) for c in go_clusters]
cluster_stats = FusionMetrics.compute_cluster_stats(go_cluster_members)
results_table = FusionMetrics.format_results_table(
    go_runtime=go_runtime, baselines=baseline_runtimes, go_cluster_stats=cluster_stats,
)
print(results_table)
