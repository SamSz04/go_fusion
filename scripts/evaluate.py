#!/usr/bin/env python3
"""
CLI entry point for evaluating a trained GO fusion policy against baselines.

Usage:
    python scripts/evaluate.py --checkpoint path/to/model.pt --hlo path/to/file.hlo
    python scripts/evaluate.py --checkpoint path/to/model.pt --hlo-dir path/to/hlos/

Examples:
    # Evaluate on a single HLO file
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --hlo ../hlos/mlp_gpt/0060before_fusion.hlo

    # Evaluate on all HLO files in a directory
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --hlo-dir ../hlos/

    # Evaluate with custom SA iterations and deterministic policy
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --hlo ../hlos/mlp_gpt/0060before_fusion.hlo \\
        --sa-iterations 50000 \\
        --deterministic
"""

import argparse
import glob as globmod
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.hlo_parser.parser import parse_hlo_file
from src.hlo_parser.graph_builder import build_graph
from src.hlo_parser.feature_encoder import encode_features
from src.hlo_parser.hlo_ir import HloInstruction
from src.env.fusion_env import FusionEnv
from src.env.fusion_simulator import simulate_fusion
from src.env.fusion_rules import FusionRules
from src.env.performance_model import estimate_total_runtime
from src.model.policy_network import GOFusionPolicy
from src.model.value_network import ValueNetwork
from src.evaluation.baselines import run_all_baselines
from src.evaluation.metrics import FusionMetrics
from src.utils.gpu_specs import get_gpu_specs, GPUSpecs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GO fusion policy against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--hlo",
        type=str,
        default=None,
        help="Path to a single HLO file to evaluate on",
    )
    parser.add_argument(
        "--hlo-dir",
        type=str,
        default=None,
        help="Path to directory of HLO files to evaluate on",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
        help="Path to config YAML (used for model hyperparameters)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override target GPU (e.g., 'v100', 'a100')",
    )
    parser.add_argument(
        "--sa-iterations",
        type=int,
        default=10000,
        help="Number of simulated annealing iterations (default: 10000)",
    )
    parser.add_argument(
        "--num-eval-runs",
        type=int,
        default=5,
        help="Number of policy evaluation runs to average (default: 5)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax instead of sampling for policy actions",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: str, config: dict, device: torch.device
) -> Tuple[GOFusionPolicy, ValueNetwork]:
    """Load policy and value networks from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: Configuration dictionary.
        device: Torch device.

    Returns:
        Tuple of (GOFusionPolicy, ValueNetwork).
    """
    model_config = config["model"]

    policy = GOFusionPolicy(
        num_node_features=model_config["num_node_features"],
        hidden_dim=model_config["hidden_dim"],
        num_priorities=model_config["num_priorities"],
        num_opcodes=model_config.get("num_opcodes", 132),
        opcode_embed_dim=model_config.get("opcode_embed_dim", 32),
        num_gnn_layers=model_config["num_gnn_layers"],
        num_transformer_layers=model_config["num_transformer_layers"],
        num_heads=model_config["num_attention_heads"],
        segment_size=model_config["segment_size"],
        num_iterations=model_config["num_iterations"],
        d_ff=model_config["ff_dim"],
        dropout=model_config["dropout"],
    ).to(device)

    value_net = ValueNetwork(
        hidden_dim=model_config["hidden_dim"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    value_net.load_state_dict(checkpoint["value_net_state_dict"])

    policy.eval()
    value_net.eval()

    update = checkpoint.get("update", "unknown")
    reward = checkpoint.get("mean_reward", "unknown")
    print(f"Loaded model from update {update} (mean_reward={reward})")

    return policy, value_net


def _extract_edges_as_name_pairs(
    instructions: List[HloInstruction],
    edge_index: torch.Tensor,
) -> List[Tuple[str, str]]:
    """Convert a PyG edge_index tensor to (producer_name, consumer_name) pairs."""
    edges = []
    if edge_index.numel() == 0:
        return edges
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        edges.append((instructions[s].name, instructions[d].name))
    return edges


def run_go_policy(
    policy: GOFusionPolicy,
    graph_data,
    device: torch.device,
    deterministic: bool = False,
) -> Dict[str, float]:
    """Run the GO policy to get fusion priorities as a name->score dict.

    Args:
        policy: Trained GOFusionPolicy.
        graph_data: PyG Data object with encoded features.
        device: Torch device.
        deterministic: If True, use argmax; if False, sample.

    Returns:
        Dictionary mapping instruction name to priority score.
    """
    with torch.no_grad():
        obs_x = graph_data.x.to(device)
        obs_edge_index = graph_data.edge_index.to(device)
        obs_opcode_ids = graph_data.opcode_ids.to(device)

        probs, _ = policy(obs_x, obs_edge_index, obs_opcode_ids)

        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

    actions_np = actions.cpu().numpy()
    instructions = graph_data.instructions

    priorities: Dict[str, float] = {}
    for i, inst in enumerate(instructions):
        if FusionRules.is_fusable(inst):
            priorities[inst.name] = float(actions_np[i])
        else:
            priorities[inst.name] = float("-inf")

    return priorities


def evaluate_single_hlo(
    hlo_path: str,
    policy: GOFusionPolicy,
    config: dict,
    device: torch.device,
    sa_iterations: int = 10000,
    num_eval_runs: int = 5,
    deterministic: bool = False,
) -> dict:
    """Evaluate GO policy on a single HLO file.

    Args:
        hlo_path: Path to the HLO text file.
        policy: Trained GOFusionPolicy.
        config: Configuration dictionary.
        device: Torch device.
        sa_iterations: Number of SA iterations for that baseline.
        num_eval_runs: Number of policy runs to average.
        deterministic: Whether to use deterministic action selection.

    Returns:
        Dictionary with GO runtime, baseline runtimes, and comparison metrics.
    """
    print(f"\nEvaluating: {hlo_path}")

    # Parse HLO and build graph
    hlo_module = parse_hlo_file(hlo_path)
    graph_data = build_graph(hlo_module)
    graph_data = encode_features(graph_data)

    instructions = graph_data.instructions
    edges = _extract_edges_as_name_pairs(instructions, graph_data.edge_index)
    instruction_map = {inst.name: inst for inst in instructions}
    computation_map = hlo_module.computations

    print(
        f"  Graph: {graph_data.num_nodes} nodes, "
        f"{graph_data.edge_index.size(1)} edges"
    )

    # Get environment parameters
    env_config = config["environment"]
    gpu_specs = get_gpu_specs(env_config.get("gpu_target", "v100"))
    num_priorities = env_config.get("num_priorities", 20)
    max_cluster_size = env_config.get("max_cluster_size", 64)

    # Run GO policy (multiple times for stochastic evaluation)
    print(f"  Running GO policy ({num_eval_runs} runs)...")
    go_runtimes = []
    go_clusters_list = []

    for run in range(num_eval_runs):
        priorities = run_go_policy(policy, graph_data, device, deterministic)

        clusters = simulate_fusion(
            instructions=instructions,
            edges=edges,
            priorities=priorities,
            max_cluster_size=max_cluster_size,
        )
        cluster_sets: List[Set[str]] = [c.members for c in clusters]
        runtime = estimate_total_runtime(
            clusters=cluster_sets,
            instruction_map=instruction_map,
            gpu_specs=gpu_specs,
            computation_map=computation_map,
        )
        go_runtimes.append(runtime)
        go_clusters_list.append(clusters)

    go_runtime = float(np.mean(go_runtimes))
    best_run_idx = int(np.argmin(go_runtimes))
    go_clusters = go_clusters_list[best_run_idx]

    print(
        f"  GO runtime: {go_runtime:.6e} s "
        f"(best: {min(go_runtimes):.6e}, worst: {max(go_runtimes):.6e})"
    )

    # Run baselines
    baseline_runtimes = run_all_baselines(
        instructions=instructions,
        edges=edges,
        gpu_specs=gpu_specs,
        num_priorities=num_priorities,
        max_cluster_size=max_cluster_size,
        sa_iterations=sa_iterations,
        computation_map=computation_map,
        verbose=True,
    )

    # Compute cluster statistics for GO
    go_cluster_members = [list(c.members) for c in go_clusters]
    cluster_stats = FusionMetrics.compute_cluster_stats(go_cluster_members)

    # Format and print results
    results_table = FusionMetrics.format_results_table(
        go_runtime=go_runtime,
        baselines=baseline_runtimes,
        go_cluster_stats=cluster_stats,
    )
    print(results_table)

    return {
        "hlo_file": hlo_path,
        "go_runtime": go_runtime,
        "go_runtimes_all": go_runtimes,
        "baselines": baseline_runtimes,
        "cluster_stats": cluster_stats,
        "comparisons": FusionMetrics.compare_all(go_runtime, baseline_runtimes),
    }


def main():
    args = parse_args()

    # Validate arguments
    if args.hlo is None and args.hlo_dir is None:
        print("Error: Must provide either --hlo or --hlo-dir")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    if args.gpu:
        config["environment"]["gpu_target"] = args.gpu

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    policy, value_net = load_model(args.checkpoint, config, device)

    # Collect HLO files
    hlo_files = []
    if args.hlo:
        if not os.path.exists(args.hlo):
            print(f"Error: HLO file not found: {args.hlo}")
            sys.exit(1)
        hlo_files.append(args.hlo)
    elif args.hlo_dir:
        patterns = [
            os.path.join(args.hlo_dir, "**", "*.hlo"),
            os.path.join(args.hlo_dir, "*.hlo"),
        ]
        for pat in patterns:
            hlo_files.extend(globmod.glob(pat, recursive=True))
        hlo_files = sorted(set(hlo_files))

    if not hlo_files:
        print("Error: No HLO files found.")
        sys.exit(1)

    print(f"\nFound {len(hlo_files)} HLO file(s) for evaluation.")

    # Evaluate each HLO file
    all_results = []
    for hlo_path in hlo_files:
        result = evaluate_single_hlo(
            hlo_path=hlo_path,
            policy=policy,
            config=config,
            device=device,
            sa_iterations=args.sa_iterations,
            num_eval_runs=args.num_eval_runs,
            deterministic=args.deterministic,
        )
        all_results.append(result)

    # Print aggregate summary if multiple files
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("  Aggregate Results Across All HLO Files")
        print("=" * 80)

        baseline_names = list(all_results[0]["baselines"].keys())
        print(
            f"\n  {'Baseline':<25s} {'Avg Speedup':<15s} {'Avg Improvement':<15s}"
        )
        print("  " + "-" * 55)

        for bname in baseline_names:
            speedups = [
                r["comparisons"][bname]["speedup"]
                for r in all_results
                if bname in r["comparisons"]
            ]
            improvements = [
                r["comparisons"][bname]["improvement_pct"]
                for r in all_results
                if bname in r["comparisons"]
            ]
            if speedups:
                display = bname.replace("_", " ").title()
                avg_speedup = np.mean(speedups)
                avg_improvement = np.mean(improvements)
                sign = "+" if avg_improvement >= 0 else ""
                print(
                    f"  {display:<25s} {avg_speedup:<15.3f} "
                    f"{sign}{avg_improvement:.1f}%"
                )

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
