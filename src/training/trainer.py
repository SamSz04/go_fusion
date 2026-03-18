"""
Training loop for the GO fusion policy.

Handles:
- Loading HLO graphs and creating environments
- Collecting rollouts using the current policy
- Running PPO updates
- TensorBoard logging
- Model checkpointing
- Both single-graph and multi-graph training
"""

import os
import time
import glob as globmod
import random
from typing import Dict, List, Optional, Any, Tuple

import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.hlo_parser.parser import parse_hlo_file
from src.hlo_parser.graph_builder import build_graph
from src.hlo_parser.feature_encoder import encode_features
from src.env.fusion_env import FusionEnv
from src.model.policy_network import GOFusionPolicy
from src.model.value_network import ValueNetwork
from src.training.ppo import PPO
from src.utils.gpu_specs import get_gpu_specs


def _extract_edges_as_name_pairs(
    instructions, edge_index: torch.Tensor,
) -> List[Tuple[str, str]]:
    """Convert a PyG edge_index tensor to a list of (producer_name, consumer_name) pairs.

    Args:
        instructions: List of HloInstruction (ordered by node index).
        edge_index: [2, E] tensor in COO format.

    Returns:
        List of (producer_name, consumer_name) string tuples.
    """
    edges = []
    if edge_index.numel() == 0:
        return edges
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        edges.append((instructions[s].name, instructions[d].name))
    return edges


class Trainer:
    """Training loop for GO fusion policy using PPO.

    Supports both single-graph training (one HLO file) and multi-graph training
    (directory of HLO files, round-robin or random selection).

    Args:
        config: Dictionary of configuration parameters loaded from YAML.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract config sections
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.env_config = config["environment"]
        self.paths_config = config["paths"]

        # Build environments from HLO files
        self.envs = self._load_environments()
        if not self.envs:
            raise RuntimeError(
                f"No HLO files found in {self.paths_config['hlo_dir']}. "
                "Please provide at least one .hlo file."
            )

        # Build model
        self.policy = GOFusionPolicy(
            num_node_features=self.model_config["num_node_features"],
            hidden_dim=self.model_config["hidden_dim"],
            num_priorities=self.model_config["num_priorities"],
            num_opcodes=self.model_config.get("num_opcodes", 132),
            opcode_embed_dim=self.model_config.get("opcode_embed_dim", 32),
            num_gnn_layers=self.model_config["num_gnn_layers"],
            num_transformer_layers=self.model_config["num_transformer_layers"],
            num_heads=self.model_config["num_attention_heads"],
            segment_size=self.model_config["segment_size"],
            num_iterations=self.model_config["num_iterations"],
            d_ff=self.model_config["ff_dim"],
            dropout=self.model_config["dropout"],
        ).to(self.device)

        self.value_net = ValueNetwork(
            hidden_dim=self.model_config["hidden_dim"],
        ).to(self.device)

        # Build PPO
        self.ppo = PPO(
            policy_network=self.policy,
            value_network=self.value_net,
            lr=self.training_config["learning_rate"],
            clip_ratio=self.training_config["clip_ratio"],
            entropy_coeff=self.training_config["entropy_coeff"],
            value_loss_coeff=self.training_config["value_loss_coeff"],
            max_grad_norm=self.training_config["max_grad_norm"],
            num_epochs=self.training_config["num_epochs_per_update"],
        )

        # Logging
        os.makedirs(self.paths_config["log_dir"], exist_ok=True)
        os.makedirs(self.paths_config["checkpoint_dir"], exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.paths_config["log_dir"])

        # Track best reward for model selection
        self.best_reward = float("-inf")

    def _load_environments(self) -> List[FusionEnv]:
        """Load HLO files and create FusionEnv instances.

        Searches hlo_dir for .hlo files. Each file becomes one environment.

        Returns:
            List of FusionEnv instances.
        """
        hlo_dir = self.paths_config["hlo_dir"]
        gpu_specs = get_gpu_specs(self.env_config["gpu_target"])

        # Find all .hlo files
        hlo_patterns = [
            os.path.join(hlo_dir, "**", "*.hlo"),
            os.path.join(hlo_dir, "*.hlo"),
        ]
        hlo_files = []
        for pattern in hlo_patterns:
            hlo_files.extend(globmod.glob(pattern, recursive=True))
        hlo_files = sorted(set(hlo_files))

        envs = []

        for hlo_file in hlo_files:
            try:
                # Parse HLO file into module
                hlo_module = parse_hlo_file(hlo_file)

                # Build PyG graph from ENTRY computation
                graph_data = build_graph(hlo_module)

                # Encode 49-dim node features
                graph_data = encode_features(graph_data)

                # Extract instructions and name-pair edges for the environment
                instructions = graph_data.instructions
                edges = _extract_edges_as_name_pairs(
                    instructions, graph_data.edge_index
                )

                # Build computation map for FLOP estimation in sub-computations
                computation_map = hlo_module.computations

                env = FusionEnv(
                    hlo_graph_data=graph_data,
                    instructions=instructions,
                    edges=edges,
                    gpu_specs=gpu_specs,
                    num_priorities=self.env_config["num_priorities"],
                    computation_map=computation_map,
                )
                envs.append(env)
                print(
                    f"Loaded environment from: {hlo_file} "
                    f"({graph_data.num_nodes} nodes, "
                    f"{graph_data.edge_index.size(1)} edges)"
                )
            except Exception as e:
                print(f"Warning: Failed to load {hlo_file}: {e}")

        return envs

    def collect_rollout(self, env: FusionEnv) -> Dict[str, Any]:
        """Collect one episode using the current policy.

        Steps:
        1. Reset environment to get initial observation (PyG Data)
        2. Run policy network to get action probabilities
        3. Sample actions from the distribution
        4. Step the environment to get reward
        5. Get value estimate from value network

        Args:
            env: The fusion environment to collect from.

        Returns:
            Dictionary containing observation, actions, log probs, reward, and value.
        """
        self.policy.eval()
        self.value_net.eval()

        with torch.no_grad():
            # Reset environment -> returns PyG Data object
            graph_data = env.reset()
            obs_x = graph_data.x.to(self.device)
            obs_edge_index = graph_data.edge_index.to(self.device)
            obs_opcode_ids = graph_data.opcode_ids.to(self.device)

            # Forward pass through policy
            probs, embeddings = self.policy(obs_x, obs_edge_index, obs_opcode_ids)

            # Sample actions from categorical distribution
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()  # [N] integer priorities
            log_probs = dist.log_prob(actions)  # [N]

            # Get value estimate
            value = self.value_net(embeddings).item()

            # Step environment with sampled priorities
            # FusionEnv.step() accepts a tensor of length num_nodes
            _, reward, done, info = env.step(actions.cpu())

        self.policy.train()
        self.value_net.train()

        return {
            "obs_x": obs_x,
            "obs_edge_index": obs_edge_index,
            "obs_opcode_ids": obs_opcode_ids,
            "actions": actions,
            "old_log_probs": log_probs,
            "reward": reward,
            "value": value,
            "info": info,
        }

    def train(self) -> None:
        """Main training loop.

        For each update:
        1. Collect rollouts_per_update episodes (round-robin across environments)
        2. Run PPO update over collected rollouts
        3. Log metrics to TensorBoard
        4. Save checkpoint every checkpoint_interval updates
        """
        num_updates = self.training_config["num_updates"]
        rollouts_per_update = self.training_config["rollouts_per_update"]
        checkpoint_interval = self.training_config["checkpoint_interval"]
        log_interval = self.training_config["log_interval"]

        print(f"\nStarting training for {num_updates} updates")
        print(f"  Rollouts per update: {rollouts_per_update}")
        print(f"  PPO epochs per update: {self.training_config['num_epochs_per_update']}")
        print(f"  Number of environments: {len(self.envs)}")
        print(f"  Device: {self.device}")
        print()

        global_step = 0
        start_time = time.time()

        for update in tqdm(range(1, num_updates + 1), desc="Training"):
            update_start = time.time()

            # Collect rollouts (round-robin across environments)
            rollouts = []
            episode_rewards = []
            episode_infos = []

            for i in range(rollouts_per_update):
                env = self.envs[i % len(self.envs)]
                rollout = self.collect_rollout(env)
                rollouts.append(rollout)
                episode_rewards.append(rollout["reward"])
                episode_infos.append(rollout["info"])

            # PPO update
            metrics = self.ppo.update(rollouts)

            global_step += rollouts_per_update
            update_time = time.time() - update_start

            # Logging
            mean_reward = float(np.mean(episode_rewards))
            max_reward = float(np.max(episode_rewards))
            min_reward = float(np.min(episode_rewards))

            if update % log_interval == 0:
                # TensorBoard logging
                self.writer.add_scalar("reward/mean", mean_reward, global_step)
                self.writer.add_scalar("reward/max", max_reward, global_step)
                self.writer.add_scalar("reward/min", min_reward, global_step)
                self.writer.add_scalar("loss/policy", metrics["policy_loss"], global_step)
                self.writer.add_scalar("loss/value", metrics["value_loss"], global_step)
                self.writer.add_scalar("loss/entropy", metrics["entropy"], global_step)
                self.writer.add_scalar("ppo/ratio", metrics["ratio"], global_step)
                self.writer.add_scalar("ppo/advantage", metrics["advantage"], global_step)
                self.writer.add_scalar("timing/update_seconds", update_time, global_step)
                self.writer.add_scalar(
                    "timing/total_minutes",
                    (time.time() - start_time) / 60,
                    global_step,
                )

                # Log environment-specific metrics if available
                if episode_infos and "runtime" in episode_infos[0]:
                    runtimes = [
                        info["runtime"] for info in episode_infos if "runtime" in info
                    ]
                    self.writer.add_scalar(
                        "env/mean_runtime", np.mean(runtimes), global_step
                    )
                if episode_infos and "num_clusters" in episode_infos[0]:
                    n_clusters = [
                        info["num_clusters"]
                        for info in episode_infos
                        if "num_clusters" in info
                    ]
                    self.writer.add_scalar(
                        "env/mean_num_clusters", np.mean(n_clusters), global_step
                    )

                # Console logging
                tqdm.write(
                    f"Update {update:4d} | "
                    f"Reward: {mean_reward:+.4f} (max: {max_reward:+.4f}) | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"Time: {update_time:.1f}s"
                )

            # Checkpoint saving
            if update % checkpoint_interval == 0:
                self.save_checkpoint(update, mean_reward)

            # Save best model
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_checkpoint(update, mean_reward, is_best=True)

        # Final checkpoint
        self.save_checkpoint(num_updates, mean_reward, is_final=True)
        self.writer.close()

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best mean reward: {self.best_reward:+.4f}")

    def save_checkpoint(
        self,
        update: int,
        mean_reward: float,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint.

        Saves both policy and value network state dicts, the optimizer state,
        and training metadata.

        Args:
            update: Current update number.
            mean_reward: Current mean reward for metadata.
            is_best: If True, save as 'best_model.pt'.
            is_final: If True, save as 'final_model.pt'.
        """
        checkpoint = {
            "update": update,
            "policy_state_dict": self.policy.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "mean_reward": mean_reward,
            "best_reward": self.best_reward,
            "config": self.config,
        }

        checkpoint_dir = self.paths_config["checkpoint_dir"]

        if is_best:
            path = os.path.join(checkpoint_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(checkpoint_dir, "final_model.pt")
        else:
            path = os.path.join(checkpoint_dir, f"checkpoint_{update:06d}.pt")

        torch.save(checkpoint, path)

        if is_best:
            tqdm.write(f"  Saved best model (reward: {mean_reward:+.4f}) -> {path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The update number from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_reward = checkpoint.get("best_reward", float("-inf"))

        update = checkpoint.get("update", 0)
        print(
            f"Loaded checkpoint from update {update} "
            f"(reward: {checkpoint.get('mean_reward', 'N/A')})"
        )

        return update
