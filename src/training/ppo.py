"""
PPO (Proximal Policy Optimization) algorithm for GO fusion policy training.

Implements the clipped surrogate objective with:
- Graph-level policy loss (sum of per-node log probs)
- Value function loss (MSE)
- Entropy bonus for exploration
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from typing import Dict, List, Any, Tuple, Optional


class PPO:
    """Proximal Policy Optimization for graph-level fusion priority assignment.

    The policy network outputs per-node priority distributions. For PPO, we treat
    the entire graph's action as a joint action: the graph-level log probability
    is the sum of per-node log probabilities. This matches the non-autoregressive
    formulation in the GO paper.

    Args:
        policy_network: PolicyNetwork that outputs (probs [N, |F|], embeddings [N, H]).
        value_network: ValueNetwork that maps embeddings -> scalar value estimate.
        lr: Learning rate for Adam optimizer.
        clip_ratio: PPO clipping parameter epsilon.
        entropy_coeff: Coefficient for entropy bonus (encourages exploration).
        value_loss_coeff: Coefficient for value function loss.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        num_epochs: Number of PPO epochs per update (reuse collected data).
    """

    def __init__(
        self,
        policy_network: torch.nn.Module,
        value_network: torch.nn.Module,
        lr: float = 1e-4,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
    ):
        self.policy = policy_network
        self.value_net = value_network
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs

        # Single optimizer for both policy and value networks (shared backbone)
        self.optimizer = Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    def compute_loss(
        self,
        obs_x: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_opcode_ids: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: float,
        old_values: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the PPO loss for a single episode.

        Args:
            obs_x: Continuous node feature matrix [N, F] from the HLO graph.
            obs_edge_index: Edge index tensor [2, E] (COO format).
            obs_opcode_ids: Integer opcode indices [N].
            actions: Sampled priority actions per node [N], integer indices.
            old_log_probs: Log probabilities from the collection policy [N].
            rewards: Scalar reward for this episode (negative sqrt of normalized runtime).
            old_values: Scalar value estimate from the collection policy.

        Returns:
            loss: Scalar loss tensor.
            info: Dictionary of logging metrics.
        """
        device = obs_x.device

        # Forward pass through policy network
        probs, embeddings = self.policy(obs_x, obs_edge_index, obs_opcode_ids)

        # Per-node action distributions
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)  # [N]
        entropy = dist.entropy().mean()  # scalar, averaged over nodes

        # Value estimate from shared embeddings
        new_value = self.value_net(embeddings)  # scalar

        # Simple advantage: reward - value estimate (single-step, no GAE)
        advantage = rewards - old_values
        advantage_tensor = torch.tensor(advantage, device=device, dtype=torch.float32)

        # Graph-level log probability: sum over all node decisions
        log_prob_sum = new_log_probs.sum()
        old_log_prob_sum = old_log_probs.sum()

        # PPO clipped surrogate objective
        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantage_tensor, clipped_ratio * advantage_tensor)

        # Value loss (MSE between predicted value and actual reward)
        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        value_loss = F.mse_loss(new_value.squeeze(), reward_tensor)

        # Total loss: policy + value - entropy bonus
        total_loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            - self.entropy_coeff * entropy
        )

        info = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "ratio": ratio.item(),
            "advantage": advantage,
            "reward": rewards,
            "value_estimate": old_values,
        }

        return total_loss, info

    def update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run multiple epochs of PPO updates over collected rollouts.

        Each rollout is a dictionary containing:
            - 'obs_x': [N, F] continuous node features
            - 'obs_edge_index': [2, E] edge indices
            - 'obs_opcode_ids': [N] integer opcode indices
            - 'actions': [N] sampled priority actions
            - 'old_log_probs': [N] log probs from collection policy
            - 'reward': scalar reward
            - 'value': scalar value estimate

        Args:
            rollouts: List of rollout dictionaries from collect_rollout().

        Returns:
            Dictionary of averaged metrics over all epochs and rollouts.
        """
        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "advantage": 0.0,
            "reward": 0.0,
        }
        num_steps = 0

        for epoch in range(self.num_epochs):
            for rollout in rollouts:
                loss, info = self.compute_loss(
                    obs_x=rollout["obs_x"],
                    obs_edge_index=rollout["obs_edge_index"],
                    obs_opcode_ids=rollout["obs_opcode_ids"],
                    actions=rollout["actions"],
                    old_log_probs=rollout["old_log_probs"],
                    rewards=rollout["reward"],
                    old_values=rollout["value"],
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # Accumulate metrics
                for key in total_metrics:
                    if key in info:
                        total_metrics[key] += info[key]
                num_steps += 1

        # Average metrics
        if num_steps > 0:
            for key in total_metrics:
                total_metrics[key] /= num_steps

        total_metrics["num_epochs"] = self.num_epochs
        total_metrics["num_rollouts"] = len(rollouts)

        return total_metrics
