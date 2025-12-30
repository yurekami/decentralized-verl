"""PPO (Proximal Policy Optimization) implementation for decentralized RLHF."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Reward tensor [batch]
        values: Value estimates [batch]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize: Whether to normalize advantages

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size = rewards.shape[0]
    device = rewards.device

    # For RLHF with single-step rewards (no temporal structure)
    # GAE simplifies to: A = R - V
    advantages = rewards - values

    # Returns = rewards (since we have terminal rewards)
    returns = rewards.clone()

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_gae_sequential(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE for sequential episodes.

    Args:
        rewards: Reward tensor [batch, seq_len]
        values: Value estimates [batch, seq_len + 1]
        dones: Done flags [batch, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=device)

    for t in reversed(range(seq_len)):
        next_values = values[:, t + 1] * (1 - dones[:, t])
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values[:, :-1]

    return advantages, returns


def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    clip_range_vf: Optional[float] = None,
    values: Optional[torch.Tensor] = None,
    old_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01,
    entropy: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO loss.

    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        clip_range: Policy clip range
        clip_range_vf: Value function clip range
        values: Current value estimates
        old_values: Old value estimates
        returns: Return targets
        vf_coef: Value function coefficient
        entropy_coef: Entropy bonus coefficient
        entropy: Entropy of policy distribution

    Returns:
        Dictionary of losses
    """
    # Policy ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped policy loss
    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Value loss
    value_loss = torch.tensor(0.0, device=log_probs.device)
    if values is not None and returns is not None:
        if clip_range_vf is not None and old_values is not None:
            # Clipped value loss
            values_clipped = old_values + torch.clamp(
                values - old_values, -clip_range_vf, clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, returns, reduction="none")
            value_loss_2 = F.mse_loss(values_clipped, returns, reduction="none")
            value_loss = torch.max(value_loss_1, value_loss_2).mean()
        else:
            value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    entropy_loss = torch.tensor(0.0, device=log_probs.device)
    if entropy is not None:
        entropy_loss = -entropy.mean()

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

    # Clip fraction for monitoring
    clip_fraction = torch.mean(
        (torch.abs(ratio - 1.0) > clip_range).float()
    )

    # Approximate KL divergence
    approx_kl = 0.5 * torch.mean((log_probs - old_log_probs) ** 2)

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01
    n_epochs: int = 4
    batch_size: int = 64
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


class PPOTrainer:
    """
    PPO trainer for RLHF.

    Implements the PPO algorithm with support for:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus
    - KL early stopping
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: Optional[nn.Module] = None,
        config: Optional[PPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        value_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: Policy network
            value_model: Value network (can be same as policy)
            config: PPO configuration
            optimizer: Policy optimizer
            value_optimizer: Value optimizer
        """
        self.policy = policy_model
        self.value = value_model or policy_model
        self.config = config or PPOConfig()

        self.optimizer = optimizer or torch.optim.AdamW(
            policy_model.parameters(), lr=1e-5
        )
        self.value_optimizer = value_optimizer

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.

        Args:
            batch: Training batch with keys:
                - input_ids: Token IDs [batch, seq]
                - attention_mask: Attention mask [batch, seq]
                - old_log_probs: Old log probs [batch]
                - advantages: Advantage estimates [batch]
                - returns: Return targets [batch]
                - old_values: Old value estimates [batch] (optional)

        Returns:
            Dictionary of metrics
        """
        self.policy.train()

        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
        }

        # Training loop
        for epoch in range(self.config.n_epochs):
            # Forward pass
            outputs = self.policy(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits

            # Compute new log probs
            log_probs = F.log_softmax(logits, dim=-1)
            # Get log probs for actual tokens
            token_log_probs = log_probs[:, :-1].gather(
                dim=-1,
                index=batch["input_ids"][:, 1:].unsqueeze(-1),
            ).squeeze(-1)

            # Sum over sequence
            new_log_probs = (token_log_probs * batch.get("response_mask", torch.ones_like(token_log_probs))).sum(dim=-1)

            # Compute entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Get values if needed
            values = None
            if self.value is not None and "returns" in batch:
                if hasattr(outputs, "value"):
                    values = outputs.value
                else:
                    # Separate forward pass for value
                    value_outputs = self.value(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    if hasattr(value_outputs, "value"):
                        values = value_outputs.value
                    else:
                        # Use last token hidden state
                        values = value_outputs.logits[:, -1, 0]

            # Compute PPO loss
            losses = ppo_loss(
                log_probs=new_log_probs,
                old_log_probs=batch["old_log_probs"],
                advantages=batch["advantages"],
                clip_range=self.config.clip_range,
                clip_range_vf=self.config.clip_range_vf,
                values=values,
                old_values=batch.get("old_values"),
                returns=batch.get("returns"),
                vf_coef=self.config.vf_coef,
                entropy_coef=self.config.entropy_coef,
                entropy=entropy,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["loss"].backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()

            # Update metrics
            for key in total_metrics:
                if key in losses:
                    total_metrics[key] += losses[key].item() / self.config.n_epochs

            total_metrics["entropy"] += entropy.item() / self.config.n_epochs
            total_metrics["kl"] += losses["approx_kl"].item() / self.config.n_epochs

            # KL early stopping
            if self.config.target_kl is not None:
                if losses["approx_kl"].item() > self.config.target_kl * 1.5:
                    logger.info(f"Early stopping at epoch {epoch} due to KL divergence")
                    break

        return total_metrics

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using GAE."""
        return compute_advantages(
            rewards=rewards,
            values=values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize=self.config.normalize_advantages,
        )
