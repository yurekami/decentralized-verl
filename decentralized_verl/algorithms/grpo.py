"""GRPO (Group Relative Policy Optimization) implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    kl_coef: float = 0.1,
    normalize_rewards: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute GRPO loss.

    GRPO uses relative rewards within a group of samples for the same prompt,
    which provides better signal than absolute rewards.

    Args:
        log_probs: Policy log probabilities [batch * group_size]
        ref_log_probs: Reference log probabilities [batch * group_size]
        rewards: Rewards for each sample [batch * group_size]
        group_size: Number of samples per prompt
        kl_coef: KL penalty coefficient
        normalize_rewards: Whether to normalize rewards within groups

    Returns:
        Dictionary of losses
    """
    device = log_probs.device
    total_samples = log_probs.shape[0]
    num_groups = total_samples // group_size

    # Reshape to groups
    log_probs = log_probs.view(num_groups, group_size)
    ref_log_probs = ref_log_probs.view(num_groups, group_size)
    rewards = rewards.view(num_groups, group_size)

    # Normalize rewards within each group (z-score)
    if normalize_rewards:
        reward_mean = rewards.mean(dim=1, keepdim=True)
        reward_std = rewards.std(dim=1, keepdim=True) + 1e-8
        normalized_rewards = (rewards - reward_mean) / reward_std
    else:
        normalized_rewards = rewards

    # Compute relative advantages
    # For each sample, advantage is its reward minus mean of other samples
    advantages = normalized_rewards - normalized_rewards.mean(dim=1, keepdim=True)

    # Policy loss: maximize log_prob * advantage
    policy_loss = -(log_probs * advantages.detach()).mean()

    # KL penalty
    kl = log_probs - ref_log_probs
    kl_loss = kl_coef * kl.mean()

    # Total loss
    total_loss = policy_loss + kl_loss

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "kl": kl.mean(),
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(),
    }


def rloo_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute RLOO (REINFORCE Leave-One-Out) loss.

    Uses leave-one-out baseline within each group for variance reduction.

    Args:
        log_probs: Policy log probabilities [batch * group_size]
        rewards: Rewards [batch * group_size]
        group_size: Number of samples per prompt

    Returns:
        Dictionary of losses
    """
    total_samples = log_probs.shape[0]
    num_groups = total_samples // group_size

    # Reshape to groups
    log_probs = log_probs.view(num_groups, group_size)
    rewards = rewards.view(num_groups, group_size)

    # Compute leave-one-out baselines
    # For sample i, baseline is mean of all other samples in the group
    reward_sum = rewards.sum(dim=1, keepdim=True)
    loo_baselines = (reward_sum - rewards) / (group_size - 1)

    # Advantages using LOO baseline
    advantages = rewards - loo_baselines

    # Policy loss
    policy_loss = -(log_probs * advantages.detach()).mean()

    return {
        "loss": policy_loss,
        "policy_loss": policy_loss,
        "reward_mean": rewards.mean(),
        "advantage_mean": advantages.mean(),
    }


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 4
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0
    n_epochs: int = 1
    normalize_rewards: bool = True
    use_rloo: bool = False


class GRPOTrainer:
    """
    GRPO trainer for RLHF.

    Implements Group Relative Policy Optimization which:
    - Generates multiple samples per prompt
    - Uses relative rewards within groups
    - Provides better learning signal than absolute rewards
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        config: Optional[GRPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize GRPO trainer.

        Args:
            policy_model: Policy network
            reference_model: Reference model for KL penalty
            config: GRPO configuration
            optimizer: Optimizer
        """
        self.policy = policy_model
        self.reference = reference_model
        self.config = config or GRPOConfig()

        self.optimizer = optimizer or torch.optim.AdamW(
            policy_model.parameters(), lr=1e-6
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.

        Args:
            batch: Training batch with keys:
                - input_ids: Token IDs [batch * group_size, seq]
                - attention_mask: Attention mask
                - rewards: Rewards [batch * group_size]
                - ref_log_probs: Reference log probs [batch * group_size]

        Returns:
            Dictionary of metrics
        """
        self.policy.train()

        total_metrics = {
            "policy_loss": 0.0,
            "kl": 0.0,
            "reward_mean": 0.0,
        }

        for epoch in range(self.config.n_epochs):
            # Forward pass
            outputs = self.policy(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[:, :-1].gather(
                dim=-1,
                index=batch["input_ids"][:, 1:].unsqueeze(-1),
            ).squeeze(-1)

            # Mask and sum
            mask = batch.get("response_mask", torch.ones_like(token_log_probs))
            new_log_probs = (token_log_probs * mask).sum(dim=-1)

            # Get reference log probs
            if self.reference is not None and "ref_log_probs" not in batch:
                with torch.no_grad():
                    ref_outputs = self.reference(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    ref_logits = ref_outputs.logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs[:, :-1].gather(
                        dim=-1,
                        index=batch["input_ids"][:, 1:].unsqueeze(-1),
                    ).squeeze(-1)
                    ref_log_probs = (ref_token_log_probs * mask).sum(dim=-1)
            else:
                ref_log_probs = batch.get("ref_log_probs", torch.zeros_like(new_log_probs))

            # Compute loss
            if self.config.use_rloo:
                losses = rloo_loss(
                    log_probs=new_log_probs,
                    rewards=batch["rewards"],
                    group_size=self.config.group_size,
                )
            else:
                losses = grpo_loss(
                    log_probs=new_log_probs,
                    ref_log_probs=ref_log_probs,
                    rewards=batch["rewards"],
                    group_size=self.config.group_size,
                    kl_coef=self.config.kl_coef,
                    normalize_rewards=self.config.normalize_rewards,
                )

            # Backward pass
            self.optimizer.zero_grad()
            losses["loss"].backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()

            # Update metrics
            total_metrics["policy_loss"] += losses["policy_loss"].item() / self.config.n_epochs
            total_metrics["kl"] += losses.get("kl", torch.tensor(0.0)).item() / self.config.n_epochs
            total_metrics["reward_mean"] += losses["reward_mean"].item() / self.config.n_epochs

        return total_metrics

    def generate_group_samples(
        self,
        prompts: List[str],
        generation_fn: callable,
        **kwargs,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate multiple samples per prompt for GRPO.

        Args:
            prompts: Input prompts
            generation_fn: Function to generate responses
            **kwargs: Generation parameters

        Returns:
            Tuple of (all_responses, all_log_probs)
        """
        all_responses = []
        all_log_probs = []

        for prompt in prompts:
            for _ in range(self.config.group_size):
                # Generate one sample
                response, log_probs = generation_fn(
                    [prompt],
                    do_sample=True,
                    **kwargs,
                )
                all_responses.extend(response)
                all_log_probs.extend(log_probs)

        return all_responses, all_log_probs
