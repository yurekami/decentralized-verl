"""Tests for RLHF algorithms."""

import pytest
import torch

from decentralized_verl.algorithms.ppo import (
    compute_advantages,
    compute_gae_sequential,
    ppo_loss,
    PPOConfig,
    PPOTrainer,
)
from decentralized_verl.algorithms.grpo import (
    grpo_loss,
    rloo_loss,
    GRPOConfig,
    GRPOTrainer,
)


class TestComputeAdvantages:
    """Tests for advantage computation."""

    def test_basic_advantages(self):
        """Test basic advantage calculation."""
        rewards = torch.tensor([1.0, 0.5, 0.8])
        values = torch.tensor([0.9, 0.6, 0.7])

        advantages, returns = compute_advantages(rewards, values)

        # Advantages = rewards - values
        expected_adv = torch.tensor([0.1, -0.1, 0.1])
        assert torch.allclose(advantages, expected_adv, atol=0.01)

    def test_normalized_advantages(self):
        """Test normalized advantages."""
        rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])

        advantages, returns = compute_advantages(rewards, values, normalize=True)

        # Normalized should have mean ~0 and std ~1
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.1


class TestComputeGAESequential:
    """Tests for sequential GAE computation."""

    def test_gae_sequential(self):
        """Test GAE with sequential rewards."""
        rewards = torch.tensor([[0.1, 0.2, 1.0]])  # [batch=1, seq=3]
        values = torch.tensor([[0.5, 0.6, 0.7, 0.0]])  # [batch=1, seq=4] (includes V(s_T+1))
        dones = torch.tensor([[0.0, 0.0, 1.0]])  # Episode ends at step 3

        advantages, returns = compute_gae_sequential(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95
        )

        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_ppo_loss_basic(self):
        """Test basic PPO loss calculation."""
        log_probs = torch.tensor([-1.0, -0.5, -0.8])
        old_log_probs = torch.tensor([-1.1, -0.6, -0.9])
        advantages = torch.tensor([0.5, -0.3, 0.2])

        losses = ppo_loss(log_probs, old_log_probs, advantages)

        assert "loss" in losses
        assert "policy_loss" in losses
        assert "clip_fraction" in losses
        assert "approx_kl" in losses

    def test_ppo_loss_clipping(self):
        """Test that clipping is applied."""
        # Large ratio should be clipped
        log_probs = torch.tensor([0.0])
        old_log_probs = torch.tensor([-2.0])  # ratio = exp(2) = 7.4
        advantages = torch.tensor([1.0])

        losses = ppo_loss(log_probs, old_log_probs, advantages, clip_range=0.2)

        # Clip fraction should be non-zero
        assert losses["clip_fraction"].item() > 0

    def test_ppo_loss_with_value(self):
        """Test PPO loss with value function."""
        log_probs = torch.tensor([-1.0, -0.5])
        old_log_probs = torch.tensor([-1.1, -0.6])
        advantages = torch.tensor([0.5, -0.3])
        values = torch.tensor([0.5, 0.6])
        returns = torch.tensor([0.6, 0.5])

        losses = ppo_loss(
            log_probs, old_log_probs, advantages,
            values=values, returns=returns, vf_coef=0.5
        )

        assert losses["value_loss"].item() > 0

    def test_ppo_loss_with_entropy(self):
        """Test PPO loss with entropy bonus."""
        log_probs = torch.tensor([-1.0, -0.5])
        old_log_probs = torch.tensor([-1.1, -0.6])
        advantages = torch.tensor([0.5, -0.3])
        entropy = torch.tensor([0.5, 0.6])

        losses = ppo_loss(
            log_probs, old_log_probs, advantages,
            entropy=entropy, entropy_coef=0.01
        )

        assert losses["entropy_loss"].item() != 0


class TestPPOConfig:
    """Tests for PPO configuration."""

    def test_default_config(self):
        """Test default PPO config values."""
        config = PPOConfig()
        assert config.clip_range == 0.2
        assert config.n_epochs == 4
        assert config.vf_coef == 0.5

    def test_custom_config(self):
        """Test custom PPO config."""
        config = PPOConfig(
            clip_range=0.1,
            n_epochs=8,
            target_kl=0.02,
        )
        assert config.clip_range == 0.1
        assert config.n_epochs == 8
        assert config.target_kl == 0.02


class TestGRPOLoss:
    """Tests for GRPO loss computation."""

    def test_grpo_loss_basic(self):
        """Test basic GRPO loss calculation."""
        # 2 prompts, 4 samples each = 8 total
        log_probs = torch.randn(8)
        ref_log_probs = torch.randn(8)
        rewards = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.9, 0.7, 0.4, 0.2])

        losses = grpo_loss(
            log_probs, ref_log_probs, rewards,
            group_size=4, kl_coef=0.1
        )

        assert "loss" in losses
        assert "policy_loss" in losses
        assert "kl_loss" in losses
        assert "kl" in losses

    def test_grpo_normalization(self):
        """Test reward normalization within groups."""
        log_probs = torch.zeros(4)
        ref_log_probs = torch.zeros(4)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        losses = grpo_loss(
            log_probs, ref_log_probs, rewards,
            group_size=4, normalize_rewards=True
        )

        # Loss should be computed with normalized rewards
        assert isinstance(losses["loss"], torch.Tensor)


class TestRLOOLoss:
    """Tests for RLOO loss computation."""

    def test_rloo_loss_basic(self):
        """Test basic RLOO loss calculation."""
        log_probs = torch.randn(8)
        rewards = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.9, 0.7, 0.4, 0.2])

        losses = rloo_loss(log_probs, rewards, group_size=4)

        assert "loss" in losses
        assert "policy_loss" in losses
        assert "advantage_mean" in losses

    def test_rloo_leave_one_out(self):
        """Test leave-one-out baseline computation."""
        log_probs = torch.zeros(4)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        losses = rloo_loss(log_probs, rewards, group_size=4)

        # Leave-one-out baselines should give balanced advantages
        # For reward[0]=1, baseline = (2+3+4)/3 = 3, advantage = 1-3 = -2
        # This is just checking the function runs
        assert isinstance(losses["loss"], torch.Tensor)


class TestGRPOConfig:
    """Tests for GRPO configuration."""

    def test_default_config(self):
        """Test default GRPO config."""
        config = GRPOConfig()
        assert config.group_size == 4
        assert config.kl_coef == 0.1

    def test_rloo_mode(self):
        """Test RLOO mode configuration."""
        config = GRPOConfig(use_rloo=True)
        assert config.use_rloo is True
