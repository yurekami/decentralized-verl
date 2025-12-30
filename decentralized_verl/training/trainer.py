"""Main decentralized RLHF trainer."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from decentralized_verl.core.config import DecentralizedConfig, Algorithm
from decentralized_verl.core.node import DecentralizedNode
from decentralized_verl.training.coordinator import DecentralizedCoordinator, EpochMetrics
from decentralized_verl.training.experience import (
    Experience,
    ExperienceBuffer,
    BatchIterator,
    collate_experiences,
)
from decentralized_verl.training.gradient_sync import GradientSynchronizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current state of training."""
    global_step: int = 0
    epoch: int = 0
    policy_version: int = 0
    best_reward: float = float("-inf")
    total_samples: int = 0
    total_time: float = 0.0


class DecentralizedTrainer:
    """
    Main trainer for decentralized RLHF.

    Orchestrates the full training loop across a P2P network,
    handling rollout generation, reward computation, advantage
    estimation, and policy updates.

    Supports:
    - PPO, GRPO, REINFORCE++ algorithms
    - Distributed gradient synchronization
    - Fault-tolerant training
    - Checkpoint management
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        reference_model: Optional[nn.Module] = None,
        tokenizer: Any = None,
    ):
        """
        Initialize the decentralized trainer.

        Args:
            config: Training configuration
            actor_model: Policy model
            critic_model: Value model (optional for some algorithms)
            reward_model: Reward model
            reference_model: Reference policy for KL penalty
            tokenizer: Tokenizer for encoding
        """
        self.config = config
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer

        # Training state
        self.state = TrainingState()

        # Node and coordinator (initialized in setup)
        self.node: Optional[DecentralizedNode] = None
        self.coordinator: Optional[DecentralizedCoordinator] = None

        # Optimizers
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

        # Schedulers
        self.actor_scheduler = None
        self.critic_scheduler = None

        # Gradient synchronizer
        self.gradient_sync: Optional[GradientSynchronizer] = None

        # Experience buffer
        self.buffer = ExperienceBuffer(
            max_size=config.algorithm.buffer_size,
            min_size=config.algorithm.min_buffer_size,
            on_policy=True,
        )

        # Metrics tracking
        self._metrics_history: List[EpochMetrics] = []
        self._callbacks: List[Callable] = []

        # Control
        self._running = False

    async def setup(self) -> None:
        """Setup trainer components."""
        logger.info("Setting up decentralized trainer...")

        # Create node
        self.node = DecentralizedNode(config=self.config)
        await self.node.start()

        # Create coordinator
        self.coordinator = DecentralizedCoordinator(
            config=self.config,
            node_id=self.node.node_id,
            dht_manager=self.node.dht_manager,
        )
        await self.coordinator.start()

        # Setup optimizers
        self._setup_optimizers()

        # Setup gradient synchronization
        self.gradient_sync = GradientSynchronizer(
            model=self.actor_model,
            dht_manager=self.node.dht_manager,
            target_batch_size=self.config.training.train_batch_size,
        )
        await self.gradient_sync.initialize()

        # Setup output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Trainer setup complete")

    def _setup_optimizers(self) -> None:
        """Setup optimizers and schedulers."""
        # Actor optimizer
        self.actor_optimizer = AdamW(
            self.actor_model.parameters(),
            lr=self.config.training.actor_lr,
            weight_decay=self.config.training.weight_decay,
        )

        # Critic optimizer
        if self.critic_model is not None:
            self.critic_optimizer = AdamW(
                self.critic_model.parameters(),
                lr=self.config.training.critic_lr,
                weight_decay=self.config.training.weight_decay,
            )

        # Schedulers
        total_steps = self.config.training.total_episodes // self.config.training.rollout_batch_size
        self.actor_scheduler = CosineAnnealingLR(
            self.actor_optimizer,
            T_max=total_steps,
            eta_min=self.config.training.actor_lr * 0.1,
        )

        if self.critic_optimizer is not None:
            self.critic_scheduler = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=total_steps,
                eta_min=self.config.training.critic_lr * 0.1,
            )

    async def train(
        self,
        prompts: List[str],
        num_epochs: Optional[int] = None,
    ) -> TrainingState:
        """
        Run the full training loop.

        Args:
            prompts: Training prompts
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Final training state
        """
        if not self.coordinator:
            await self.setup()

        num_epochs = num_epochs or (
            self.config.training.total_episodes // self.config.training.rollout_batch_size
        )

        logger.info(f"Starting training for {num_epochs} epochs")
        self._running = True
        start_time = time.time()

        try:
            for epoch in range(num_epochs):
                if not self._running:
                    break

                # Sample prompts for this epoch
                epoch_prompts = self._sample_prompts(
                    prompts,
                    self.config.training.rollout_batch_size,
                )

                # Run epoch
                metrics = await self._train_epoch(epoch_prompts)

                # Update state
                self.state.epoch = epoch
                self.state.policy_version = metrics.policy_version
                self.state.total_samples += metrics.num_rollouts
                self.state.total_time = time.time() - start_time

                if metrics.avg_reward > self.state.best_reward:
                    self.state.best_reward = metrics.avg_reward
                    await self._save_checkpoint("best")

                # Callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # Periodic saves
                if epoch % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint(f"epoch_{epoch}")

                # Logging
                if epoch % self.config.training.log_interval == 0:
                    self._log_metrics(metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            self._running = False

        # Final save
        await self._save_checkpoint("final")

        logger.info(
            f"Training complete. "
            f"Best reward: {self.state.best_reward:.4f}, "
            f"Total time: {self.state.total_time:.2f}s"
        )

        return self.state

    async def _train_epoch(self, prompts: List[str]) -> EpochMetrics:
        """Train for one epoch."""
        epoch_start = time.time()

        # Generate rollouts
        experiences = await self._generate_rollouts(prompts)

        # Compute rewards
        experiences = await self._compute_rewards(experiences)

        # Compute advantages
        experiences = self._compute_advantages(experiences)

        # Training updates
        metrics = await self._update_policy(experiences)

        # Synchronize gradients
        await self.gradient_sync.synchronize(
            step=self.state.global_step,
            local_batch_size=len(experiences),
        )

        # Update schedulers
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

        metrics.total_time = time.time() - epoch_start
        self._metrics_history.append(metrics)

        return metrics

    def _sample_prompts(
        self,
        prompts: List[str],
        batch_size: int,
    ) -> List[str]:
        """Sample prompts for an epoch."""
        import random
        if len(prompts) <= batch_size:
            return prompts
        return random.sample(prompts, batch_size)

    async def _generate_rollouts(
        self,
        prompts: List[str],
    ) -> List[Experience]:
        """Generate rollouts from actor model."""
        experiences = []

        # Use coordinator for distributed generation
        if self.coordinator:
            return await self.coordinator._generate_rollouts(
                prompts=prompts,
                actor_model=self.actor_model,
                reference_model=self.reference_model,
            )

        # Local generation fallback
        self.actor_model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # Encode prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=self.config.training.max_prompt_length,
                    truncation=True,
                )

                device = next(self.actor_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate
                outputs = self.actor_model.generate(
                    **inputs,
                    max_new_tokens=self.config.training.max_response_length,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Decode response
                response_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Get log probs
                log_probs = []
                if hasattr(outputs, "scores"):
                    for i, score in enumerate(outputs.scores):
                        if i < len(response_ids):
                            token_id = response_ids[i].item()
                            log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                            log_probs.append(log_prob)

                # Get reference log probs
                ref_log_probs = []
                if self.reference_model is not None:
                    ref_log_probs = self._get_ref_log_probs(prompt, response)

                exp = Experience(
                    prompt=prompt,
                    response=response,
                    log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                    policy_version=self.state.policy_version,
                )
                experiences.append(exp)

        return experiences

    def _get_ref_log_probs(
        self,
        prompt: str,
        response: str,
    ) -> List[float]:
        """Get log probs from reference model."""
        if self.reference_model is None:
            return []

        self.reference_model.eval()
        with torch.no_grad():
            # Encode full sequence
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                truncation=True,
            )

            device = next(self.reference_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = self.reference_model(**inputs)
            logits = outputs.logits

            # Get log probs for response tokens
            prompt_len = len(self.tokenizer.encode(prompt))
            log_probs = []

            for i in range(prompt_len, inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1].item()
                log_prob = torch.log_softmax(logits[0, i], dim=-1)[token_id].item()
                log_probs.append(log_prob)

            return log_probs

    async def _compute_rewards(
        self,
        experiences: List[Experience],
    ) -> List[Experience]:
        """Compute rewards for experiences."""
        if self.reward_model is None:
            # Use placeholder rewards
            for exp in experiences:
                exp.reward = 0.0
            return experiences

        self.reward_model.eval()
        with torch.no_grad():
            for exp in experiences:
                # Encode prompt + response
                full_text = exp.prompt + exp.response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                    truncation=True,
                )

                device = next(self.reward_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get reward
                outputs = self.reward_model(**inputs)

                # Handle different output formats
                if hasattr(outputs, "rewards"):
                    reward = outputs.rewards[0].item()
                elif hasattr(outputs, "logits"):
                    reward = outputs.logits[0, -1].item()
                else:
                    reward = outputs[0].item()

                # Scale and clip reward
                reward = reward * self.config.algorithm.reward_scale
                if self.config.algorithm.reward_clip is not None:
                    reward = max(-self.config.algorithm.reward_clip,
                               min(self.config.algorithm.reward_clip, reward))

                exp.reward = reward

        return experiences

    def _compute_advantages(
        self,
        experiences: List[Experience],
    ) -> List[Experience]:
        """Compute advantages using GAE."""
        from decentralized_verl.algorithms.ppo import compute_advantages

        if not experiences:
            return experiences

        # Get values from critic
        values = []
        if self.critic_model is not None:
            self.critic_model.eval()
            with torch.no_grad():
                for exp in experiences:
                    full_text = exp.prompt + exp.response
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                        truncation=True,
                    )

                    device = next(self.critic_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = self.critic_model(**inputs)

                    if hasattr(outputs, "value"):
                        value = outputs.value[0].item()
                    else:
                        value = outputs[0].item()

                    values.append(value)
                    exp.values = [value]

        # Compute advantages
        rewards = torch.tensor([exp.reward for exp in experiences])
        values_tensor = torch.tensor(values) if values else torch.zeros(len(experiences))

        advantages, returns = compute_advantages(
            rewards=rewards,
            values=values_tensor,
            gamma=self.config.algorithm.gamma,
            gae_lambda=self.config.algorithm.gae_lambda,
        )

        # Normalize advantages
        if self.config.algorithm.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i, exp in enumerate(experiences):
            exp.advantages = [advantages[i].item()]
            exp.returns = [returns[i].item()]

        return experiences

    async def _update_policy(
        self,
        experiences: List[Experience],
    ) -> EpochMetrics:
        """Update policy using PPO."""
        metrics = EpochMetrics(
            epoch=self.state.epoch,
            policy_version=self.state.policy_version,
            num_rollouts=len(experiences),
        )

        if not experiences:
            return metrics

        # Prepare batch
        batch = collate_experiences(experiences, self.tokenizer)

        # Training loop
        self.actor_model.train()
        if self.critic_model:
            self.critic_model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        for ppo_epoch in range(self.config.training.ppo_epochs):
            # Forward pass for actor
            self.actor_optimizer.zero_grad()

            # Get new log probs
            device = next(self.actor_model.parameters()).device
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                input_ids = input_ids.to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = self.actor_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

                # Compute new log probs
                log_probs = torch.log_softmax(logits, dim=-1)

            # Compute PPO loss
            advantages = batch.get("advantages")
            if advantages is not None:
                advantages = advantages.to(device)

                old_log_probs = batch.get("old_log_probs")
                if old_log_probs is not None:
                    old_log_probs = old_log_probs.to(device)

                    # Simplified PPO loss
                    ratio = torch.exp(log_probs.mean(dim=-1) - old_log_probs.mean(dim=-1))
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.config.training.clip_range,
                        1 + self.config.training.clip_range,
                    )

                    policy_loss = -torch.min(
                        ratio * advantages.squeeze(),
                        clipped_ratio * advantages.squeeze(),
                    ).mean()

                    # Entropy bonus
                    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
                    policy_loss = policy_loss - self.config.training.entropy_coef * entropy

                    # Backward
                    policy_loss.backward()

                    # Clip gradients
                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_model.parameters(),
                            self.config.training.max_grad_norm,
                        )

                    self.actor_optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_entropy += entropy.item()

            # Value loss
            if self.critic_model and self.critic_optimizer:
                self.critic_optimizer.zero_grad()

                returns = batch.get("returns")
                if returns is not None:
                    returns = returns.to(device)

                    critic_outputs = self.critic_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    if hasattr(critic_outputs, "value"):
                        values = critic_outputs.value
                    else:
                        values = critic_outputs[0]

                    value_loss = nn.functional.mse_loss(values.squeeze(), returns.squeeze())

                    value_loss.backward()

                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.critic_model.parameters(),
                            self.config.training.max_grad_norm,
                        )

                    self.critic_optimizer.step()
                    total_value_loss += value_loss.item()

            self.state.global_step += 1

        # Average metrics
        n_epochs = self.config.training.ppo_epochs
        metrics.policy_loss = total_policy_loss / n_epochs
        metrics.value_loss = total_value_loss / n_epochs
        metrics.entropy = total_entropy / n_epochs
        metrics.avg_reward = sum(e.reward for e in experiences) / len(experiences)

        return metrics

    def _log_metrics(self, metrics: EpochMetrics) -> None:
        """Log training metrics."""
        logger.info(
            f"Epoch {metrics.epoch} | "
            f"Reward: {metrics.avg_reward:.4f} | "
            f"Policy Loss: {metrics.policy_loss:.4f} | "
            f"Value Loss: {metrics.value_loss:.4f} | "
            f"KL: {metrics.avg_kl:.4f} | "
            f"Time: {metrics.total_time:.2f}s"
        )

    async def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name

        # Save model
        torch.save(
            self.actor_model.state_dict(),
            checkpoint_path / "actor_model.pt",
        )

        if self.critic_model:
            torch.save(
                self.critic_model.state_dict(),
                checkpoint_path / "critic_model.pt",
            )

        # Save optimizer states
        torch.save(
            self.actor_optimizer.state_dict(),
            checkpoint_path / "actor_optimizer.pt",
        )

        # Save training state
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "policy_version": self.state.policy_version,
            "best_reward": self.state.best_reward,
            "total_samples": self.state.total_samples,
        }

        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state_dict, f)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    async def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(path)

        # Load model
        self.actor_model.load_state_dict(
            torch.load(checkpoint_path / "actor_model.pt")
        )

        if self.critic_model and (checkpoint_path / "critic_model.pt").exists():
            self.critic_model.load_state_dict(
                torch.load(checkpoint_path / "critic_model.pt")
            )

        # Load optimizer
        if (checkpoint_path / "actor_optimizer.pt").exists():
            self.actor_optimizer.load_state_dict(
                torch.load(checkpoint_path / "actor_optimizer.pt")
            )

        # Load state
        with open(checkpoint_path / "training_state.json", "r") as f:
            state_dict = json.load(f)

        self.state.global_step = state_dict["global_step"]
        self.state.epoch = state_dict["epoch"]
        self.state.policy_version = state_dict["policy_version"]
        self.state.best_reward = state_dict["best_reward"]
        self.state.total_samples = state_dict["total_samples"]

        logger.info(f"Loaded checkpoint from {path}")

    def add_callback(self, callback: Callable[[EpochMetrics], None]) -> None:
        """Add training callback."""
        self._callbacks.append(callback)

    async def cleanup(self) -> None:
        """Cleanup trainer resources."""
        if self.coordinator:
            await self.coordinator.stop()
        if self.node:
            await self.node.stop()

        logger.info("Trainer cleanup complete")
