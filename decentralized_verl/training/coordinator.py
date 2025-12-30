"""Decentralized training coordinator for RLHF."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
from enum import Enum

from decentralized_verl.core.config import DecentralizedConfig, Algorithm, NodeRole
from decentralized_verl.core.protocol import (
    Message,
    MessageType,
    PeerInfo,
    RolloutRequest,
    RolloutResponse,
    GradientUpdate,
)
from decentralized_verl.network.dht_manager import DHTManager
from decentralized_verl.network.peer_router import PeerRouter, WorkloadBalancer, RoutingStrategy
from decentralized_verl.network.gossip import GossipProtocol, PolicyVersionGossip
from decentralized_verl.training.experience import ExperienceBuffer, Experience

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Current phase of the training loop."""
    IDLE = "idle"
    ROLLOUT = "rollout"
    REWARD = "reward"
    TRAINING = "training"
    SYNCHRONIZING = "synchronizing"


@dataclass
class EpochMetrics:
    """Metrics for a training epoch."""
    epoch: int = 0
    policy_version: int = 0
    num_rollouts: int = 0
    avg_reward: float = 0.0
    avg_kl: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    rollout_time: float = 0.0
    training_time: float = 0.0
    total_time: float = 0.0


class DecentralizedCoordinator:
    """
    Coordinates decentralized RLHF training across P2P network.

    Replaces veRL's centralized Ray-based orchestration with
    DHT-based peer coordination. Implements the hybrid-controller
    pattern in a decentralized manner.

    Key responsibilities:
    - Coordinate rollout generation across actor nodes
    - Manage reward computation distribution
    - Orchestrate gradient updates and synchronization
    - Maintain policy version consistency
    - Handle fault tolerance and recovery
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        node_id: str,
        dht_manager: DHTManager,
    ):
        """
        Initialize the decentralized coordinator.

        Args:
            config: Training configuration
            node_id: This node's identifier
            dht_manager: DHT manager for network communication
        """
        self.config = config
        self.node_id = node_id
        self.dht_manager = dht_manager

        # State
        self.policy_version = 0
        self.current_epoch = 0
        self.current_phase = TrainingPhase.IDLE
        self.is_leader = False  # Leader election (not centralized - rotates)

        # Peer management
        self.router = PeerRouter(strategy=RoutingStrategy.LEAST_LOADED)
        self.balancer = WorkloadBalancer(self.router)
        self.known_peers: Dict[str, PeerInfo] = {}

        # Gossip for state sync
        self.gossip = GossipProtocol(
            node_id=node_id,
            fanout=config.network.gossip_fanout,
            gossip_interval=config.network.gossip_interval,
        )
        self.version_gossip = PolicyVersionGossip(self.gossip)

        # Experience buffer
        self.experience_buffer = ExperienceBuffer(
            max_size=config.algorithm.buffer_size,
            min_size=config.algorithm.min_buffer_size,
        )

        # Pending requests
        self._pending_rollouts: Dict[str, asyncio.Future] = {}
        self._pending_rewards: Dict[str, asyncio.Future] = {}
        self._pending_gradients: Dict[str, asyncio.Future] = {}

        # Callbacks
        self._epoch_callbacks: List[Callable[[EpochMetrics], None]] = []

        # Control
        self._running = False
        self._epoch_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the coordinator."""
        logger.info(f"Starting coordinator on node {self.node_id}")

        # Start gossip
        self.gossip.set_send_function(self._send_gossip_message)
        await self.gossip.start()

        # Register for version updates
        self.version_gossip.on_version_update(self._on_policy_version_update)

        # Start peer discovery loop
        self._running = True
        asyncio.create_task(self._peer_update_loop())

        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        logger.info("Stopping coordinator...")
        self._running = False
        await self.gossip.stop()
        logger.info("Coordinator stopped")

    async def _send_gossip_message(self, message: Message, peer_id: str) -> None:
        """Send gossip message to peer."""
        await self.dht_manager.send_to_peer(message, peer_id)

    async def _peer_update_loop(self) -> None:
        """Periodically update peer information."""
        while self._running:
            try:
                peers = await self.dht_manager.discover_peers()
                for peer in peers:
                    self.known_peers[peer.peer_id] = peer

                self.router.set_peers = peers
                self.gossip.set_peers(list(self.known_peers.values()))

                # Update leader (simple: node with lowest ID)
                all_ids = sorted([self.node_id] + list(self.known_peers.keys()))
                self.is_leader = (all_ids[0] == self.node_id)

                await asyncio.sleep(self.config.network.peer_discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer update loop: {e}")
                await asyncio.sleep(5)

    def _on_policy_version_update(self, new_version: int) -> None:
        """Handle policy version update from network."""
        if new_version > self.policy_version:
            logger.info(f"Policy version updated: {self.policy_version} -> {new_version}")
            self.policy_version = new_version

    async def run_epoch(
        self,
        prompts: List[str],
        actor_model: Any,
        critic_model: Any,
        reward_model: Any,
        reference_model: Optional[Any] = None,
    ) -> EpochMetrics:
        """
        Run one training epoch.

        This implements the on-policy RLHF loop:
        1. Generate rollouts from current policy
        2. Compute rewards
        3. Compute advantages (GAE)
        4. Update policy with PPO/GRPO
        5. Synchronize across network

        Args:
            prompts: Training prompts for this epoch
            actor_model: Actor/policy model
            critic_model: Critic/value model
            reward_model: Reward model
            reference_model: Reference model for KL penalty

        Returns:
            Epoch metrics
        """
        async with self._epoch_lock:
            self.current_epoch += 1
            epoch_start = time.time()
            metrics = EpochMetrics(
                epoch=self.current_epoch,
                policy_version=self.policy_version,
            )

            logger.info(f"Starting epoch {self.current_epoch}")

            # Phase 1: Generate rollouts
            self.current_phase = TrainingPhase.ROLLOUT
            rollout_start = time.time()

            experiences = await self._generate_rollouts(
                prompts=prompts,
                actor_model=actor_model,
                reference_model=reference_model,
            )
            metrics.num_rollouts = len(experiences)
            metrics.rollout_time = time.time() - rollout_start

            logger.info(f"Generated {len(experiences)} rollouts in {metrics.rollout_time:.2f}s")

            # Phase 2: Compute rewards
            self.current_phase = TrainingPhase.REWARD

            experiences = await self._compute_rewards(
                experiences=experiences,
                reward_model=reward_model,
            )

            # Add to buffer
            for exp in experiences:
                self.experience_buffer.add(exp)

            metrics.avg_reward = sum(e.reward for e in experiences) / len(experiences)
            logger.info(f"Average reward: {metrics.avg_reward:.4f}")

            # Phase 3: Training
            self.current_phase = TrainingPhase.TRAINING
            training_start = time.time()

            training_metrics = await self._train_step(
                actor_model=actor_model,
                critic_model=critic_model,
                experiences=experiences,
            )

            metrics.policy_loss = training_metrics.get("policy_loss", 0.0)
            metrics.value_loss = training_metrics.get("value_loss", 0.0)
            metrics.entropy = training_metrics.get("entropy", 0.0)
            metrics.avg_kl = training_metrics.get("kl", 0.0)
            metrics.clip_fraction = training_metrics.get("clip_fraction", 0.0)
            metrics.training_time = time.time() - training_start

            # Phase 4: Synchronize
            self.current_phase = TrainingPhase.SYNCHRONIZING

            await self._synchronize_policy()

            # Update policy version
            self.policy_version += 1
            await self.version_gossip.announce_version(self.policy_version)

            metrics.total_time = time.time() - epoch_start
            self.current_phase = TrainingPhase.IDLE

            # Trigger callbacks
            for callback in self._epoch_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in epoch callback: {e}")

            logger.info(
                f"Epoch {self.current_epoch} complete: "
                f"reward={metrics.avg_reward:.4f}, "
                f"kl={metrics.avg_kl:.4f}, "
                f"time={metrics.total_time:.2f}s"
            )

            return metrics

    async def _generate_rollouts(
        self,
        prompts: List[str],
        actor_model: Any,
        reference_model: Optional[Any],
    ) -> List[Experience]:
        """Generate rollouts from actor nodes."""
        experiences = []

        # Get available actor peers
        actor_peers = [
            p for p in self.known_peers.values()
            if p.role in (NodeRole.ACTOR.value, NodeRole.HYBRID.value)
            and p.is_alive()
        ]

        if actor_peers:
            # Distribute prompts across peers
            distribution = self.balancer.distribute_prompts(prompts, actor_peers)

            # Send requests and gather responses
            tasks = []
            for peer_id, peer_prompts in distribution.items():
                if peer_prompts:
                    task = self._request_rollouts_from_peer(
                        peer_id=peer_id,
                        prompts=peer_prompts,
                    )
                    tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"Rollout request failed: {response}")
                    continue
                if isinstance(response, list):
                    experiences.extend(response)

        # Generate locally if needed
        remaining = len(prompts) - len(experiences)
        if remaining > 0 and actor_model is not None:
            local_prompts = prompts[len(experiences):]
            local_experiences = await self._generate_local_rollouts(
                prompts=local_prompts,
                actor_model=actor_model,
                reference_model=reference_model,
            )
            experiences.extend(local_experiences)

        return experiences

    async def _request_rollouts_from_peer(
        self,
        peer_id: str,
        prompts: List[str],
    ) -> List[Experience]:
        """Request rollouts from a specific peer."""
        request = RolloutRequest(
            prompts=prompts,
            prompt_ids=[f"p{i}" for i in range(len(prompts))],
            policy_version=self.policy_version,
            max_new_tokens=self.config.training.max_response_length,
            temperature=1.0,
            top_p=1.0,
        )

        message = Message(
            message_type=MessageType.ROLLOUT_REQUEST,
            sender_id=self.node_id,
            payload=request.to_dict(),
        )

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_rollouts[request.request_id] = future

        # Send request
        await self.dht_manager.send_to_peer(message, peer_id)

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=120.0)
            return self._responses_to_experiences(prompts, response)
        except asyncio.TimeoutError:
            logger.warning(f"Rollout request to {peer_id} timed out")
            return []
        finally:
            self._pending_rollouts.pop(request.request_id, None)

    async def _generate_local_rollouts(
        self,
        prompts: List[str],
        actor_model: Any,
        reference_model: Optional[Any],
    ) -> List[Experience]:
        """Generate rollouts locally."""
        experiences = []

        # Generate responses
        responses, log_probs = await actor_model.generate(
            prompts=prompts,
            max_new_tokens=self.config.training.max_response_length,
            temperature=1.0,
            top_p=1.0,
            return_log_probs=True,
        )

        # Get reference log probs if available
        ref_log_probs = None
        if reference_model is not None and not self.config.training.reference_free:
            ref_log_probs = await reference_model.get_log_probs(
                prompts=prompts,
                responses=responses,
            )

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            exp = Experience(
                prompt=prompt,
                response=response,
                log_probs=log_probs[i] if log_probs else [],
                ref_log_probs=ref_log_probs[i] if ref_log_probs else [],
                policy_version=self.policy_version,
            )
            experiences.append(exp)

        return experiences

    def _responses_to_experiences(
        self,
        prompts: List[str],
        response: RolloutResponse,
    ) -> List[Experience]:
        """Convert rollout responses to experiences."""
        experiences = []
        for i, (prompt, resp) in enumerate(zip(prompts, response.responses)):
            exp = Experience(
                prompt=prompt,
                response=resp,
                log_probs=response.log_probs[i] if response.log_probs else [],
                policy_version=response.policy_version,
            )
            experiences.append(exp)
        return experiences

    async def _compute_rewards(
        self,
        experiences: List[Experience],
        reward_model: Any,
    ) -> List[Experience]:
        """Compute rewards for experiences."""
        # Get reward peers
        reward_peers = [
            p for p in self.known_peers.values()
            if p.role == NodeRole.REWARD.value
            and p.is_alive()
        ]

        if reward_peers:
            # Distribute reward computation
            # For simplicity, compute locally here
            pass

        # Compute rewards locally
        if reward_model is not None:
            prompts = [e.prompt for e in experiences]
            responses = [e.response for e in experiences]

            rewards = await reward_model.compute_rewards(
                prompts=prompts,
                responses=responses,
            )

            for exp, reward in zip(experiences, rewards):
                exp.reward = reward

        return experiences

    async def _train_step(
        self,
        actor_model: Any,
        critic_model: Any,
        experiences: List[Experience],
    ) -> Dict[str, float]:
        """Execute one training step."""
        from decentralized_verl.algorithms.ppo import compute_advantages, ppo_loss

        # Convert experiences to training batch
        batch = self._prepare_training_batch(experiences)

        # Compute advantages
        if critic_model is not None:
            values = await critic_model.get_values(
                prompts=[e.prompt for e in experiences],
                responses=[e.response for e in experiences],
            )
            batch["values"] = values

            advantages, returns = compute_advantages(
                rewards=batch["rewards"],
                values=values,
                gamma=self.config.algorithm.gamma,
                gae_lambda=self.config.algorithm.gae_lambda,
            )
            batch["advantages"] = advantages
            batch["returns"] = returns

            # Normalize advantages
            if self.config.algorithm.normalize_advantages:
                batch["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}

        for _ in range(self.config.training.ppo_epochs):
            # Compute PPO loss
            loss_dict = await actor_model.train_step(
                batch=batch,
                clip_range=self.config.training.clip_range,
                vf_coef=self.config.training.vf_coef,
                entropy_coef=self.config.training.entropy_coef,
            )

            for key, value in loss_dict.items():
                metrics[key] = metrics.get(key, 0.0) + value / self.config.training.ppo_epochs

            # Early stopping on KL
            if self.config.training.target_kl is not None:
                if metrics.get("kl", 0) > self.config.training.target_kl * 1.5:
                    logger.info(f"Early stopping: KL={metrics['kl']:.4f}")
                    break

        return metrics

    def _prepare_training_batch(
        self,
        experiences: List[Experience],
    ) -> Dict[str, Any]:
        """Prepare training batch from experiences."""
        import torch

        return {
            "prompts": [e.prompt for e in experiences],
            "responses": [e.response for e in experiences],
            "rewards": torch.tensor([e.reward for e in experiences]),
            "old_log_probs": [e.log_probs for e in experiences],
            "ref_log_probs": [e.ref_log_probs for e in experiences],
        }

    async def _synchronize_policy(self) -> None:
        """Synchronize policy across network."""
        # Get peers to sync with
        peers = list(self.known_peers.values())
        if not peers:
            return

        # Broadcast sync message
        sync_message = Message(
            message_type=MessageType.POLICY_SYNC,
            sender_id=self.node_id,
            payload={
                "policy_version": self.policy_version,
                "node_id": self.node_id,
            },
        )

        await self.dht_manager.broadcast(sync_message)

        # Update DHT with latest version
        await self.dht_manager.update_policy_version(self.policy_version)

    def on_epoch_complete(self, callback: Callable[[EpochMetrics], None]) -> None:
        """Register callback for epoch completion."""
        self._epoch_callbacks.append(callback)

    async def handle_rollout_response(self, message: Message) -> None:
        """Handle incoming rollout response."""
        response = RolloutResponse.from_dict(message.payload)
        future = self._pending_rollouts.get(response.request_id)
        if future and not future.done():
            future.set_result(response)

    async def handle_gradient_update(self, message: Message) -> None:
        """Handle incoming gradient update."""
        update = GradientUpdate.from_dict(message.payload)
        future = self._pending_gradients.get(f"{update.step}:{message.sender_id}")
        if future and not future.done():
            future.set_result(update)

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "node_id": self.node_id,
            "is_leader": self.is_leader,
            "policy_version": self.policy_version,
            "current_epoch": self.current_epoch,
            "current_phase": self.current_phase.value,
            "known_peers": len(self.known_peers),
            "buffer_size": len(self.experience_buffer),
        }
