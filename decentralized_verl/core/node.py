"""Decentralized node implementation for veRL."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import time
import torch
from pathlib import Path

from decentralized_verl.core.config import (
    DecentralizedConfig,
    NodeRole,
    GenerationBackend,
    TrainingBackend,
)
from decentralized_verl.core.protocol import (
    Message,
    MessageType,
    PeerInfo,
    RolloutRequest,
    RolloutResponse,
    GradientUpdate,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeState:
    """State of a decentralized node."""

    policy_version: int = 0
    current_step: int = 0
    is_training: bool = False
    is_generating: bool = False
    load: float = 0.0
    last_checkpoint: int = 0
    connected_peers: int = 0
    total_rollouts: int = 0
    total_training_steps: int = 0


class DecentralizedNode:
    """
    A node in the decentralized RLHF training network.

    Each node can serve as an actor (generator), critic (value estimator),
    reward model, or a hybrid of multiple roles. Nodes communicate via
    DHT-based P2P network and coordinate training without central server.
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        node_id: Optional[str] = None,
    ):
        """
        Initialize a decentralized node.

        Args:
            config: Configuration for the node
            node_id: Optional unique identifier (generated if not provided)
        """
        self.config = config
        self.node_id = node_id or self._generate_node_id()
        self.state = NodeState()

        # P2P network components (initialized later)
        self.dht_manager = None
        self.peer_info: Optional[PeerInfo] = None
        self.known_peers: Dict[str, PeerInfo] = {}

        # Model components (initialized later)
        self.actor_model = None
        self.critic_model = None
        self.reference_model = None
        self.reward_model = None

        # Training components
        self.optimizer = None
        self.scheduler = None

        # Message handlers
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._setup_message_handlers()

        # Async components
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        logger.info(f"Initialized DecentralizedNode {self.node_id}")

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        import uuid
        content = f"{uuid.uuid4()}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _setup_message_handlers(self) -> None:
        """Setup message handlers for different message types."""
        self._message_handlers = {
            MessageType.PEER_ANNOUNCE: self._handle_peer_announce,
            MessageType.PEER_DISCOVERY: self._handle_peer_discovery,
            MessageType.ROLLOUT_REQUEST: self._handle_rollout_request,
            MessageType.ROLLOUT_RESPONSE: self._handle_rollout_response,
            MessageType.GRADIENT_UPDATE: self._handle_gradient_update,
            MessageType.POLICY_SYNC: self._handle_policy_sync,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.STATUS_REQUEST: self._handle_status_request,
            MessageType.CHECKPOINT_REQUEST: self._handle_checkpoint_request,
            MessageType.ERROR: self._handle_error,
        }

    async def start(self) -> None:
        """Start the node and connect to the network."""
        logger.info(f"Starting node {self.node_id}")

        # Initialize DHT and connect to network
        from decentralized_verl.network.dht_manager import DHTManager

        self.dht_manager = DHTManager(self.config.network, self.node_id)
        await self.dht_manager.start()

        # Create peer info
        self.peer_info = PeerInfo(
            peer_id=self.node_id,
            multiaddr=self.dht_manager.get_multiaddr(),
            role=self.config.node_role.value,
            blocks_served=self.config.block_indices or list(
                range(self.config.num_blocks_to_serve)
            ),
            capabilities=self._get_capabilities(),
        )

        # Announce self to network
        await self.dht_manager.announce_peer(self.peer_info)

        # Discover existing peers
        await self._discover_peers()

        # Initialize models based on role
        await self._initialize_models()

        # Start background tasks
        self._running = True
        self._tasks.add(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.add(asyncio.create_task(self._peer_discovery_loop()))
        self._tasks.add(asyncio.create_task(self._message_processing_loop()))

        logger.info(f"Node {self.node_id} started successfully")

    async def stop(self) -> None:
        """Stop the node gracefully."""
        logger.info(f"Stopping node {self.node_id}")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Disconnect from network
        if self.dht_manager:
            await self.dht_manager.stop()

        logger.info(f"Node {self.node_id} stopped")

    def _get_capabilities(self) -> Dict[str, Any]:
        """Get node capabilities based on hardware and config."""
        capabilities = {
            "role": self.config.node_role.value,
            "generation_backend": self.config.generation_backend.value,
            "training_backend": self.config.training_backend.value,
            "model_name": self.config.model.model_name_or_path,
            "num_blocks": self.config.num_blocks_to_serve,
            "dtype": self.config.model.dtype,
            "use_lora": self.config.model.use_lora,
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            capabilities["gpu_count"] = torch.cuda.device_count()
            capabilities["gpu_memory"] = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            ]
            capabilities["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]

        return capabilities

    async def _discover_peers(self) -> None:
        """Discover peers in the network."""
        if not self.dht_manager:
            return

        peers = await self.dht_manager.discover_peers()
        for peer in peers:
            if peer.peer_id != self.node_id:
                self.known_peers[peer.peer_id] = peer
                logger.debug(f"Discovered peer: {peer.peer_id}")

        self.state.connected_peers = len(self.known_peers)
        logger.info(f"Discovered {len(self.known_peers)} peers")

    async def _initialize_models(self) -> None:
        """Initialize models based on node role."""
        role = self.config.node_role

        if role in (NodeRole.ACTOR, NodeRole.HYBRID):
            await self._initialize_actor_model()

        if role in (NodeRole.CRITIC, NodeRole.HYBRID):
            await self._initialize_critic_model()

        if role in (NodeRole.REFERENCE, NodeRole.HYBRID):
            await self._initialize_reference_model()

        if role == NodeRole.REWARD:
            await self._initialize_reward_model()

    async def _initialize_actor_model(self) -> None:
        """Initialize the actor (policy) model."""
        from decentralized_verl.workers.actor import ActorWorker

        logger.info("Initializing actor model...")
        self.actor_model = ActorWorker(
            config=self.config,
            block_indices=self.config.block_indices,
        )
        await self.actor_model.initialize()
        logger.info("Actor model initialized")

    async def _initialize_critic_model(self) -> None:
        """Initialize the critic (value) model."""
        from decentralized_verl.workers.critic import CriticWorker

        logger.info("Initializing critic model...")
        self.critic_model = CriticWorker(
            config=self.config,
            block_indices=self.config.block_indices,
        )
        await self.critic_model.initialize()
        logger.info("Critic model initialized")

    async def _initialize_reference_model(self) -> None:
        """Initialize the reference model."""
        from decentralized_verl.workers.reference import ReferenceWorker

        logger.info("Initializing reference model...")
        self.reference_model = ReferenceWorker(
            config=self.config,
            block_indices=self.config.block_indices,
        )
        await self.reference_model.initialize()
        logger.info("Reference model initialized")

    async def _initialize_reward_model(self) -> None:
        """Initialize the reward model."""
        from decentralized_verl.workers.reward import RewardWorker

        logger.info("Initializing reward model...")
        self.reward_model = RewardWorker(config=self.config)
        await self.reward_model.initialize()
        logger.info("Reward model initialized")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to peers."""
        while self._running:
            try:
                # Update peer info
                if self.peer_info:
                    self.peer_info.last_seen = time.time()
                    self.peer_info.load = self.state.load
                    await self.dht_manager.announce_peer(self.peer_info)

                # Send heartbeat to known peers
                heartbeat = Message(
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.node_id,
                    payload={
                        "policy_version": self.state.policy_version,
                        "current_step": self.state.current_step,
                        "load": self.state.load,
                    },
                )

                await self._broadcast_message(heartbeat)

                await asyncio.sleep(self.config.network.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)

    async def _peer_discovery_loop(self) -> None:
        """Periodically discover new peers."""
        while self._running:
            try:
                await self._discover_peers()
                await asyncio.sleep(self.config.network.peer_discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(5)

    async def _message_processing_loop(self) -> None:
        """Process incoming messages."""
        while self._running:
            try:
                if self.dht_manager:
                    message = await self.dht_manager.receive_message()
                    if message:
                        await self._handle_message(message)
                else:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)

    async def _handle_message(self, message: Message) -> None:
        """Route message to appropriate handler."""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                response = await handler(message)
                if response:
                    await self._send_message(response, message.sender_id)
            except Exception as e:
                logger.error(f"Error handling message {message.message_type}: {e}")
                error_msg = Message(
                    message_type=MessageType.ERROR,
                    sender_id=self.node_id,
                    payload={"error": str(e), "original_message_id": message.message_id},
                )
                await self._send_message(error_msg, message.sender_id)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")

    async def _broadcast_message(self, message: Message) -> None:
        """Broadcast message to all known peers."""
        if self.dht_manager:
            await self.dht_manager.broadcast(message)

    async def _send_message(self, message: Message, peer_id: str) -> None:
        """Send message to specific peer."""
        if self.dht_manager:
            await self.dht_manager.send_to_peer(message, peer_id)

    # Message handlers

    async def _handle_peer_announce(self, message: Message) -> Optional[Message]:
        """Handle peer announcement."""
        peer_info = PeerInfo.from_dict(message.payload)
        self.known_peers[peer_info.peer_id] = peer_info
        self.state.connected_peers = len(self.known_peers)
        logger.debug(f"Peer announced: {peer_info.peer_id}")
        return None

    async def _handle_peer_discovery(self, message: Message) -> Optional[Message]:
        """Handle peer discovery request."""
        peer_list = [p.to_dict() for p in self.known_peers.values()]
        return Message(
            message_type=MessageType.PEER_LIST,
            sender_id=self.node_id,
            payload={"peers": peer_list},
        )

    async def _handle_rollout_request(self, message: Message) -> Optional[Message]:
        """Handle rollout generation request."""
        if not self.actor_model:
            return Message(
                message_type=MessageType.ERROR,
                sender_id=self.node_id,
                payload={"error": "Node does not have actor model"},
            )

        request = RolloutRequest.from_dict(message.payload)
        self.state.is_generating = True

        try:
            start_time = time.time()
            responses, log_probs = await self.actor_model.generate(
                prompts=request.prompts,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
            )
            generation_time = time.time() - start_time

            response = RolloutResponse(
                request_id=request.request_id,
                responses=responses,
                response_ids=[f"{request.request_id}_{i}" for i in range(len(responses))],
                log_probs=log_probs,
                policy_version=self.state.policy_version,
                generation_time=generation_time,
                worker_id=self.node_id,
            )

            self.state.total_rollouts += len(responses)

            return Message(
                message_type=MessageType.ROLLOUT_RESPONSE,
                sender_id=self.node_id,
                payload=response.to_dict(),
            )
        finally:
            self.state.is_generating = False

    async def _handle_rollout_response(self, message: Message) -> Optional[Message]:
        """Handle rollout response."""
        response = RolloutResponse.from_dict(message.payload)
        # Process response - typically handled by coordinator
        logger.debug(f"Received rollout response: {response.request_id}")
        return None

    async def _handle_gradient_update(self, message: Message) -> Optional[Message]:
        """Handle gradient update from peer."""
        update = GradientUpdate.from_dict(message.payload)
        # Apply gradient update to local model
        if self.actor_model:
            await self.actor_model.apply_gradient(update.gradients)
        return Message(
            message_type=MessageType.GRADIENT_ACK,
            sender_id=self.node_id,
            payload={"step": update.step, "success": True},
        )

    async def _handle_policy_sync(self, message: Message) -> Optional[Message]:
        """Handle policy synchronization."""
        policy_version = message.payload.get("policy_version", 0)
        if policy_version > self.state.policy_version:
            # Request full sync
            return Message(
                message_type=MessageType.CHECKPOINT_REQUEST,
                sender_id=self.node_id,
                payload={"policy_version": policy_version},
            )
        return None

    async def _handle_heartbeat(self, message: Message) -> Optional[Message]:
        """Handle heartbeat from peer."""
        peer_id = message.sender_id
        if peer_id in self.known_peers:
            self.known_peers[peer_id].last_seen = time.time()
            self.known_peers[peer_id].load = message.payload.get("load", 0)
        return None

    async def _handle_status_request(self, message: Message) -> Optional[Message]:
        """Handle status request."""
        return Message(
            message_type=MessageType.STATUS_RESPONSE,
            sender_id=self.node_id,
            payload={
                "policy_version": self.state.policy_version,
                "current_step": self.state.current_step,
                "is_training": self.state.is_training,
                "is_generating": self.state.is_generating,
                "load": self.state.load,
                "connected_peers": self.state.connected_peers,
                "total_rollouts": self.state.total_rollouts,
                "total_training_steps": self.state.total_training_steps,
            },
        )

    async def _handle_checkpoint_request(self, message: Message) -> Optional[Message]:
        """Handle checkpoint request."""
        # Return checkpoint data in chunks
        if self.actor_model:
            checkpoint = await self.actor_model.get_checkpoint()
            return Message(
                message_type=MessageType.CHECKPOINT_RESPONSE,
                sender_id=self.node_id,
                payload={
                    "policy_version": self.state.policy_version,
                    "checkpoint": checkpoint,
                },
            )
        return None

    async def _handle_error(self, message: Message) -> Optional[Message]:
        """Handle error message."""
        logger.error(f"Error from {message.sender_id}: {message.payload.get('error')}")
        return None

    # Public API

    async def request_rollouts(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[RolloutResponse]:
        """Request rollouts from actor nodes in the network."""
        request = RolloutRequest(
            prompts=prompts,
            prompt_ids=[f"prompt_{i}" for i in range(len(prompts))],
            policy_version=self.state.policy_version,
            **kwargs,
        )

        # Find available actor nodes
        actor_peers = [
            p for p in self.known_peers.values()
            if p.role in (NodeRole.ACTOR.value, NodeRole.HYBRID.value)
            and p.is_alive()
        ]

        if not actor_peers:
            # Generate locally if we have actor model
            if self.actor_model:
                responses, log_probs = await self.actor_model.generate(
                    prompts=prompts,
                    **kwargs,
                )
                return [RolloutResponse(
                    request_id=request.request_id,
                    responses=responses,
                    response_ids=[f"{request.request_id}_{i}" for i in range(len(responses))],
                    log_probs=log_probs,
                    policy_version=self.state.policy_version,
                    worker_id=self.node_id,
                )]
            raise RuntimeError("No actor nodes available")

        # Distribute prompts across actor nodes
        responses = []
        batch_size = len(prompts) // len(actor_peers) + 1

        for i, peer in enumerate(actor_peers):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(prompts))
            if batch_start >= len(prompts):
                break

            batch_prompts = prompts[batch_start:batch_end]
            batch_request = RolloutRequest(
                prompts=batch_prompts,
                prompt_ids=[f"prompt_{j}" for j in range(batch_start, batch_end)],
                policy_version=self.state.policy_version,
                **kwargs,
            )

            message = Message(
                message_type=MessageType.ROLLOUT_REQUEST,
                sender_id=self.node_id,
                payload=batch_request.to_dict(),
            )

            await self._send_message(message, peer.peer_id)

        # Wait for responses (simplified - in practice use async gathering)
        # This is handled by the coordinator
        return responses

    def get_status(self) -> Dict[str, Any]:
        """Get current node status."""
        return {
            "node_id": self.node_id,
            "role": self.config.node_role.value,
            "policy_version": self.state.policy_version,
            "current_step": self.state.current_step,
            "is_training": self.state.is_training,
            "is_generating": self.state.is_generating,
            "connected_peers": self.state.connected_peers,
            "known_peers": list(self.known_peers.keys()),
            "total_rollouts": self.state.total_rollouts,
            "total_training_steps": self.state.total_training_steps,
        }
