"""Gossip protocol for state synchronization in decentralized veRL."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import struct

from decentralized_verl.core.protocol import Message, MessageType, PeerInfo

logger = logging.getLogger(__name__)


@dataclass
class GossipState:
    """State shared via gossip protocol."""
    key: str
    value: Any
    version: int
    timestamp: float = field(default_factory=time.time)
    origin_node: str = ""
    ttl: int = 10  # Hops before expiry

    def __hash__(self):
        return hash(f"{self.key}:{self.version}")

    @property
    def state_id(self) -> str:
        """Unique identifier for this state version."""
        content = f"{self.key}:{self.version}:{self.origin_node}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class GossipMessage:
    """Message format for gossip protocol."""
    message_type: str  # "push", "pull", "push_pull"
    sender_id: str
    states: List[GossipState]
    vector_clock: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class GossipProtocol:
    """
    Implements epidemic (gossip) protocol for state synchronization.

    Used for:
    - Policy version propagation
    - Peer health status
    - Training metrics aggregation
    - Lightweight state synchronization
    """

    def __init__(
        self,
        node_id: str,
        fanout: int = 6,
        gossip_interval: float = 1.0,
        state_ttl: float = 60.0,
    ):
        """
        Initialize gossip protocol.

        Args:
            node_id: This node's identifier
            fanout: Number of peers to gossip with each round
            gossip_interval: Seconds between gossip rounds
            state_ttl: Time-to-live for state entries
        """
        self.node_id = node_id
        self.fanout = fanout
        self.gossip_interval = gossip_interval
        self.state_ttl = state_ttl

        # Local state store
        self._states: Dict[str, GossipState] = {}
        self._seen_state_ids: Set[str] = set()
        self._vector_clock: Dict[str, int] = {node_id: 0}

        # Callbacks
        self._state_handlers: Dict[str, Callable] = {}

        # Control
        self._running = False
        self._peers: List[PeerInfo] = []
        self._send_func: Optional[Callable] = None

        self._lock = asyncio.Lock()

    def set_peers(self, peers: List[PeerInfo]) -> None:
        """Update known peers list."""
        self._peers = [p for p in peers if p.peer_id != self.node_id]

    def set_send_function(self, send_func: Callable) -> None:
        """Set function for sending messages to peers."""
        self._send_func = send_func

    def register_handler(
        self,
        key_prefix: str,
        handler: Callable[[GossipState], None],
    ) -> None:
        """
        Register handler for state updates matching key prefix.

        Args:
            key_prefix: Prefix to match state keys
            handler: Callback for state updates
        """
        self._state_handlers[key_prefix] = handler

    async def start(self) -> None:
        """Start gossip protocol."""
        self._running = True
        asyncio.create_task(self._gossip_loop())
        logger.info("Gossip protocol started")

    async def stop(self) -> None:
        """Stop gossip protocol."""
        self._running = False
        logger.info("Gossip protocol stopped")

    async def update_state(
        self,
        key: str,
        value: Any,
        immediate_broadcast: bool = False,
    ) -> None:
        """
        Update local state and propagate via gossip.

        Args:
            key: State key
            value: State value
            immediate_broadcast: If True, broadcast immediately
        """
        async with self._lock:
            # Increment vector clock
            self._vector_clock[self.node_id] = self._vector_clock.get(self.node_id, 0) + 1

            # Get current version for this key
            current = self._states.get(key)
            version = (current.version + 1) if current else 1

            state = GossipState(
                key=key,
                value=value,
                version=version,
                origin_node=self.node_id,
            )

            self._states[key] = state
            self._seen_state_ids.add(state.state_id)

        if immediate_broadcast:
            await self._gossip_round()

    async def get_state(self, key: str) -> Optional[GossipState]:
        """Get current state for key."""
        async with self._lock:
            return self._states.get(key)

    async def get_all_states(self, prefix: Optional[str] = None) -> Dict[str, GossipState]:
        """Get all states, optionally filtered by prefix."""
        async with self._lock:
            if prefix:
                return {k: v for k, v in self._states.items() if k.startswith(prefix)}
            return dict(self._states)

    async def handle_gossip_message(self, message: GossipMessage) -> Optional[GossipMessage]:
        """
        Handle incoming gossip message.

        Args:
            message: Received gossip message

        Returns:
            Response message if pull requested
        """
        response_states = []

        async with self._lock:
            # Merge vector clocks
            for node_id, clock in message.vector_clock.items():
                self._vector_clock[node_id] = max(
                    self._vector_clock.get(node_id, 0),
                    clock
                )

            # Process received states
            for state in message.states:
                if state.state_id in self._seen_state_ids:
                    continue  # Already seen

                # Check if this is newer
                current = self._states.get(state.key)
                if current is None or state.version > current.version:
                    # Accept new state
                    self._states[state.key] = state
                    self._seen_state_ids.add(state.state_id)

                    # Trigger handlers
                    self._trigger_handlers(state)

                    logger.debug(f"Accepted gossip state: {state.key} v{state.version}")

            # Handle pull request
            if message.message_type in ("pull", "push_pull"):
                # Send states that sender might not have
                for key, state in self._states.items():
                    sender_clock = message.vector_clock.get(state.origin_node, 0)
                    if state.version > sender_clock:
                        response_states.append(state)

        if response_states and message.message_type in ("pull", "push_pull"):
            return GossipMessage(
                message_type="push",
                sender_id=self.node_id,
                states=response_states,
                vector_clock=dict(self._vector_clock),
            )

        return None

    def _trigger_handlers(self, state: GossipState) -> None:
        """Trigger registered handlers for state update."""
        for prefix, handler in self._state_handlers.items():
            if state.key.startswith(prefix):
                try:
                    handler(state)
                except Exception as e:
                    logger.error(f"Error in gossip handler for {prefix}: {e}")

    async def _gossip_loop(self) -> None:
        """Background loop for periodic gossip rounds."""
        while self._running:
            try:
                await self._gossip_round()
                await asyncio.sleep(self.gossip_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(1)

    async def _gossip_round(self) -> None:
        """Execute one gossip round."""
        if not self._peers or not self._send_func:
            return

        # Select random peers
        selected_peers = random.sample(
            self._peers,
            min(self.fanout, len(self._peers))
        )

        # Get states to send
        async with self._lock:
            states_to_send = list(self._states.values())

        if not states_to_send:
            return

        # Create gossip message
        message = GossipMessage(
            message_type="push_pull",
            sender_id=self.node_id,
            states=states_to_send,
            vector_clock=dict(self._vector_clock),
        )

        # Send to selected peers
        for peer in selected_peers:
            try:
                await self._send_func(
                    Message(
                        message_type=MessageType.PEER_ANNOUNCE,  # Using existing type
                        sender_id=self.node_id,
                        payload=self._serialize_gossip_message(message),
                    ),
                    peer.peer_id,
                )
            except Exception as e:
                logger.debug(f"Failed to gossip to {peer.peer_id}: {e}")

    def _serialize_gossip_message(self, message: GossipMessage) -> Dict[str, Any]:
        """Serialize gossip message to dict."""
        return {
            "gossip_type": message.message_type,
            "sender": message.sender_id,
            "states": [
                {
                    "key": s.key,
                    "value": s.value,
                    "version": s.version,
                    "timestamp": s.timestamp,
                    "origin": s.origin_node,
                    "ttl": s.ttl,
                }
                for s in message.states
            ],
            "vector_clock": message.vector_clock,
            "ts": message.timestamp,
        }

    def _deserialize_gossip_message(self, data: Dict[str, Any]) -> GossipMessage:
        """Deserialize gossip message from dict."""
        return GossipMessage(
            message_type=data["gossip_type"],
            sender_id=data["sender"],
            states=[
                GossipState(
                    key=s["key"],
                    value=s["value"],
                    version=s["version"],
                    timestamp=s["timestamp"],
                    origin_node=s["origin"],
                    ttl=s["ttl"],
                )
                for s in data["states"]
            ],
            vector_clock=data["vector_clock"],
            timestamp=data["ts"],
        )

    async def cleanup_expired(self) -> int:
        """Remove expired state entries."""
        now = time.time()
        removed = 0

        async with self._lock:
            expired_keys = [
                key for key, state in self._states.items()
                if (now - state.timestamp) > self.state_ttl
            ]

            for key in expired_keys:
                del self._states[key]
                removed += 1

        if removed:
            logger.debug(f"Cleaned up {removed} expired gossip states")

        return removed


class PolicyVersionGossip:
    """
    Specialized gossip for policy version synchronization.

    Ensures all nodes converge on the latest policy version
    for on-policy training consistency.
    """

    def __init__(self, gossip: GossipProtocol):
        """
        Initialize policy version gossip.

        Args:
            gossip: Base gossip protocol instance
        """
        self.gossip = gossip
        self._current_version = 0
        self._version_callbacks: List[Callable[[int], None]] = []

        # Register handler
        gossip.register_handler("policy:version", self._handle_version_update)

    def _handle_version_update(self, state: GossipState) -> None:
        """Handle policy version update."""
        new_version = state.value
        if new_version > self._current_version:
            self._current_version = new_version
            for callback in self._version_callbacks:
                try:
                    callback(new_version)
                except Exception as e:
                    logger.error(f"Error in version callback: {e}")

    def on_version_update(self, callback: Callable[[int], None]) -> None:
        """Register callback for version updates."""
        self._version_callbacks.append(callback)

    async def announce_version(self, version: int) -> None:
        """Announce new policy version to network."""
        await self.gossip.update_state(
            key="policy:version",
            value=version,
            immediate_broadcast=True,
        )
        self._current_version = max(self._current_version, version)

    @property
    def current_version(self) -> int:
        """Get current known policy version."""
        return self._current_version


class MetricsAggregator:
    """
    Aggregates training metrics across nodes via gossip.
    """

    def __init__(self, gossip: GossipProtocol):
        """
        Initialize metrics aggregator.

        Args:
            gossip: Base gossip protocol instance
        """
        self.gossip = gossip
        self._metrics: Dict[str, Dict[str, float]] = {}

        gossip.register_handler("metrics:", self._handle_metrics_update)

    def _handle_metrics_update(self, state: GossipState) -> None:
        """Handle metrics update from peer."""
        node_id = state.origin_node
        metrics = state.value
        self._metrics[node_id] = metrics

    async def report_metrics(self, metrics: Dict[str, float]) -> None:
        """Report local metrics to network."""
        await self.gossip.update_state(
            key=f"metrics:{self.gossip.node_id}",
            value=metrics,
        )

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from all nodes."""
        if not self._metrics:
            return {}

        # Aggregate by averaging
        aggregated = {}
        all_keys = set()
        for node_metrics in self._metrics.values():
            all_keys.update(node_metrics.keys())

        for key in all_keys:
            values = [
                m[key] for m in self._metrics.values()
                if key in m
            ]
            if values:
                aggregated[key] = sum(values) / len(values)
                aggregated[f"{key}_sum"] = sum(values)
                aggregated[f"{key}_count"] = len(values)

        return aggregated
