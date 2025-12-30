"""DHT (Distributed Hash Table) manager using Hivemind."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
import time
import threading
from dataclasses import dataclass, field
from queue import Queue
import pickle

try:
    import hivemind
    from hivemind import DHT, get_dht_time
    from hivemind.dht.node import DHTNode
    from hivemind.utils.serializer import MSGPackSerializer
    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False
    hivemind = None
    DHT = None

from decentralized_verl.core.config import NetworkConfig
from decentralized_verl.core.protocol import (
    Message,
    MessageType,
    PeerInfo,
    frame_message,
    unframe_message,
)

logger = logging.getLogger(__name__)


# DHT key prefixes for different data types
DHT_PREFIX_PEERS = "dverl:peers:"
DHT_PREFIX_POLICY = "dverl:policy:"
DHT_PREFIX_CHECKPOINT = "dverl:checkpoint:"
DHT_PREFIX_GRADIENT = "dverl:gradient:"
DHT_PREFIX_MESSAGE = "dverl:message:"


@dataclass
class DHTValue:
    """Value stored in DHT with metadata."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # Time to live in seconds
    node_id: str = ""

    def is_expired(self) -> bool:
        """Check if value has expired."""
        return (time.time() - self.timestamp) > self.ttl


class DHTManager:
    """
    Manages DHT-based peer discovery and coordination.

    Uses Hivemind DHT for decentralized peer discovery and state sharing.
    Replaces Ray's centralized coordination with P2P mechanisms.
    """

    def __init__(
        self,
        config: NetworkConfig,
        node_id: str,
    ):
        """
        Initialize DHT manager.

        Args:
            config: Network configuration
            node_id: Unique identifier for this node
        """
        if not HIVEMIND_AVAILABLE:
            raise ImportError(
                "Hivemind is required for DHT functionality. "
                "Install with: pip install hivemind"
            )

        self.config = config
        self.node_id = node_id
        self.dht: Optional[DHT] = None
        self._running = False
        self._message_queue: Queue = Queue()
        self._message_handlers: Dict[str, Callable] = {}
        self._peer_cache: Dict[str, PeerInfo] = {}
        self._lock = threading.Lock()

    async def start(self) -> None:
        """Start DHT and connect to network."""
        logger.info(f"Starting DHT on port {self.config.dht_port}")

        # Parse initial peers
        initial_peers = self.config.initial_peers if self.config.initial_peers else None

        # Create DHT instance
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers,
            host_maddrs=[self.config.get_listen_addr()],
            announce_maddrs=self._get_announce_addrs(),
            wait_timeout=self.config.connection_timeout,
        )

        self._running = True

        # Start message receiving thread
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

        logger.info(f"DHT started. Peer ID: {self.dht.peer_id}")
        logger.info(f"Multiaddr: {self.get_multiaddr()}")

    async def stop(self) -> None:
        """Stop DHT gracefully."""
        logger.info("Stopping DHT...")
        self._running = False

        if self.dht:
            self.dht.shutdown()
            self.dht = None

        logger.info("DHT stopped")

    def _get_announce_addrs(self) -> Optional[List[str]]:
        """Get addresses to announce to network."""
        if self.config.announce_host:
            return [f"/ip4/{self.config.announce_host}/tcp/{self.config.dht_port}"]
        return None

    def get_multiaddr(self) -> str:
        """Get this node's multiaddr for peer connections."""
        if not self.dht:
            return ""
        return str(self.dht.get_visible_maddrs()[0])

    def get_peer_id(self) -> str:
        """Get this node's DHT peer ID."""
        if not self.dht:
            return ""
        return str(self.dht.peer_id)

    async def announce_peer(self, peer_info: PeerInfo) -> bool:
        """
        Announce this peer to the DHT network.

        Args:
            peer_info: Information about this peer

        Returns:
            True if announcement was successful
        """
        if not self.dht:
            return False

        key = f"{DHT_PREFIX_PEERS}{peer_info.peer_id}"
        value = DHTValue(
            data=peer_info.to_dict(),
            node_id=self.node_id,
            ttl=self.config.heartbeat_interval * 3,  # Expire after 3 missed heartbeats
        )

        try:
            success = self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )
            return success
        except Exception as e:
            logger.error(f"Failed to announce peer: {e}")
            return False

    async def discover_peers(self) -> List[PeerInfo]:
        """
        Discover peers in the network.

        Returns:
            List of discovered peer information
        """
        if not self.dht:
            return []

        peers = []

        # Get all peer keys from DHT
        try:
            # In a real implementation, we'd use DHT traversal
            # For now, we use cached peers and DHT queries
            with self._lock:
                peers = list(self._peer_cache.values())

            # Also query DHT for peers we know about
            for peer_id in list(self._peer_cache.keys()):
                key = f"{DHT_PREFIX_PEERS}{peer_id}"
                result = self.dht.get(key)
                if result and result.value:
                    try:
                        dht_value = pickle.loads(result.value)
                        if not dht_value.is_expired():
                            peer_info = PeerInfo.from_dict(dht_value.data)
                            with self._lock:
                                self._peer_cache[peer_id] = peer_info
                    except Exception as e:
                        logger.debug(f"Error parsing peer info: {e}")

        except Exception as e:
            logger.error(f"Error discovering peers: {e}")

        return peers

    async def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        # Check cache first
        with self._lock:
            if peer_id in self._peer_cache:
                return self._peer_cache[peer_id]

        if not self.dht:
            return None

        key = f"{DHT_PREFIX_PEERS}{peer_id}"
        try:
            result = self.dht.get(key)
            if result and result.value:
                dht_value = pickle.loads(result.value)
                if not dht_value.is_expired():
                    peer_info = PeerInfo.from_dict(dht_value.data)
                    with self._lock:
                        self._peer_cache[peer_id] = peer_info
                    return peer_info
        except Exception as e:
            logger.error(f"Error getting peer {peer_id}: {e}")

        return None

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all known peers.

        Args:
            message: Message to broadcast

        Returns:
            Number of peers message was sent to
        """
        if not self.dht:
            return 0

        sent_count = 0
        serialized = frame_message(message.serialize())

        # Store message in DHT for peers to retrieve
        key = f"{DHT_PREFIX_MESSAGE}{message.message_id}"
        value = DHTValue(
            data=serialized,
            node_id=self.node_id,
            ttl=60.0,  # Messages expire after 1 minute
        )

        try:
            self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )

            # Notify peers via DHT
            for peer_id in list(self._peer_cache.keys()):
                if peer_id != self.node_id:
                    await self.send_to_peer(message, peer_id)
                    sent_count += 1

        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

        return sent_count

    async def send_to_peer(self, message: Message, peer_id: str) -> bool:
        """
        Send message to specific peer.

        Args:
            message: Message to send
            peer_id: Target peer ID

        Returns:
            True if send was successful
        """
        if not self.dht:
            return False

        # Store message in peer's inbox
        key = f"{DHT_PREFIX_MESSAGE}{peer_id}:{message.message_id}"
        serialized = frame_message(message.serialize())
        value = DHTValue(
            data=serialized,
            node_id=self.node_id,
            ttl=60.0,
        )

        try:
            success = self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )
            return success
        except Exception as e:
            logger.error(f"Error sending to peer {peer_id}: {e}")
            return False

    async def receive_message(self, timeout: float = 1.0) -> Optional[Message]:
        """
        Receive next message from queue.

        Args:
            timeout: Maximum time to wait for message

        Returns:
            Received message or None if timeout
        """
        try:
            data = self._message_queue.get(timeout=timeout)
            return data
        except:
            return None

    def _receive_loop(self) -> None:
        """Background loop to receive messages from DHT."""
        while self._running:
            try:
                # Poll DHT for messages addressed to us
                if self.dht:
                    # Check for broadcast messages
                    self._poll_messages()
                time.sleep(0.1)
            except Exception as e:
                if self._running:
                    logger.error(f"Error in receive loop: {e}")
                time.sleep(1)

    def _poll_messages(self) -> None:
        """Poll DHT for new messages."""
        # In a production system, this would use Hivemind's pubsub or
        # direct RPC capabilities. Here we simulate with DHT polling.
        pass

    # Gradient averaging methods for distributed training

    async def store_gradient(
        self,
        step: int,
        gradients: Dict[str, Any],
        batch_size: int,
    ) -> bool:
        """
        Store gradients in DHT for averaging.

        Args:
            step: Training step
            gradients: Gradient dictionary
            batch_size: Batch size used for these gradients

        Returns:
            True if storage was successful
        """
        if not self.dht:
            return False

        key = f"{DHT_PREFIX_GRADIENT}{step}:{self.node_id}"
        value = DHTValue(
            data={
                "gradients": gradients,
                "batch_size": batch_size,
                "step": step,
            },
            node_id=self.node_id,
            ttl=120.0,  # Gradients valid for 2 minutes
        )

        try:
            success = self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )
            return success
        except Exception as e:
            logger.error(f"Error storing gradient: {e}")
            return False

    async def get_gradients_for_step(
        self,
        step: int,
        peer_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get gradients from multiple peers for a given step.

        Args:
            step: Training step
            peer_ids: List of peer IDs to fetch from

        Returns:
            List of gradient dictionaries
        """
        if not self.dht:
            return []

        gradients = []

        for peer_id in peer_ids:
            key = f"{DHT_PREFIX_GRADIENT}{step}:{peer_id}"
            try:
                result = self.dht.get(key)
                if result and result.value:
                    dht_value = pickle.loads(result.value)
                    if not dht_value.is_expired():
                        gradients.append(dht_value.data)
            except Exception as e:
                logger.debug(f"Error getting gradient from {peer_id}: {e}")

        return gradients

    # Checkpoint management

    async def store_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_data: bytes,
        chunk_index: int,
        total_chunks: int,
    ) -> bool:
        """
        Store checkpoint chunk in DHT.

        Args:
            checkpoint_id: Unique checkpoint identifier
            checkpoint_data: Checkpoint data chunk
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            True if storage was successful
        """
        if not self.dht:
            return False

        key = f"{DHT_PREFIX_CHECKPOINT}{checkpoint_id}:chunk:{chunk_index}"
        value = DHTValue(
            data={
                "chunk_data": checkpoint_data,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            },
            node_id=self.node_id,
            ttl=3600.0,  # Checkpoints valid for 1 hour
        )

        try:
            success = self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )
            return success
        except Exception as e:
            logger.error(f"Error storing checkpoint chunk: {e}")
            return False

    async def get_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Optional[bytes]:
        """
        Retrieve checkpoint from DHT.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Assembled checkpoint data or None
        """
        if not self.dht:
            return None

        # First get metadata to know total chunks
        meta_key = f"{DHT_PREFIX_CHECKPOINT}{checkpoint_id}:meta"
        try:
            result = self.dht.get(meta_key)
            if not result or not result.value:
                return None

            meta = pickle.loads(result.value)
            total_chunks = meta.data.get("total_chunks", 0)

            # Retrieve all chunks
            chunks = {}
            for i in range(total_chunks):
                chunk_key = f"{DHT_PREFIX_CHECKPOINT}{checkpoint_id}:chunk:{i}"
                chunk_result = self.dht.get(chunk_key)
                if chunk_result and chunk_result.value:
                    chunk_data = pickle.loads(chunk_result.value)
                    chunks[i] = chunk_data.data["chunk_data"]

            # Assemble checkpoint
            if len(chunks) == total_chunks:
                return b"".join(chunks[i] for i in range(total_chunks))

        except Exception as e:
            logger.error(f"Error retrieving checkpoint: {e}")

        return None

    # Policy version tracking

    async def get_latest_policy_version(self) -> int:
        """Get the latest policy version from the network."""
        if not self.dht:
            return 0

        key = f"{DHT_PREFIX_POLICY}latest"
        try:
            result = self.dht.get(key)
            if result and result.value:
                dht_value = pickle.loads(result.value)
                return dht_value.data.get("version", 0)
        except Exception as e:
            logger.debug(f"Error getting policy version: {e}")

        return 0

    async def update_policy_version(self, version: int) -> bool:
        """Update the latest policy version in the network."""
        if not self.dht:
            return False

        key = f"{DHT_PREFIX_POLICY}latest"
        value = DHTValue(
            data={"version": version, "updated_by": self.node_id},
            node_id=self.node_id,
            ttl=3600.0,
        )

        try:
            success = self.dht.store(
                key=key,
                value=pickle.dumps(value),
                expiration_time=get_dht_time() + value.ttl,
            )
            return success
        except Exception as e:
            logger.error(f"Error updating policy version: {e}")
            return False
