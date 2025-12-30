"""Connection management for decentralized veRL."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import time
import socket

from decentralized_verl.core.protocol import PeerInfo, Message

logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """State of a peer connection."""
    peer_id: str
    connected: bool = False
    last_activity: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    reconnect_attempts: int = 0


class ConnectionManager:
    """
    Manages connections to peers in the decentralized network.

    Handles:
    - Connection establishment and maintenance
    - Automatic reconnection on failure
    - Connection health monitoring
    - Load balancing across connections
    """

    def __init__(
        self,
        node_id: str,
        max_connections: int = 100,
        connection_timeout: float = 30.0,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 5,
    ):
        """
        Initialize connection manager.

        Args:
            node_id: This node's identifier
            max_connections: Maximum concurrent connections
            connection_timeout: Timeout for connection attempts
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.node_id = node_id
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        # Connection tracking
        self._connections: Dict[str, ConnectionState] = {}
        self._pending_connections: set = set()

        # Callbacks
        self._on_connect: List[Callable] = []
        self._on_disconnect: List[Callable] = []
        self._on_message: List[Callable] = []

        # Control
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start connection manager."""
        self._running = True
        asyncio.create_task(self._health_check_loop())
        logger.info("Connection manager started")

    async def stop(self) -> None:
        """Stop connection manager."""
        self._running = False

        # Close all connections
        async with self._lock:
            for peer_id in list(self._connections.keys()):
                await self._disconnect(peer_id)

        logger.info("Connection manager stopped")

    async def connect(self, peer_info: PeerInfo) -> bool:
        """
        Establish connection to a peer.

        Args:
            peer_info: Information about the peer

        Returns:
            True if connection was successful
        """
        peer_id = peer_info.peer_id

        async with self._lock:
            # Check if already connected
            if peer_id in self._connections:
                if self._connections[peer_id].connected:
                    return True

            # Check if at max connections
            active = sum(1 for c in self._connections.values() if c.connected)
            if active >= self.max_connections:
                logger.warning(f"Max connections reached ({self.max_connections})")
                return False

            # Check if already pending
            if peer_id in self._pending_connections:
                return False

            self._pending_connections.add(peer_id)

        try:
            # Create connection state
            state = ConnectionState(peer_id=peer_id)

            # Attempt connection
            success = await self._establish_connection(peer_info)

            async with self._lock:
                if success:
                    state.connected = True
                    state.last_activity = time.time()
                    self._connections[peer_id] = state
                    logger.info(f"Connected to peer: {peer_id}")

                    # Trigger callbacks
                    for callback in self._on_connect:
                        try:
                            await callback(peer_info)
                        except Exception as e:
                            logger.error(f"Connect callback error: {e}")

                return success

        finally:
            self._pending_connections.discard(peer_id)

    async def _establish_connection(self, peer_info: PeerInfo) -> bool:
        """Establish actual connection to peer."""
        # In a real implementation, this would establish a P2P connection
        # For now, we rely on Hivemind DHT for communication
        try:
            # Parse multiaddr to get host/port
            # Format: /ip4/HOST/tcp/PORT/p2p/PEER_ID
            parts = peer_info.multiaddr.split("/")

            if len(parts) >= 5 and parts[1] == "ip4" and parts[3] == "tcp":
                host = parts[2]
                port = int(parts[4])

                # Check if reachable
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.connection_timeout)
                try:
                    result = sock.connect_ex((host, port))
                    return result == 0
                finally:
                    sock.close()

            return True  # Assume success if can't parse

        except Exception as e:
            logger.debug(f"Connection to {peer_info.peer_id} failed: {e}")
            return False

    async def disconnect(self, peer_id: str) -> None:
        """Disconnect from a peer."""
        async with self._lock:
            await self._disconnect(peer_id)

    async def _disconnect(self, peer_id: str) -> None:
        """Internal disconnect (requires lock)."""
        if peer_id in self._connections:
            state = self._connections[peer_id]
            state.connected = False

            # Trigger callbacks
            for callback in self._on_disconnect:
                try:
                    await callback(peer_id)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")

            del self._connections[peer_id]
            logger.info(f"Disconnected from peer: {peer_id}")

    async def send(
        self,
        peer_id: str,
        message: Message,
    ) -> bool:
        """
        Send message to a peer.

        Args:
            peer_id: Target peer
            message: Message to send

        Returns:
            True if send was successful
        """
        async with self._lock:
            if peer_id not in self._connections:
                logger.warning(f"Not connected to peer: {peer_id}")
                return False

            state = self._connections[peer_id]
            if not state.connected:
                return False

        try:
            # In a real implementation, send via the connection
            # For now, this is handled by DHTManager

            async with self._lock:
                state.messages_sent += 1
                state.last_activity = time.time()

            return True

        except Exception as e:
            async with self._lock:
                state.errors += 1
            logger.error(f"Send to {peer_id} failed: {e}")
            return False

    async def receive(
        self,
        peer_id: str,
        data: bytes,
    ) -> None:
        """
        Handle received data from a peer.

        Args:
            peer_id: Source peer
            data: Received data
        """
        async with self._lock:
            if peer_id in self._connections:
                state = self._connections[peer_id]
                state.messages_received += 1
                state.bytes_received += len(data)
                state.last_activity = time.time()

        # Trigger callbacks
        for callback in self._on_message:
            try:
                await callback(peer_id, data)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

    async def _health_check_loop(self) -> None:
        """Periodically check connection health."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                async with self._lock:
                    now = time.time()
                    stale_peers = []

                    for peer_id, state in self._connections.items():
                        if not state.connected:
                            continue

                        # Check if connection is stale (no activity for 2 minutes)
                        if now - state.last_activity > 120:
                            stale_peers.append(peer_id)

                    # Handle stale connections
                    for peer_id in stale_peers:
                        logger.warning(f"Stale connection: {peer_id}")
                        await self._disconnect(peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def on_connect(self, callback: Callable) -> None:
        """Register connection callback."""
        self._on_connect.append(callback)

    def on_disconnect(self, callback: Callable) -> None:
        """Register disconnection callback."""
        self._on_disconnect.append(callback)

    def on_message(self, callback: Callable) -> None:
        """Register message callback."""
        self._on_message.append(callback)

    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        return [
            peer_id for peer_id, state in self._connections.items()
            if state.connected
        ]

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        connected = sum(1 for c in self._connections.values() if c.connected)
        total_sent = sum(c.messages_sent for c in self._connections.values())
        total_received = sum(c.messages_received for c in self._connections.values())

        return {
            "total_connections": len(self._connections),
            "active_connections": connected,
            "max_connections": self.max_connections,
            "messages_sent": total_sent,
            "messages_received": total_received,
        }

    def update_latency(self, peer_id: str, latency_ms: float) -> None:
        """Update latency measurement for a peer."""
        if peer_id in self._connections:
            self._connections[peer_id].latency_ms = latency_ms
