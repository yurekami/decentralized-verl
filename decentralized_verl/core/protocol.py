"""Protocol definitions for P2P communication in decentralized veRL."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import time
import hashlib
import msgpack
import struct
from abc import ABC, abstractmethod


class MessageType(Enum):
    """Types of messages in the P2P protocol."""

    # Discovery
    PEER_ANNOUNCE = auto()
    PEER_DISCOVERY = auto()
    PEER_LIST = auto()

    # Training coordination
    ROLLOUT_REQUEST = auto()
    ROLLOUT_RESPONSE = auto()
    GRADIENT_UPDATE = auto()
    GRADIENT_ACK = auto()

    # Model synchronization
    POLICY_SYNC = auto()
    CHECKPOINT_REQUEST = auto()
    CHECKPOINT_RESPONSE = auto()
    CHECKPOINT_SYNC = auto()

    # Health and status
    HEARTBEAT = auto()
    STATUS_REQUEST = auto()
    STATUS_RESPONSE = auto()

    # Training control
    EPOCH_START = auto()
    EPOCH_END = auto()
    TRAINING_COMPLETE = auto()

    # Error handling
    ERROR = auto()
    RECOVERY_REQUEST = auto()
    RECOVERY_RESPONSE = auto()


@dataclass
class Message:
    """Base message class for P2P communication."""

    message_type: MessageType
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate message ID after initialization."""
        self._message_id: Optional[str] = None

    @property
    def message_id(self) -> str:
        """Generate unique message ID."""
        if self._message_id is None:
            content = f"{self.message_type.name}:{self.sender_id}:{self.timestamp}:{self.sequence_num}"
            self._message_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._message_id

    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        data = {
            "type": self.message_type.value,
            "sender": self.sender_id,
            "ts": self.timestamp,
            "seq": self.sequence_num,
            "payload": self.payload,
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        unpacked = msgpack.unpackb(data, raw=False)
        return cls(
            message_type=MessageType(unpacked["type"]),
            sender_id=unpacked["sender"],
            timestamp=unpacked["ts"],
            sequence_num=unpacked["seq"],
            payload=unpacked["payload"],
        )


@dataclass
class RolloutRequest:
    """Request for generating rollouts."""

    prompts: List[str]
    prompt_ids: List[str]
    policy_version: int
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.sha256(
                f"{self.prompts[0]}:{time.time()}".encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for message payload."""
        return {
            "prompts": self.prompts,
            "prompt_ids": self.prompt_ids,
            "policy_version": self.policy_version,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolloutRequest":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RolloutResponse:
    """Response containing generated rollouts."""

    request_id: str
    responses: List[str]
    response_ids: List[str]
    log_probs: List[List[float]]
    policy_version: int
    generation_time: float = 0.0
    worker_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for message payload."""
        return {
            "request_id": self.request_id,
            "responses": self.responses,
            "response_ids": self.response_ids,
            "log_probs": self.log_probs,
            "policy_version": self.policy_version,
            "generation_time": self.generation_time,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolloutResponse":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GradientUpdate:
    """Gradient update for distributed training."""

    step: int
    policy_version: int
    gradients: Dict[str, Any]  # Compressed gradient tensors
    metrics: Dict[str, float] = field(default_factory=dict)
    worker_id: str = ""
    batch_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for message payload."""
        return {
            "step": self.step,
            "policy_version": self.policy_version,
            "gradients": self.gradients,
            "metrics": self.metrics,
            "worker_id": self.worker_id,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientUpdate":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointSync:
    """Checkpoint synchronization message."""

    checkpoint_id: str
    step: int
    policy_version: int
    chunk_index: int
    total_chunks: int
    chunk_data: bytes
    model_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for message payload."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "policy_version": self.policy_version,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_data": self.chunk_data,
            "model_hash": self.model_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointSync":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PeerInfo:
    """Information about a peer in the network."""

    peer_id: str
    multiaddr: str
    role: str  # NodeRole as string
    blocks_served: List[int]
    capabilities: Dict[str, Any]
    last_seen: float = field(default_factory=time.time)
    status: str = "active"
    load: float = 0.0  # Current load percentage

    def is_alive(self, timeout: float = 60.0) -> bool:
        """Check if peer is still alive based on last seen time."""
        return (time.time() - self.last_seen) < timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peer_id": self.peer_id,
            "multiaddr": self.multiaddr,
            "role": self.role,
            "blocks_served": self.blocks_served,
            "capabilities": self.capabilities,
            "last_seen": self.last_seen,
            "status": self.status,
            "load": self.load,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        """Create from dictionary."""
        return cls(**data)


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return response."""
        pass


class ProtocolError(Exception):
    """Exception raised for protocol errors."""

    def __init__(self, message: str, error_code: int = 0):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


# Protocol version for compatibility checking
PROTOCOL_VERSION = "1.0.0"

# Magic bytes for message framing
MAGIC_BYTES = b"DVERL"

# Maximum message size (100 MB)
MAX_MESSAGE_SIZE = 100 * 1024 * 1024


def frame_message(data: bytes) -> bytes:
    """Frame a message with magic bytes and length prefix."""
    length = len(data)
    if length > MAX_MESSAGE_SIZE:
        raise ProtocolError(f"Message too large: {length} > {MAX_MESSAGE_SIZE}")
    return MAGIC_BYTES + struct.pack(">I", length) + data


def unframe_message(data: bytes) -> Tuple[bytes, bytes]:
    """Unframe a message and return (message, remaining)."""
    if len(data) < len(MAGIC_BYTES) + 4:
        raise ProtocolError("Incomplete frame header")

    if not data.startswith(MAGIC_BYTES):
        raise ProtocolError("Invalid magic bytes")

    length = struct.unpack(">I", data[len(MAGIC_BYTES):len(MAGIC_BYTES) + 4])[0]
    header_size = len(MAGIC_BYTES) + 4

    if len(data) < header_size + length:
        raise ProtocolError("Incomplete message")

    message = data[header_size:header_size + length]
    remaining = data[header_size + length:]

    return message, remaining
