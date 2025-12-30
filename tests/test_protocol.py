"""Tests for P2P protocol definitions."""

import pytest
import time

from decentralized_verl.core.protocol import (
    Message,
    MessageType,
    PeerInfo,
    RolloutRequest,
    RolloutResponse,
    GradientUpdate,
    CheckpointSync,
    frame_message,
    unframe_message,
    ProtocolError,
    MAGIC_BYTES,
    MAX_MESSAGE_SIZE,
)


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            message_type=MessageType.HEARTBEAT,
            sender_id="node_123",
            payload={"status": "alive"},
        )
        assert msg.message_type == MessageType.HEARTBEAT
        assert msg.sender_id == "node_123"
        assert msg.payload["status"] == "alive"

    def test_message_id_generation(self):
        """Test message ID is generated consistently."""
        msg = Message(
            message_type=MessageType.HEARTBEAT,
            sender_id="node_123",
            timestamp=12345.0,
            sequence_num=1,
        )
        id1 = msg.message_id
        id2 = msg.message_id
        assert id1 == id2
        assert len(id1) == 16

    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = Message(
            message_type=MessageType.ROLLOUT_REQUEST,
            sender_id="node_abc",
            payload={"prompts": ["test1", "test2"]},
        )

        serialized = original.serialize()
        assert isinstance(serialized, bytes)

        restored = Message.deserialize(serialized)
        assert restored.message_type == original.message_type
        assert restored.sender_id == original.sender_id
        assert restored.payload == original.payload


class TestPeerInfo:
    """Tests for PeerInfo class."""

    def test_peer_info_creation(self):
        """Test creating peer info."""
        peer = PeerInfo(
            peer_id="peer_123",
            multiaddr="/ip4/192.168.1.1/tcp/31337/p2p/QmTest",
            role="actor",
            blocks_served=[0, 1, 2],
            capabilities={"gpu_count": 2},
        )
        assert peer.peer_id == "peer_123"
        assert peer.role == "actor"
        assert len(peer.blocks_served) == 3

    def test_is_alive(self):
        """Test peer liveness check."""
        peer = PeerInfo(
            peer_id="peer_123",
            multiaddr="/ip4/192.168.1.1/tcp/31337/p2p/QmTest",
            role="actor",
            blocks_served=[],
            capabilities={},
            last_seen=time.time(),
        )
        assert peer.is_alive(timeout=60.0) is True

        # Set old timestamp
        peer.last_seen = time.time() - 120
        assert peer.is_alive(timeout=60.0) is False

    def test_to_dict_from_dict(self):
        """Test conversion to and from dict."""
        original = PeerInfo(
            peer_id="peer_456",
            multiaddr="/ip4/10.0.0.1/tcp/31337/p2p/QmTest2",
            role="critic",
            blocks_served=[4, 5, 6, 7],
            capabilities={"model": "llama"},
        )

        as_dict = original.to_dict()
        restored = PeerInfo.from_dict(as_dict)

        assert restored.peer_id == original.peer_id
        assert restored.role == original.role
        assert restored.blocks_served == original.blocks_served


class TestRolloutRequest:
    """Tests for RolloutRequest class."""

    def test_request_creation(self):
        """Test creating a rollout request."""
        request = RolloutRequest(
            prompts=["What is AI?", "Explain ML"],
            prompt_ids=["p1", "p2"],
            policy_version=5,
            max_new_tokens=100,
        )
        assert len(request.prompts) == 2
        assert request.policy_version == 5
        assert request.max_new_tokens == 100
        assert request.request_id != ""

    def test_to_dict_from_dict(self):
        """Test conversion."""
        original = RolloutRequest(
            prompts=["test prompt"],
            prompt_ids=["id1"],
            policy_version=10,
        )

        as_dict = original.to_dict()
        restored = RolloutRequest.from_dict(as_dict)

        assert restored.prompts == original.prompts
        assert restored.policy_version == original.policy_version


class TestRolloutResponse:
    """Tests for RolloutResponse class."""

    def test_response_creation(self):
        """Test creating a rollout response."""
        response = RolloutResponse(
            request_id="req_123",
            responses=["AI is...", "ML is..."],
            response_ids=["r1", "r2"],
            log_probs=[[-0.5, -0.3], [-0.4, -0.2]],
            policy_version=5,
            generation_time=1.5,
        )
        assert response.request_id == "req_123"
        assert len(response.responses) == 2
        assert response.generation_time == 1.5


class TestGradientUpdate:
    """Tests for GradientUpdate class."""

    def test_gradient_update_creation(self):
        """Test creating a gradient update."""
        update = GradientUpdate(
            step=100,
            policy_version=5,
            gradients={"layer1": b"gradient_data"},
            metrics={"loss": 0.5},
            batch_size=32,
        )
        assert update.step == 100
        assert update.batch_size == 32
        assert "loss" in update.metrics


class TestMessageFraming:
    """Tests for message framing functions."""

    def test_frame_unframe(self):
        """Test framing and unframing messages."""
        data = b"test message data"
        framed = frame_message(data)

        assert framed.startswith(MAGIC_BYTES)

        unframed, remaining = unframe_message(framed)
        assert unframed == data
        assert remaining == b""

    def test_frame_with_remaining_data(self):
        """Test unframing with remaining data."""
        data1 = b"first message"
        data2 = b"second message"

        framed1 = frame_message(data1)
        framed2 = frame_message(data2)

        combined = framed1 + framed2

        msg1, remaining = unframe_message(combined)
        assert msg1 == data1

        msg2, remaining = unframe_message(remaining)
        assert msg2 == data2
        assert remaining == b""

    def test_frame_too_large(self):
        """Test framing rejects oversized messages."""
        large_data = b"x" * (MAX_MESSAGE_SIZE + 1)
        with pytest.raises(ProtocolError):
            frame_message(large_data)

    def test_unframe_invalid_magic(self):
        """Test unframing rejects invalid magic bytes."""
        invalid = b"INVALID_DATA"
        with pytest.raises(ProtocolError):
            unframe_message(invalid)

    def test_unframe_incomplete_header(self):
        """Test unframing handles incomplete header."""
        incomplete = MAGIC_BYTES + b"\x00"  # Missing length bytes
        with pytest.raises(ProtocolError):
            unframe_message(incomplete)
