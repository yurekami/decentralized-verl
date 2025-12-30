"""Pytest configuration and fixtures for decentralized veRL tests."""

import asyncio
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import torch
import torch.nn as nn


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    from decentralized_verl.core.config import (
        DecentralizedConfig,
        NetworkConfig,
        ModelConfig,
        TrainingConfig,
        AlgorithmConfig,
    )

    return DecentralizedConfig(
        network=NetworkConfig(
            dht_port=31337,
            initial_peers=[],
            heartbeat_interval=1.0,
        ),
        model=ModelConfig(
            model_name_or_path="gpt2",
            num_layers=12,
            hidden_size=768,
        ),
        training=TrainingConfig(
            rollout_batch_size=4,
            train_batch_size=2,
            ppo_epochs=2,
        ),
        algorithm=AlgorithmConfig(),
    )


@pytest.fixture
def mock_dht_manager():
    """Create a mock DHT manager."""
    manager = MagicMock()
    manager.node_id = "test_node"
    manager.dht = MagicMock()
    manager._peer_cache = {}
    manager._message_queue = MagicMock()

    # Mock async methods
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.announce_peer = AsyncMock(return_value=True)
    manager.discover_peers = AsyncMock(return_value=[])
    manager.send_to_peer = AsyncMock(return_value=True)
    manager.broadcast = AsyncMock(return_value=1)
    manager.receive_message = AsyncMock(return_value=None)
    manager.store_gradient = AsyncMock(return_value=True)
    manager.get_gradients_for_step = AsyncMock(return_value=[])
    manager.update_policy_version = AsyncMock(return_value=True)
    manager.get_latest_policy_version = AsyncMock(return_value=0)
    manager.get_multiaddr = MagicMock(return_value="/ip4/127.0.0.1/tcp/31337/p2p/test")
    manager.get_peer_id = MagicMock(return_value="test_peer_id")

    return manager


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 1000)

        def forward(self, input_ids, attention_mask=None):
            x = self.embed(input_ids)
            logits = self.linear(x)
            return MagicMock(logits=logits)

        def generate(self, input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = kwargs.get("max_new_tokens", 10)
            generated = torch.randint(0, 1000, (batch_size, input_ids.shape[1] + seq_len))
            return MagicMock(
                sequences=generated,
                scores=[torch.randn(batch_size, 1000) for _ in range(seq_len)],
            )

    return SimpleModel()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def encode_side_effect(text, **kwargs):
        return {"input_ids": torch.randint(0, 1000, (1, 20)), "attention_mask": torch.ones(1, 20)}

    tokenizer.side_effect = encode_side_effect
    tokenizer.__call__ = MagicMock(side_effect=encode_side_effect)
    tokenizer.encode = MagicMock(return_value=list(range(20)))
    tokenizer.decode = MagicMock(return_value="test response")

    return tokenizer


@pytest.fixture
def sample_experiences():
    """Create sample experiences for testing."""
    from decentralized_verl.training.experience import Experience

    return [
        Experience(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            reward=0.8,
            log_probs=[-0.5, -0.3, -0.4],
            ref_log_probs=[-0.6, -0.4, -0.5],
            policy_version=1,
        ),
        Experience(
            prompt="Explain ML.",
            response="ML is machine learning.",
            reward=0.7,
            log_probs=[-0.4, -0.2, -0.3],
            ref_log_probs=[-0.5, -0.3, -0.4],
            policy_version=1,
        ),
    ]


@pytest.fixture
def sample_peer_info():
    """Create sample peer info."""
    from decentralized_verl.core.protocol import PeerInfo

    return PeerInfo(
        peer_id="peer_123",
        multiaddr="/ip4/192.168.1.100/tcp/31337/p2p/QmTest",
        role="hybrid",
        blocks_served=[0, 1, 2, 3],
        capabilities={"gpu_count": 1, "model_name": "llama"},
    )
