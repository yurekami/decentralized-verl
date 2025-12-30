"""Tests for configuration classes."""

import pytest
import json
import tempfile
from pathlib import Path

from decentralized_verl.core.config import (
    DecentralizedConfig,
    NetworkConfig,
    ModelConfig,
    TrainingConfig,
    AlgorithmConfig,
    NodeRole,
    Algorithm,
    GenerationBackend,
)


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NetworkConfig()
        assert config.dht_port == 31337
        assert config.max_peers == 100
        assert config.heartbeat_interval == 10.0

    def test_get_listen_addr(self):
        """Test listen address generation."""
        config = NetworkConfig(dht_port=12345)
        assert config.get_listen_addr() == "/ip4/0.0.0.0/tcp/12345"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NetworkConfig(
            dht_port=31338,
            initial_peers=["/ip4/1.2.3.4/tcp/31337/p2p/QmTest"],
            max_peers=50,
        )
        assert config.dht_port == 31338
        assert len(config.initial_peers) == 1
        assert config.max_peers == 50


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert "llama" in config.model_name_or_path.lower()
        assert config.dtype == "bfloat16"

    def test_num_blocks_calculation(self):
        """Test number of blocks calculation."""
        config = ModelConfig(num_layers=32, layers_per_block=4)
        assert config.num_blocks == 8

        config = ModelConfig(num_layers=30, layers_per_block=4)
        assert config.num_blocks == 8  # Ceiling division

    def test_lora_settings(self):
        """Test LoRA configuration."""
        config = ModelConfig(
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
        )
        assert config.use_lora is True
        assert config.lora_rank == 32
        assert config.lora_alpha == 64


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_algorithm(self):
        """Test default algorithm is PPO."""
        config = TrainingConfig()
        assert config.algorithm == Algorithm.PPO

    def test_ppo_hyperparameters(self):
        """Test PPO hyperparameters."""
        config = TrainingConfig(
            clip_range=0.1,
            ppo_epochs=8,
            entropy_coef=0.02,
        )
        assert config.clip_range == 0.1
        assert config.ppo_epochs == 8
        assert config.entropy_coef == 0.02


class TestDecentralizedConfig:
    """Tests for main DecentralizedConfig."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "network": {"dht_port": 12345},
            "model": {"model_name_or_path": "gpt2"},
            "node_role": "actor",
        }
        config = DecentralizedConfig.from_dict(config_dict)
        assert config.network.dht_port == 12345
        assert config.model.model_name_or_path == "gpt2"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = DecentralizedConfig()
        config_dict = config.to_dict()
        assert "network" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict

    def test_save_and_load(self):
        """Test saving and loading config."""
        config = DecentralizedConfig(
            network=NetworkConfig(dht_port=54321),
            model=ModelConfig(model_name_or_path="test-model"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            loaded = DecentralizedConfig.load(path)
            assert loaded.network.dht_port == 54321
            assert loaded.model.model_name_or_path == "test-model"

    def test_node_roles(self):
        """Test different node roles."""
        for role in NodeRole:
            config = DecentralizedConfig(node_role=role)
            assert config.node_role == role

    def test_generation_backends(self):
        """Test different generation backends."""
        for backend in GenerationBackend:
            config = DecentralizedConfig(generation_backend=backend)
            assert config.generation_backend == backend
