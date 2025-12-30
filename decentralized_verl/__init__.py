"""
Decentralized veRL - P2P Reinforcement Learning for LLMs

A decentralized implementation of RLHF training that enables peer-to-peer
training across heterogeneous nodes without central coordination.

Key Features:
- DHT-based peer discovery using Hivemind
- Distributed actor-critic training across heterogeneous GPUs
- Fault-tolerant training with automatic recovery
- Support for PPO, GRPO, and other RLHF algorithms
- Compatible with vLLM, SGLang, and HuggingFace backends
"""

__version__ = "0.1.0"
__author__ = "Decentralized veRL Contributors"

from decentralized_verl.core.config import (
    DecentralizedConfig,
    NetworkConfig,
    TrainingConfig,
    ModelConfig,
)
from decentralized_verl.core.node import DecentralizedNode
from decentralized_verl.network.dht_manager import DHTManager
from decentralized_verl.training.coordinator import DecentralizedCoordinator
from decentralized_verl.training.trainer import DecentralizedTrainer

__all__ = [
    "__version__",
    "DecentralizedConfig",
    "NetworkConfig",
    "TrainingConfig",
    "ModelConfig",
    "DecentralizedNode",
    "DHTManager",
    "DecentralizedCoordinator",
    "DecentralizedTrainer",
]
