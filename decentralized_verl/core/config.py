"""Configuration classes for decentralized veRL."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class NodeRole(str, Enum):
    """Role of a node in the decentralized network."""
    ACTOR = "actor"
    CRITIC = "critic"
    REWARD = "reward"
    REFERENCE = "reference"
    HYBRID = "hybrid"  # Can serve multiple roles


class Algorithm(str, Enum):
    """Supported RLHF algorithms."""
    PPO = "ppo"
    GRPO = "grpo"
    REINFORCE = "reinforce"
    RLOO = "rloo"
    REMAX = "remax"
    DAPO = "dapo"


class GenerationBackend(str, Enum):
    """Supported generation backends."""
    VLLM = "vllm"
    SGLANG = "sglang"
    HUGGINGFACE = "huggingface"


class TrainingBackend(str, Enum):
    """Supported training backends."""
    FSDP = "fsdp"
    FSDP2 = "fsdp2"
    DEEPSPEED = "deepspeed"
    HIVEMIND = "hivemind"


@dataclass
class NetworkConfig:
    """Configuration for P2P network layer."""

    # DHT settings
    initial_peers: List[str] = field(default_factory=list)
    dht_port: int = 31337
    announce_host: Optional[str] = None

    # P2P settings
    max_peers: int = 100
    connection_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    peer_discovery_interval: float = 60.0

    # Multiaddr prefix
    listen_on: str = "/ip4/0.0.0.0/tcp"

    # Gossip protocol
    gossip_fanout: int = 6
    gossip_interval: float = 1.0

    # Authentication (optional)
    auth_key: Optional[str] = None
    use_relay: bool = False

    def get_listen_addr(self) -> str:
        """Get the full listen address."""
        return f"{self.listen_on}/{self.dht_port}"


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""

    # Model identification
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    revision: Optional[str] = None
    trust_remote_code: bool = False

    # Model architecture
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    vocab_size: int = 32000

    # Partitioning
    layers_per_block: int = 4  # How many layers per partition

    # Precision
    dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LoRA settings
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks for partitioning."""
        return (self.num_layers + self.layers_per_block - 1) // self.layers_per_block


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    # Algorithm
    algorithm: Algorithm = Algorithm.PPO

    # PPO hyperparameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # GRPO/RLOO specific
    num_samples_per_prompt: int = 4

    # Learning rates
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    reference_free: bool = False

    # KL penalty
    kl_coef: float = 0.1
    target_kl: Optional[float] = 0.01
    kl_penalty: str = "kl"  # "kl" or "abs" or "mse"

    # Batch sizes
    rollout_batch_size: int = 512
    train_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Sequence settings
    max_prompt_length: int = 512
    max_response_length: int = 1024

    # Training duration
    total_episodes: int = 100000
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Mixed precision
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration."""

    # Reward settings
    reward_model_name: Optional[str] = None
    use_reward_function: bool = False  # For verifiable rewards
    reward_scale: float = 1.0
    reward_clip: Optional[float] = 10.0

    # GAE settings
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # Normalization
    normalize_rewards: bool = True
    normalize_advantages: bool = True

    # Experience buffer
    buffer_size: int = 10000
    min_buffer_size: int = 1000


@dataclass
class DecentralizedConfig:
    """Main configuration combining all settings."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    # Node settings
    node_role: NodeRole = NodeRole.HYBRID
    num_blocks_to_serve: int = 4  # How many transformer blocks this node serves
    block_indices: Optional[List[int]] = None  # Specific blocks to serve

    # Generation settings
    generation_backend: GenerationBackend = GenerationBackend.HUGGINGFACE
    training_backend: TrainingBackend = TrainingBackend.HIVEMIND

    # Distributed settings
    world_size: int = 1
    local_rank: int = 0

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Fault tolerance
    checkpoint_interval: int = 100
    max_recovery_attempts: int = 3

    # Logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DecentralizedConfig":
        """Create config from dictionary."""
        network = NetworkConfig(**config_dict.get("network", {}))
        model = ModelConfig(**config_dict.get("model", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        algorithm = AlgorithmConfig(**config_dict.get("algorithm", {}))

        # Remove nested configs from dict
        filtered_dict = {
            k: v for k, v in config_dict.items()
            if k not in ["network", "model", "training", "algorithm"]
        }

        return cls(
            network=network,
            model=model,
            training=training,
            algorithm=algorithm,
            **filtered_dict
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DecentralizedConfig":
        """Load config from JSON file."""
        import json
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
