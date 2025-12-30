# Decentralized veRL

**Decentralized Reinforcement Learning for LLMs - P2P RLHF Training**

A decentralized implementation of RLHF (Reinforcement Learning from Human Feedback) training that enables peer-to-peer training across heterogeneous nodes without central coordination. Inspired by [veRL](https://github.com/volcengine/verl) (ByteDance) and [BloomBee](https://github.com/yottalabsai/BloomBee) (YottaLabs).

## Features

- **Fully Decentralized**: No central coordinator - uses DHT-based peer discovery via Hivemind
- **Heterogeneous GPU Support**: Run training across consumer and datacenter GPUs
- **Fault Tolerant**: Automatic recovery and rebalancing when nodes fail
- **Multiple Algorithms**: PPO, GRPO, REINFORCE++, RLOO
- **Flexible Backends**: vLLM, SGLang, or HuggingFace Transformers
- **Distributed Model Serving**: Partition large models across multiple nodes

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Decentralized veRL Network                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐  │
│   │  Node 1  │◄───►│  Node 2  │◄───►│  Node 3  │◄───►│  Node N  │  │
│   │ (Actor)  │     │ (Critic) │     │ (Hybrid) │     │ (Reward) │  │
│   └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘  │
│        │                │                │                │         │
│        └────────────────┴────────────────┴────────────────┘         │
│                              │                                       │
│                    ┌─────────▼─────────┐                            │
│                    │   Hivemind DHT    │                            │
│                    │ (Peer Discovery)  │                            │
│                    └───────────────────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

- **DHT Manager**: Handles peer discovery and coordination via Hivemind
- **Gossip Protocol**: Synchronizes state (policy versions, metrics) across nodes
- **Peer Router**: Intelligent load balancing and peer selection
- **Gradient Synchronizer**: Decentralized gradient averaging
- **Experience Buffer**: Manages RLHF rollout data

## Installation

```bash
# From PyPI (when published)
pip install decentralized-verl

# From source
git clone https://github.com/decentralized-verl/decentralized-verl.git
cd decentralized-verl
pip install -e .

# With vLLM support
pip install -e ".[vllm]"

# Development install
pip install -e ".[dev]"
```

## Quick Start

### 1. Start a DHT Bootstrap Node

```bash
# Terminal 1: Start bootstrap node
dverl-dht --port 31337
```

Output:
```
Decentralized veRL DHT Bootstrap Node
============================================================
Node ID: bootstrap
Peer ID: QmXx...

Connect other nodes with:
  --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx...
============================================================
```

### 2. Start Worker Nodes

```bash
# Terminal 2: Start actor worker
dverl-worker \
    --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \
    --model meta-llama/Llama-2-7b-hf \
    --role actor \
    --port 31338

# Terminal 3: Start critic worker
dverl-worker \
    --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \
    --model meta-llama/Llama-2-7b-hf \
    --role critic \
    --port 31339
```

### 3. Run Training

```bash
# Create prompts file
echo '["Write a poem about AI", "Explain quantum computing"]' > prompts.json

# Start training
dverl-train \
    --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \
    --model meta-llama/Llama-2-7b-hf \
    --prompts-file prompts.json \
    --algorithm ppo \
    --num-epochs 100
```

## Python API

```python
import asyncio
from decentralized_verl import (
    DecentralizedConfig,
    DecentralizedTrainer,
    NetworkConfig,
    ModelConfig,
)

async def train():
    # Configure
    config = DecentralizedConfig(
        network=NetworkConfig(
            initial_peers=["/ip4/192.168.1.100/tcp/31337/p2p/QmXx..."],
            dht_port=31340,
        ),
        model=ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            dtype="bfloat16",
        ),
    )

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

    # Create trainer
    trainer = DecentralizedTrainer(
        config=config,
        actor_model=model,
        tokenizer=tokenizer,
    )

    # Train
    prompts = ["Write a poem about AI", "Explain quantum computing"]
    await trainer.train(prompts, num_epochs=100)

asyncio.run(train())
```

## Algorithms

### PPO (Proximal Policy Optimization)
```python
config.training.algorithm = Algorithm.PPO
config.training.clip_range = 0.2
config.training.ppo_epochs = 4
```

### GRPO (Group Relative Policy Optimization)
```python
config.training.algorithm = Algorithm.GRPO
config.training.num_samples_per_prompt = 4  # Generate multiple samples per prompt
```

### RLOO (REINFORCE Leave-One-Out)
```python
config.training.algorithm = Algorithm.RLOO
```

## Configuration

```python
DecentralizedConfig(
    # Network settings
    network=NetworkConfig(
        initial_peers=["..."],
        dht_port=31337,
        gossip_fanout=6,
        heartbeat_interval=10.0,
    ),

    # Model settings
    model=ModelConfig(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        dtype="bfloat16",
        use_lora=True,
        lora_rank=16,
    ),

    # Training settings
    training=TrainingConfig(
        algorithm=Algorithm.PPO,
        actor_lr=1e-6,
        critic_lr=1e-5,
        train_batch_size=64,
        rollout_batch_size=512,
        max_grad_norm=1.0,
    ),

    # Algorithm settings
    algorithm=AlgorithmConfig(
        reward_model_name="OpenAssistant/reward-model-deberta-v3-large",
        gamma=1.0,
        gae_lambda=0.95,
    ),

    # Node settings
    node_role=NodeRole.HYBRID,
    generation_backend=GenerationBackend.VLLM,
)
```

## Comparison with veRL

| Feature | veRL | Decentralized veRL |
|---------|------|-------------------|
| Coordination | Ray (centralized) | Hivemind DHT (decentralized) |
| Fault Tolerance | Ray fault tolerance | DHT-based recovery |
| GPU Support | Datacenter GPUs | Heterogeneous (consumer + datacenter) |
| Network | Cluster network | P2P over internet |
| Scaling | Vertical | Horizontal (add more peers) |

## Comparison with BloomBee

| Feature | BloomBee | Decentralized veRL |
|---------|----------|-------------------|
| Focus | Inference | RLHF Training |
| Algorithms | - | PPO, GRPO, RLOO |
| Model Types | LLM | LLM + Reward Models |
| Training | Basic fine-tuning | Full RLHF loop |

## Project Structure

```
decentralized-verl/
├── decentralized_verl/
│   ├── core/
│   │   ├── config.py      # Configuration classes
│   │   ├── node.py        # Decentralized node
│   │   └── protocol.py    # P2P protocol definitions
│   ├── network/
│   │   ├── dht_manager.py # Hivemind DHT integration
│   │   ├── peer_router.py # Peer selection and routing
│   │   └── gossip.py      # State synchronization
│   ├── training/
│   │   ├── coordinator.py # Training coordination
│   │   ├── trainer.py     # Main trainer
│   │   ├── experience.py  # Experience buffer
│   │   └── gradient_sync.py # Gradient synchronization
│   ├── workers/
│   │   ├── actor.py       # Actor/policy model
│   │   ├── critic.py      # Critic/value model
│   │   ├── reward.py      # Reward model
│   │   └── reference.py   # Reference model
│   ├── algorithms/
│   │   ├── ppo.py         # PPO implementation
│   │   └── grpo.py        # GRPO implementation
│   └── cli/
│       ├── run_dht.py     # DHT bootstrap
│       ├── run_worker.py  # Worker node
│       └── run_training.py # Training script
└── pyproject.toml
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Hivemind >= 1.1.10
- Transformers >= 4.35
- (Optional) vLLM >= 0.4.0
- (Optional) SGLang >= 0.1.0

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

Apache-2.0

## Citation

If you use Decentralized veRL in your research, please cite:

```bibtex
@software{decentralized_verl,
  title = {Decentralized veRL: P2P RLHF Training for LLMs},
  year = {2024},
  url = {https://github.com/decentralized-verl/decentralized-verl}
}
```

## Acknowledgments

- [veRL](https://github.com/volcengine/verl) - ByteDance's RLHF framework
- [BloomBee](https://github.com/yottalabsai/BloomBee) - YottaLabs' decentralized LLM
- [Hivemind](https://github.com/learning-at-home/hivemind) - Decentralized deep learning
- [Petals](https://github.com/bigscience-workshop/petals) - Distributed LLM inference
