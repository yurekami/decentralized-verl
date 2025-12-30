#!/usr/bin/env python3
"""Run decentralized RLHF training."""

import argparse
import asyncio
import logging
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decentralized_verl.core.config import (
    DecentralizedConfig,
    NetworkConfig,
    ModelConfig,
    TrainingConfig,
    AlgorithmConfig,
    Algorithm,
)
from decentralized_verl.training.trainer import DecentralizedTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run decentralized RLHF training"
    )

    # Network
    parser.add_argument(
        "--initial-peers",
        type=str,
        nargs="+",
        default=[],
        help="DHT peers to connect to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=31340,
        help="Port for this node",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Policy model name or path",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Reward model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype",
    )

    # Training
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "grpo", "reinforce", "rloo"],
        help="Training algorithm",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=512,
        help="Rollout generation batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate",
    )

    # Data
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="JSON file with training prompts",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )

    # Misc
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_prompts(path: str) -> list:
    """Load prompts from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # List of prompts
        if isinstance(data[0], str):
            return data
        elif isinstance(data[0], dict):
            return [d.get("prompt", d.get("text", "")) for d in data]

    elif isinstance(data, dict):
        # Dataset format
        return data.get("prompts", data.get("train", []))

    return []


async def run_training(args):
    """Run the training."""
    # Load config
    if args.config:
        config = DecentralizedConfig.load(args.config)
    else:
        algorithm_map = {
            "ppo": Algorithm.PPO,
            "grpo": Algorithm.GRPO,
            "reinforce": Algorithm.REINFORCE,
            "rloo": Algorithm.RLOO,
        }

        config = DecentralizedConfig(
            network=NetworkConfig(
                dht_port=args.port,
                initial_peers=args.initial_peers,
            ),
            model=ModelConfig(
                model_name_or_path=args.model,
                dtype=args.dtype,
            ),
            training=TrainingConfig(
                algorithm=algorithm_map[args.algorithm],
                actor_lr=args.lr,
                train_batch_size=args.batch_size,
                rollout_batch_size=args.rollout_batch_size,
            ),
            algorithm=AlgorithmConfig(
                reward_model_name=args.reward_model,
            ),
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
        )

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Load models
    logger.info(f"Loading model: {config.model.model_name_or_path}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reward model if specified
    reward_model = None
    if config.algorithm.reward_model_name:
        logger.info(f"Loading reward model: {config.algorithm.reward_model_name}")
        from transformers import AutoModelForSequenceClassification
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.algorithm.reward_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    # Create trainer
    trainer = DecentralizedTrainer(
        config=config,
        actor_model=actor_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
    )

    # Resume if specified
    if args.resume:
        await trainer.load_checkpoint(args.resume)

    try:
        # Run training
        final_state = await trainer.train(
            prompts=prompts,
            num_epochs=args.num_epochs,
        )

        logger.info(f"Training complete!")
        logger.info(f"Final state: {final_state}")

    finally:
        await trainer.cleanup()


def main():
    args = parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
