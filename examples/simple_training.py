#!/usr/bin/env python3
"""
Simple example of decentralized RLHF training.

This example demonstrates how to set up a basic training run
using the decentralized veRL framework.

Usage:
    python simple_training.py --model gpt2 --prompts prompts.json
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decentralized_verl import (
    DecentralizedConfig,
    NetworkConfig,
    ModelConfig,
    TrainingConfig,
)
from decentralized_verl.training.trainer import DecentralizedTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple decentralized RLHF training")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--prompts", type=str, default=None, help="Prompts JSON file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    return parser.parse_args()


def create_sample_prompts():
    """Create sample prompts for demonstration."""
    return [
        "Explain artificial intelligence in simple terms.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How does machine learning differ from traditional programming?",
        "What is the importance of biodiversity?",
        "Explain quantum computing to a beginner.",
        "What are the causes of climate change?",
        "How do neural networks work?",
        "Describe the water cycle.",
        "What is the theory of relativity?",
    ]


def simple_reward_function(prompt: str, response: str) -> float:
    """
    Simple reward function based on response quality heuristics.

    In practice, you would use a trained reward model.
    """
    reward = 0.0

    # Reward for appropriate length
    words = response.split()
    if 20 <= len(words) <= 200:
        reward += 0.3
    elif len(words) < 10:
        reward -= 0.3

    # Reward for not being repetitive
    unique_words = set(words)
    if len(unique_words) / max(len(words), 1) > 0.5:
        reward += 0.2

    # Reward for containing relevant terms
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    overlap = len(prompt_words & response_words)
    reward += min(overlap * 0.1, 0.3)

    # Penalty for ending mid-sentence
    if response and response[-1] not in ".!?":
        reward -= 0.1

    return max(0.0, min(1.0, reward))


async def main():
    args = parse_args()

    # Load prompts
    if args.prompts and Path(args.prompts).exists():
        with open(args.prompts) as f:
            prompts = json.load(f)
    else:
        prompts = create_sample_prompts()
        logger.info("Using sample prompts")

    logger.info(f"Loaded {len(prompts)} prompts")

    # Create configuration
    config = DecentralizedConfig(
        network=NetworkConfig(
            dht_port=31337,
            initial_peers=[],  # Standalone mode
        ),
        model=ModelConfig(
            model_name_or_path=args.model,
            dtype="float32" if not torch.cuda.is_available() else "bfloat16",
        ),
        training=TrainingConfig(
            actor_lr=args.lr,
            train_batch_size=args.batch_size,
            rollout_batch_size=args.batch_size,
            ppo_epochs=2,
            max_prompt_length=128,
            max_response_length=128,
        ),
        output_dir=args.output,
        checkpoint_dir=f"{args.output}/checkpoints",
    )

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
    trainer = DecentralizedTrainer(
        config=config,
        actor_model=model,
        tokenizer=tokenizer,
    )

    # Add simple callback
    def on_epoch(metrics):
        logger.info(f"Epoch {metrics.epoch}: reward={metrics.avg_reward:.4f}")

    trainer.add_callback(on_epoch)

    # Run training
    logger.info("Starting training...")
    try:
        final_state = await trainer.train(
            prompts=prompts,
            num_epochs=args.epochs,
        )
        logger.info(f"Training complete! Best reward: {final_state.best_reward:.4f}")
    finally:
        await trainer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
