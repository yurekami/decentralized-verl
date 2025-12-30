#!/usr/bin/env python3
"""Run a worker node for decentralized veRL."""

import argparse
import asyncio
import logging
import signal

from decentralized_verl.core.config import (
    DecentralizedConfig,
    NetworkConfig,
    ModelConfig,
    TrainingConfig,
    NodeRole,
    GenerationBackend,
)
from decentralized_verl.core.node import DecentralizedNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a worker node for decentralized veRL"
    )

    # Network
    parser.add_argument(
        "--initial-peers",
        type=str,
        nargs="+",
        required=True,
        help="DHT peers to connect to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=31338,
        help="Port for this worker (default: 31338)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )

    # Role
    parser.add_argument(
        "--role",
        type=str,
        default="hybrid",
        choices=["actor", "critic", "reward", "reference", "hybrid"],
        help="Worker role (default: hybrid)",
    )

    # Blocks
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of transformer blocks to serve",
    )
    parser.add_argument(
        "--block-indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific block indices to serve",
    )

    # Generation
    parser.add_argument(
        "--backend",
        type=str,
        default="huggingface",
        choices=["vllm", "sglang", "huggingface"],
        help="Generation backend",
    )

    return parser.parse_args()


async def run_worker(args):
    """Run the worker node."""
    # Create config
    network_config = NetworkConfig(
        dht_port=args.port,
        initial_peers=args.initial_peers,
        listen_on=f"/ip4/{args.host}/tcp",
    )

    model_config = ModelConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    role_map = {
        "actor": NodeRole.ACTOR,
        "critic": NodeRole.CRITIC,
        "reward": NodeRole.REWARD,
        "reference": NodeRole.REFERENCE,
        "hybrid": NodeRole.HYBRID,
    }

    backend_map = {
        "vllm": GenerationBackend.VLLM,
        "sglang": GenerationBackend.SGLANG,
        "huggingface": GenerationBackend.HUGGINGFACE,
    }

    config = DecentralizedConfig(
        network=network_config,
        model=model_config,
        node_role=role_map[args.role],
        num_blocks_to_serve=args.num_blocks,
        block_indices=args.block_indices,
        generation_backend=backend_map[args.backend],
    )

    node = DecentralizedNode(config)

    # Setup signal handlers
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await node.start()

        print("\n" + "=" * 60)
        print("Decentralized veRL Worker Node")
        print("=" * 60)
        print(f"\nNode ID: {node.node_id}")
        print(f"Role: {args.role}")
        print(f"Model: {args.model}")
        print(f"Backend: {args.backend}")
        print(f"Blocks: {args.block_indices or list(range(args.num_blocks))}")
        print(f"\nStatus: {node.get_status()}")
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Run until shutdown
        await stop_event.wait()

    finally:
        await node.stop()
        logger.info("Worker node stopped")


def main():
    args = parse_args()
    asyncio.run(run_worker(args))


if __name__ == "__main__":
    main()
