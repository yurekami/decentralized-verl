#!/usr/bin/env python3
"""Run a DHT bootstrap node for decentralized veRL network."""

import argparse
import asyncio
import logging
import signal
import sys

from decentralized_verl.core.config import NetworkConfig
from decentralized_verl.network.dht_manager import DHTManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a DHT bootstrap node for decentralized veRL"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=31337,
        help="Port for DHT (default: 31337)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--announce-host",
        type=str,
        default=None,
        help="Public IP to announce (for NAT traversal)",
    )
    parser.add_argument(
        "--initial-peers",
        type=str,
        nargs="*",
        default=[],
        help="Initial peers to connect to",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default="bootstrap",
        help="Node identifier",
    )
    return parser.parse_args()


async def run_dht_node(args):
    """Run the DHT node."""
    config = NetworkConfig(
        dht_port=args.port,
        announce_host=args.announce_host,
        initial_peers=args.initial_peers,
        listen_on=f"/ip4/{args.host}/tcp",
    )

    dht_manager = DHTManager(config, args.node_id)

    # Setup signal handlers
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await dht_manager.start()

        print("\n" + "=" * 60)
        print("Decentralized veRL DHT Bootstrap Node")
        print("=" * 60)
        print(f"\nNode ID: {args.node_id}")
        print(f"Peer ID: {dht_manager.get_peer_id()}")
        print(f"\nConnect other nodes with:")
        print(f"  --initial-peers {dht_manager.get_multiaddr()}")
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Run until shutdown
        await stop_event.wait()

    finally:
        await dht_manager.stop()
        logger.info("DHT node stopped")


def main():
    args = parse_args()
    asyncio.run(run_dht_node(args))


if __name__ == "__main__":
    main()
