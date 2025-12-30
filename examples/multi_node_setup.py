#!/usr/bin/env python3
"""
Multi-node setup example for decentralized veRL.

This example shows how to set up multiple nodes for distributed
RLHF training. Run each command on a different machine/terminal.

Network Topology:
    Bootstrap Node (DHT) --> Worker 1 (Actor)
                       |
                       +--> Worker 2 (Critic)
                       |
                       +--> Worker 3 (Hybrid)
                       |
                       +--> Coordinator (Training)
"""

import asyncio
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


SETUP_INSTRUCTIONS = """
=================================================================
Decentralized veRL Multi-Node Setup Guide
=================================================================

STEP 1: Start the DHT Bootstrap Node
-------------------------------------
On Machine 1 (or Terminal 1):

    dverl-dht --port 31337

This will output a multiaddr like:
    /ip4/192.168.1.100/tcp/31337/p2p/QmXx...

Save this address - you'll need it for other nodes.


STEP 2: Start Actor Worker(s)
-----------------------------
On Machine 2 (or Terminal 2):

    dverl-worker \\
        --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \\
        --model meta-llama/Llama-2-7b-hf \\
        --role actor \\
        --port 31338 \\
        --backend vllm  # or huggingface

For multiple actors, use different ports:
    --port 31339, 31340, etc.


STEP 3: Start Critic Worker(s)
------------------------------
On Machine 3 (or Terminal 3):

    dverl-worker \\
        --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \\
        --model meta-llama/Llama-2-7b-hf \\
        --role critic \\
        --port 31341


STEP 4: Start Training Coordinator
----------------------------------
On Machine 4 (or Terminal 4):

    dverl-train \\
        --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \\
        --model meta-llama/Llama-2-7b-hf \\
        --prompts-file prompts.json \\
        --algorithm ppo \\
        --num-epochs 100 \\
        --output-dir ./outputs


ALTERNATIVE: Hybrid Setup (All-in-One Workers)
----------------------------------------------
For smaller setups, use hybrid workers that handle all roles:

    dverl-worker \\
        --initial-peers /ip4/192.168.1.100/tcp/31337/p2p/QmXx... \\
        --model meta-llama/Llama-2-7b-hf \\
        --role hybrid \\
        --port 31338


TIPS:
-----
1. NAT Traversal: If nodes are behind NAT, use --announce-host with
   your public IP address.

2. Firewall: Ensure the DHT port (31337) and worker ports are open.

3. GPU Allocation: Use CUDA_VISIBLE_DEVICES to control GPU assignment:
   CUDA_VISIBLE_DEVICES=0 dverl-worker --role actor ...
   CUDA_VISIBLE_DEVICES=1 dverl-worker --role critic ...

4. Monitoring: Check node status via the coordinator's status endpoint
   or watch the logs for peer discovery messages.

5. Fault Tolerance: The system automatically handles node failures.
   Workers will reconnect and training will resume.

=================================================================
"""


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node setup guide for decentralized veRL"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate a local multi-node setup"
    )
    args = parser.parse_args()

    if args.simulate:
        print("\nSimulating local multi-node setup...")
        print("This would start multiple processes locally.")
        print("For production, run each component on separate machines.\n")

        # In a real simulation, we'd spawn subprocesses
        # For now, just show the commands
        commands = [
            "dverl-dht --port 31337",
            "dverl-worker --initial-peers ... --role actor --port 31338",
            "dverl-worker --initial-peers ... --role critic --port 31339",
            "dverl-train --initial-peers ... --prompts-file prompts.json",
        ]

        print("Commands that would be executed:")
        for i, cmd in enumerate(commands, 1):
            print(f"  [{i}] {cmd}")

    else:
        print(SETUP_INSTRUCTIONS)


if __name__ == "__main__":
    main()
