"""Peer routing and load balancing for decentralized veRL."""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import heapq

from decentralized_verl.core.protocol import PeerInfo
from decentralized_verl.core.config import NodeRole

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy for peer selection."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    LOCALITY_AWARE = "locality_aware"
    CAPABILITY_MATCH = "capability_match"


class RoutingPolicy(ABC):
    """Abstract base class for routing policies."""

    @abstractmethod
    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        **kwargs,
    ) -> List[PeerInfo]:
        """Select peers based on policy."""
        pass


class RoundRobinPolicy(RoutingPolicy):
    """Round-robin peer selection."""

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()

    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        **kwargs,
    ) -> List[PeerInfo]:
        if not peers:
            return []

        selected = []
        for _ in range(min(count, len(peers))):
            selected.append(peers[self._index % len(peers)])
            self._index += 1

        return selected


class LeastLoadedPolicy(RoutingPolicy):
    """Select peers with lowest load."""

    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        **kwargs,
    ) -> List[PeerInfo]:
        if not peers:
            return []

        # Sort by load (ascending)
        sorted_peers = sorted(peers, key=lambda p: p.load)
        return sorted_peers[:count]


class RandomPolicy(RoutingPolicy):
    """Random peer selection."""

    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        **kwargs,
    ) -> List[PeerInfo]:
        if not peers:
            return []

        return random.sample(peers, min(count, len(peers)))


class LocalityAwarePolicy(RoutingPolicy):
    """Select peers that are geographically close or have low latency."""

    def __init__(self, latency_map: Optional[Dict[str, float]] = None):
        self.latency_map = latency_map or {}

    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        **kwargs,
    ) -> List[PeerInfo]:
        if not peers:
            return []

        # Sort by latency (lowest first), unknown latency goes to end
        def get_latency(peer: PeerInfo) -> float:
            return self.latency_map.get(peer.peer_id, float('inf'))

        sorted_peers = sorted(peers, key=get_latency)
        return sorted_peers[:count]

    def update_latency(self, peer_id: str, latency: float) -> None:
        """Update latency measurement for a peer."""
        self.latency_map[peer_id] = latency


class CapabilityMatchPolicy(RoutingPolicy):
    """Select peers that have required capabilities."""

    def select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        required_capabilities: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[PeerInfo]:
        if not peers:
            return []

        required_capabilities = required_capabilities or {}

        def matches_capabilities(peer: PeerInfo) -> bool:
            for key, value in required_capabilities.items():
                peer_cap = peer.capabilities.get(key)
                if peer_cap != value:
                    return False
            return True

        matching = [p for p in peers if matches_capabilities(p)]

        # If not enough matching, fall back to least loaded
        if len(matching) < count:
            non_matching = [p for p in peers if p not in matching]
            matching.extend(sorted(non_matching, key=lambda p: p.load))

        return matching[:count]


@dataclass
class PeerScore:
    """Score for peer selection with multiple criteria."""
    peer_id: str
    load_score: float = 0.0
    latency_score: float = 0.0
    capability_score: float = 0.0
    reliability_score: float = 0.0
    total_score: float = 0.0

    def compute_total(
        self,
        load_weight: float = 0.3,
        latency_weight: float = 0.2,
        capability_weight: float = 0.3,
        reliability_weight: float = 0.2,
    ) -> None:
        """Compute weighted total score."""
        self.total_score = (
            load_weight * self.load_score +
            latency_weight * self.latency_score +
            capability_weight * self.capability_score +
            reliability_weight * self.reliability_score
        )


class PeerRouter:
    """
    Routes requests to appropriate peers based on various criteria.

    Implements intelligent peer selection with load balancing,
    latency awareness, and capability matching.
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        latency_map: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize peer router.

        Args:
            strategy: Default routing strategy
            latency_map: Initial latency measurements
        """
        self.strategy = strategy
        self.latency_map = latency_map or {}

        # Initialize policies
        self._policies = {
            RoutingStrategy.ROUND_ROBIN: RoundRobinPolicy(),
            RoutingStrategy.LEAST_LOADED: LeastLoadedPolicy(),
            RoutingStrategy.RANDOM: RandomPolicy(),
            RoutingStrategy.LOCALITY_AWARE: LocalityAwarePolicy(self.latency_map),
            RoutingStrategy.CAPABILITY_MATCH: CapabilityMatchPolicy(),
        }

        # Track peer reliability
        self._success_count: Dict[str, int] = {}
        self._failure_count: Dict[str, int] = {}
        self._last_used: Dict[str, float] = {}

    def select_actors(
        self,
        peers: List[PeerInfo],
        count: int = 1,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> List[PeerInfo]:
        """
        Select actor peers for rollout generation.

        Args:
            peers: Available peers
            count: Number of peers to select
            strategy: Routing strategy (uses default if None)
            **kwargs: Additional arguments for routing policy

        Returns:
            Selected peer list
        """
        # Filter to only actor/hybrid peers
        actor_peers = [
            p for p in peers
            if p.role in (NodeRole.ACTOR.value, NodeRole.HYBRID.value)
            and p.is_alive()
        ]

        return self._select_peers(actor_peers, count, strategy, **kwargs)

    def select_critics(
        self,
        peers: List[PeerInfo],
        count: int = 1,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> List[PeerInfo]:
        """
        Select critic peers for value estimation.

        Args:
            peers: Available peers
            count: Number of peers to select
            strategy: Routing strategy (uses default if None)

        Returns:
            Selected peer list
        """
        # Filter to only critic/hybrid peers
        critic_peers = [
            p for p in peers
            if p.role in (NodeRole.CRITIC.value, NodeRole.HYBRID.value)
            and p.is_alive()
        ]

        return self._select_peers(critic_peers, count, strategy, **kwargs)

    def select_reward_peers(
        self,
        peers: List[PeerInfo],
        count: int = 1,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> List[PeerInfo]:
        """
        Select reward model peers.

        Args:
            peers: Available peers
            count: Number of peers to select
            strategy: Routing strategy (uses default if None)

        Returns:
            Selected peer list
        """
        reward_peers = [
            p for p in peers
            if p.role == NodeRole.REWARD.value
            and p.is_alive()
        ]

        return self._select_peers(reward_peers, count, strategy, **kwargs)

    def select_for_blocks(
        self,
        peers: List[PeerInfo],
        block_indices: List[int],
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> Dict[int, PeerInfo]:
        """
        Select peers that serve specific transformer blocks.

        Args:
            peers: Available peers
            block_indices: Required block indices
            strategy: Routing strategy

        Returns:
            Mapping of block index to selected peer
        """
        block_to_peer: Dict[int, PeerInfo] = {}

        for block_idx in block_indices:
            # Find peers serving this block
            block_peers = [
                p for p in peers
                if block_idx in p.blocks_served
                and p.is_alive()
            ]

            if block_peers:
                selected = self._select_peers(block_peers, 1, strategy, **kwargs)
                if selected:
                    block_to_peer[block_idx] = selected[0]

        return block_to_peer

    def _select_peers(
        self,
        peers: List[PeerInfo],
        count: int,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> List[PeerInfo]:
        """Internal peer selection using configured policy."""
        if not peers:
            return []

        strategy = strategy or self.strategy
        policy = self._policies.get(strategy, self._policies[RoutingStrategy.LEAST_LOADED])

        selected = policy.select_peers(peers, count, **kwargs)

        # Update last used timestamp
        for peer in selected:
            self._last_used[peer.peer_id] = time.time()

        return selected

    def record_success(self, peer_id: str) -> None:
        """Record successful interaction with peer."""
        self._success_count[peer_id] = self._success_count.get(peer_id, 0) + 1

    def record_failure(self, peer_id: str) -> None:
        """Record failed interaction with peer."""
        self._failure_count[peer_id] = self._failure_count.get(peer_id, 0) + 1

    def update_latency(self, peer_id: str, latency: float) -> None:
        """Update latency measurement for peer."""
        self.latency_map[peer_id] = latency
        if isinstance(self._policies[RoutingStrategy.LOCALITY_AWARE], LocalityAwarePolicy):
            self._policies[RoutingStrategy.LOCALITY_AWARE].update_latency(peer_id, latency)

    def get_reliability(self, peer_id: str) -> float:
        """
        Get reliability score for peer.

        Returns:
            Reliability between 0 and 1
        """
        successes = self._success_count.get(peer_id, 0)
        failures = self._failure_count.get(peer_id, 0)
        total = successes + failures

        if total == 0:
            return 0.5  # Unknown reliability

        return successes / total

    def get_peer_scores(
        self,
        peers: List[PeerInfo],
        required_capabilities: Optional[Dict[str, Any]] = None,
    ) -> List[PeerScore]:
        """
        Compute comprehensive scores for peers.

        Args:
            peers: Peers to score
            required_capabilities: Required capabilities

        Returns:
            List of peer scores
        """
        scores = []
        required_capabilities = required_capabilities or {}

        # Normalize load values
        max_load = max((p.load for p in peers), default=1.0) or 1.0

        for peer in peers:
            score = PeerScore(peer_id=peer.peer_id)

            # Load score (lower is better)
            score.load_score = 1.0 - (peer.load / max_load)

            # Latency score (lower is better)
            latency = self.latency_map.get(peer.peer_id, 100.0)
            score.latency_score = 1.0 / (1.0 + latency / 100.0)

            # Capability score
            if required_capabilities:
                matches = sum(
                    1 for k, v in required_capabilities.items()
                    if peer.capabilities.get(k) == v
                )
                score.capability_score = matches / len(required_capabilities)
            else:
                score.capability_score = 1.0

            # Reliability score
            score.reliability_score = self.get_reliability(peer.peer_id)

            score.compute_total()
            scores.append(score)

        return sorted(scores, key=lambda s: s.total_score, reverse=True)


class WorkloadBalancer:
    """
    Balances workload across peers for distributed training.

    Distributes prompts and training batches based on peer capacity.
    """

    def __init__(self, router: PeerRouter):
        """
        Initialize workload balancer.

        Args:
            router: Peer router for selection
        """
        self.router = router
        self._assigned_work: Dict[str, int] = {}

    def distribute_prompts(
        self,
        prompts: List[str],
        peers: List[PeerInfo],
        balance_by_capacity: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Distribute prompts across peers.

        Args:
            prompts: Prompts to distribute
            peers: Available peers
            balance_by_capacity: Weight by GPU capacity if True

        Returns:
            Mapping of peer_id to assigned prompts
        """
        if not peers:
            return {}

        distribution: Dict[str, List[str]] = {p.peer_id: [] for p in peers}

        if balance_by_capacity:
            # Weight by GPU memory capacity
            capacities = {}
            for peer in peers:
                gpu_mem = peer.capabilities.get("gpu_memory", [0])
                capacities[peer.peer_id] = sum(gpu_mem) if isinstance(gpu_mem, list) else gpu_mem

            total_capacity = sum(capacities.values()) or 1

            # Distribute proportionally
            for i, prompt in enumerate(prompts):
                # Find peer with most remaining capacity
                best_peer = min(
                    peers,
                    key=lambda p: len(distribution[p.peer_id]) / (capacities[p.peer_id] / total_capacity + 0.001)
                )
                distribution[best_peer.peer_id].append(prompt)
        else:
            # Simple round-robin
            for i, prompt in enumerate(prompts):
                peer_idx = i % len(peers)
                distribution[peers[peer_idx].peer_id].append(prompt)

        return distribution

    def distribute_batches(
        self,
        total_samples: int,
        batch_size: int,
        peers: List[PeerInfo],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Distribute training batches across peers.

        Args:
            total_samples: Total number of samples
            batch_size: Batch size per peer
            peers: Available peers

        Returns:
            Mapping of peer_id to (start_idx, end_idx)
        """
        if not peers:
            return {}

        distribution: Dict[str, Tuple[int, int]] = {}
        samples_per_peer = total_samples // len(peers)
        remainder = total_samples % len(peers)

        current_idx = 0
        for i, peer in enumerate(peers):
            peer_samples = samples_per_peer + (1 if i < remainder else 0)
            distribution[peer.peer_id] = (current_idx, current_idx + peer_samples)
            current_idx += peer_samples

        return distribution
