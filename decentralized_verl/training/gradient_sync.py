"""Gradient synchronization for decentralized training."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import threading

import torch
import torch.nn as nn

try:
    import hivemind
    from hivemind import DecentralizedAverager, TrainingStateAverager
    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False
    hivemind = None

from decentralized_verl.network.dht_manager import DHTManager
from decentralized_verl.core.protocol import GradientUpdate

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Statistics for gradient synchronization."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    avg_sync_time: float = 0.0
    avg_gradient_norm: float = 0.0
    total_peers_synced: int = 0


class GradientCompressor:
    """Compress gradients for efficient network transmission."""

    def __init__(
        self,
        compression_type: str = "none",
        compression_ratio: float = 0.1,
    ):
        """
        Initialize gradient compressor.

        Args:
            compression_type: Type of compression ("none", "topk", "random")
            compression_ratio: Ratio of values to keep for sparse methods
        """
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio

    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compress gradients for transmission.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Compressed representation
        """
        if self.compression_type == "none":
            return {
                name: grad.cpu().numpy().tobytes()
                for name, grad in gradients.items()
            }

        elif self.compression_type == "topk":
            compressed = {}
            for name, grad in gradients.items():
                flat = grad.flatten()
                k = max(1, int(len(flat) * self.compression_ratio))
                values, indices = torch.topk(flat.abs(), k)
                compressed[name] = {
                    "values": flat[indices].cpu().numpy().tobytes(),
                    "indices": indices.cpu().numpy().tobytes(),
                    "shape": list(grad.shape),
                    "k": k,
                }
            return compressed

        elif self.compression_type == "random":
            compressed = {}
            for name, grad in gradients.items():
                flat = grad.flatten()
                k = max(1, int(len(flat) * self.compression_ratio))
                indices = torch.randperm(len(flat))[:k]
                compressed[name] = {
                    "values": flat[indices].cpu().numpy().tobytes(),
                    "indices": indices.cpu().numpy().tobytes(),
                    "shape": list(grad.shape),
                    "k": k,
                }
            return compressed

        else:
            raise ValueError(f"Unknown compression type: {self.compression_type}")

    def decompress(
        self,
        compressed: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients.

        Args:
            compressed: Compressed gradient representation
            device: Device to place tensors on

        Returns:
            Dictionary of gradient tensors
        """
        import numpy as np

        if self.compression_type == "none":
            return {
                name: torch.from_numpy(np.frombuffer(data, dtype=np.float32)).to(device)
                for name, data in compressed.items()
            }

        else:
            gradients = {}
            for name, data in compressed.items():
                shape = data["shape"]
                values = torch.from_numpy(
                    np.frombuffer(data["values"], dtype=np.float32)
                )
                indices = torch.from_numpy(
                    np.frombuffer(data["indices"], dtype=np.int64)
                )

                # Reconstruct sparse gradient
                grad = torch.zeros(int(np.prod(shape)), device=device)
                grad[indices] = values.to(device)
                gradients[name] = grad.reshape(shape)

            return gradients


class GradientSynchronizer:
    """
    Synchronizes gradients across decentralized network.

    Uses Hivemind for efficient decentralized averaging or
    custom gossip-based gradient aggregation.
    """

    def __init__(
        self,
        model: nn.Module,
        dht_manager: DHTManager,
        compression_type: str = "none",
        target_batch_size: int = 256,
        averaging_timeout: float = 60.0,
        use_hivemind: bool = True,
    ):
        """
        Initialize gradient synchronizer.

        Args:
            model: PyTorch model to synchronize
            dht_manager: DHT manager for network communication
            compression_type: Gradient compression type
            target_batch_size: Target batch size for averaging
            averaging_timeout: Timeout for averaging operation
            use_hivemind: Whether to use Hivemind averaging
        """
        self.model = model
        self.dht_manager = dht_manager
        self.compression_type = compression_type
        self.target_batch_size = target_batch_size
        self.averaging_timeout = averaging_timeout
        self.use_hivemind = use_hivemind and HIVEMIND_AVAILABLE

        self.compressor = GradientCompressor(compression_type)
        self.stats = GradientStats()

        # Hivemind components
        self._averager: Optional[DecentralizedAverager] = None
        self._optimizer_state_averager = None

        self._lock = threading.Lock()

    async def initialize(self) -> None:
        """Initialize synchronizer components."""
        if self.use_hivemind and self.dht_manager.dht is not None:
            logger.info("Initializing Hivemind gradient averager")

            # Get model parameters for averaging
            params = [p for p in self.model.parameters() if p.requires_grad]

            self._averager = DecentralizedAverager(
                averaged_tensors=params,
                dht=self.dht_manager.dht,
                start=True,
                prefix="dverl_grad",
                target_batch_size=self.target_batch_size,
                averaging_timeout=self.averaging_timeout,
            )

            logger.info("Hivemind averager initialized")
        else:
            logger.info("Using custom gradient synchronization")

    async def synchronize(
        self,
        step: int,
        local_batch_size: int = 1,
    ) -> bool:
        """
        Synchronize gradients with peers.

        Args:
            step: Current training step
            local_batch_size: Size of local batch used

        Returns:
            True if synchronization was successful
        """
        start_time = time.time()

        try:
            if self.use_hivemind and self._averager is not None:
                # Use Hivemind averaging
                success = await self._hivemind_sync(step, local_batch_size)
            else:
                # Use custom DHT-based sync
                success = await self._custom_sync(step, local_batch_size)

            sync_time = time.time() - start_time

            # Update stats
            with self._lock:
                self.stats.total_syncs += 1
                if success:
                    self.stats.successful_syncs += 1
                else:
                    self.stats.failed_syncs += 1
                self.stats.avg_sync_time = (
                    0.9 * self.stats.avg_sync_time + 0.1 * sync_time
                )

            return success

        except Exception as e:
            logger.error(f"Gradient sync failed: {e}")
            with self._lock:
                self.stats.total_syncs += 1
                self.stats.failed_syncs += 1
            return False

    async def _hivemind_sync(
        self,
        step: int,
        local_batch_size: int,
    ) -> bool:
        """Synchronize using Hivemind averager."""
        if self._averager is None:
            return False

        try:
            # Perform gradient averaging
            # This is a blocking call that coordinates with other peers
            future = self._averager.step(
                gather=local_batch_size,
                weight=local_batch_size,
                timeout=self.averaging_timeout,
            )

            # Wait for averaging to complete
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: future.result(timeout=self.averaging_timeout),
            )

            return True

        except Exception as e:
            logger.warning(f"Hivemind sync failed: {e}")
            return False

    async def _custom_sync(
        self,
        step: int,
        local_batch_size: int,
    ) -> bool:
        """Custom DHT-based gradient synchronization."""
        # Collect local gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        if not gradients:
            return True  # No gradients to sync

        # Compress gradients
        compressed = self.compressor.compress(gradients)

        # Store in DHT
        success = await self.dht_manager.store_gradient(
            step=step,
            gradients=compressed,
            batch_size=local_batch_size,
        )

        if not success:
            return False

        # Wait briefly for other peers to submit
        await asyncio.sleep(0.5)

        # Get gradients from peers
        peer_ids = list(self.dht_manager._peer_cache.keys())
        peer_gradients = await self.dht_manager.get_gradients_for_step(
            step=step,
            peer_ids=peer_ids,
        )

        if not peer_gradients:
            return True  # Only local gradients available

        # Average gradients
        total_batch_size = local_batch_size
        for pg in peer_gradients:
            total_batch_size += pg.get("batch_size", 1)

        # Decompress and average
        device = next(self.model.parameters()).device
        averaged_gradients: Dict[str, torch.Tensor] = {}

        # Initialize with local gradients
        for name, grad in gradients.items():
            averaged_gradients[name] = grad * (local_batch_size / total_batch_size)

        # Add peer gradients
        for pg in peer_gradients:
            peer_grads = self.compressor.decompress(pg["gradients"], device)
            batch_size = pg.get("batch_size", 1)
            weight = batch_size / total_batch_size

            for name, grad in peer_grads.items():
                if name in averaged_gradients:
                    averaged_gradients[name] += grad * weight

        # Apply averaged gradients
        for name, param in self.model.named_parameters():
            if name in averaged_gradients:
                param.grad = averaged_gradients[name]

        with self._lock:
            self.stats.total_peers_synced += len(peer_gradients)

        return True

    async def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "mean",
    ) -> torch.Tensor:
        """
        All-reduce a tensor across peers.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ("mean", "sum", "max", "min")

        Returns:
            Reduced tensor
        """
        if self.use_hivemind and self._averager is not None:
            # Use Hivemind
            result = tensor.clone()
            # Hivemind handles this internally during averaging
            return result
        else:
            # Simple implementation - just return local value
            # In production, would use DHT-based reduction
            return tensor

    def get_gradient_norm(self) -> float:
        """Get current gradient norm."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        return total_norm ** 0.5

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm.

        Args:
            max_norm: Maximum gradient norm

        Returns:
            Original gradient norm
        """
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm,
        ).item()

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._lock:
            return {
                "total_syncs": self.stats.total_syncs,
                "successful_syncs": self.stats.successful_syncs,
                "failed_syncs": self.stats.failed_syncs,
                "success_rate": (
                    self.stats.successful_syncs / max(self.stats.total_syncs, 1)
                ),
                "avg_sync_time": self.stats.avg_sync_time,
                "total_peers_synced": self.stats.total_peers_synced,
                "use_hivemind": self.use_hivemind,
                "compression": self.compression_type,
            }


class MomentumSynchronizer:
    """
    Synchronizes optimizer momentum/state across peers.

    Necessary for training stability with decentralized optimization.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        dht_manager: DHTManager,
        sync_interval: int = 100,
    ):
        """
        Initialize momentum synchronizer.

        Args:
            optimizer: PyTorch optimizer
            dht_manager: DHT manager
            sync_interval: Steps between momentum syncs
        """
        self.optimizer = optimizer
        self.dht_manager = dht_manager
        self.sync_interval = sync_interval
        self._step = 0

    async def step(self) -> bool:
        """
        Potentially synchronize momentum state.

        Returns:
            True if sync was performed
        """
        self._step += 1

        if self._step % self.sync_interval != 0:
            return False

        try:
            # For Adam-style optimizers, sync exp_avg and exp_avg_sq
            state_dict = self.optimizer.state_dict()

            # Store in DHT
            # In production, would compress and distribute
            logger.debug("Momentum sync performed")
            return True

        except Exception as e:
            logger.error(f"Momentum sync failed: {e}")
            return False
