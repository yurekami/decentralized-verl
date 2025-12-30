"""Checkpoint management for decentralized veRL."""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    step: int
    policy_version: int
    timestamp: float
    model_hash: str
    node_id: str
    metrics: Dict[str, float]


class CheckpointManager:
    """
    Manages checkpoints for decentralized training.

    Handles:
    - Local checkpoint saving/loading
    - Distributed checkpoint storage via DHT
    - Checkpoint validation and recovery
    """

    def __init__(
        self,
        checkpoint_dir: str,
        dht_manager: Optional[Any] = None,
        max_checkpoints: int = 5,
        chunk_size: int = 10 * 1024 * 1024,  # 10 MB chunks
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Local directory for checkpoints
            dht_manager: DHT manager for distributed storage
            max_checkpoints: Maximum checkpoints to keep locally
            chunk_size: Size of chunks for distributed storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.dht_manager = dht_manager
        self.max_checkpoints = max_checkpoints
        self.chunk_size = chunk_size

        self._metadata_cache: Dict[str, CheckpointMetadata] = {}

    async def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        policy_version: int,
        node_id: str,
        metrics: Optional[Dict[str, float]] = None,
        distributed: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer to save
            step: Training step
            policy_version: Policy version
            node_id: Node identifier
            metrics: Optional training metrics
            distributed: Whether to distribute via DHT

        Returns:
            Checkpoint ID
        """
        metrics = metrics or {}

        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(step, policy_version, node_id)

        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save optimizer if provided
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)

        # Compute model hash
        model_hash = self._compute_file_hash(model_path)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            step=step,
            policy_version=policy_version,
            timestamp=time.time(),
            model_hash=model_hash,
            node_id=node_id,
            metrics=metrics,
        )

        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "checkpoint_id": metadata.checkpoint_id,
                "step": metadata.step,
                "policy_version": metadata.policy_version,
                "timestamp": metadata.timestamp,
                "model_hash": metadata.model_hash,
                "node_id": metadata.node_id,
                "metrics": metadata.metrics,
            }, f, indent=2)

        self._metadata_cache[checkpoint_id] = metadata

        # Distribute via DHT if requested
        if distributed and self.dht_manager is not None:
            await self._distribute_checkpoint(checkpoint_id, checkpoint_path)

        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints()

        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id

    async def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_id: Optional[str] = None,
        from_dht: bool = False,
    ) -> Optional[CheckpointMetadata]:
        """
        Load a checkpoint.

        Args:
            model: Model to load into
            optimizer: Optional optimizer to load into
            checkpoint_id: Specific checkpoint to load (latest if None)
            from_dht: Whether to retrieve from DHT

        Returns:
            Checkpoint metadata or None if not found
        """
        if checkpoint_id is None:
            checkpoint_id = await self.get_latest_checkpoint()

        if checkpoint_id is None:
            logger.warning("No checkpoint found")
            return None

        checkpoint_path = self.checkpoint_dir / checkpoint_id

        # Try to retrieve from DHT if not local
        if not checkpoint_path.exists() and from_dht and self.dht_manager:
            await self._retrieve_checkpoint(checkpoint_id)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None

        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
                metadata = CheckpointMetadata(**data)
                self._metadata_cache[checkpoint_id] = metadata
                return metadata

        return None

    async def get_latest_checkpoint(self) -> Optional[str]:
        """Get the ID of the latest checkpoint."""
        checkpoints = []

        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        checkpoints.append((path.name, data["timestamp"]))

        if not checkpoints:
            return None

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]

    async def _distribute_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_path: Path,
    ) -> None:
        """Distribute checkpoint to DHT."""
        model_path = checkpoint_path / "model.pt"

        if not model_path.exists():
            return

        # Read and chunk the model file
        with open(model_path, "rb") as f:
            data = f.read()

        total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

        for i in range(total_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, len(data))
            chunk = data[start:end]

            await self.dht_manager.store_checkpoint(
                checkpoint_id=checkpoint_id,
                checkpoint_data=chunk,
                chunk_index=i,
                total_chunks=total_chunks,
            )

        logger.info(f"Distributed checkpoint {checkpoint_id} in {total_chunks} chunks")

    async def _retrieve_checkpoint(self, checkpoint_id: str) -> bool:
        """Retrieve checkpoint from DHT."""
        if not self.dht_manager:
            return False

        data = await self.dht_manager.get_checkpoint(checkpoint_id)

        if data is None:
            return False

        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_path = checkpoint_path / "model.pt"
        with open(model_path, "wb") as f:
            f.write(data)

        logger.info(f"Retrieved checkpoint {checkpoint_id} from DHT")
        return True

    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay under limit."""
        checkpoints = []

        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        checkpoints.append((path, data["timestamp"]))

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by timestamp (oldest first)
        checkpoints.sort(key=lambda x: x[1])

        # Remove oldest
        to_remove = len(checkpoints) - self.max_checkpoints
        for i in range(to_remove):
            path, _ = checkpoints[i]
            import shutil
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path.name}")

    def _generate_checkpoint_id(
        self,
        step: int,
        policy_version: int,
        node_id: str,
    ) -> str:
        """Generate unique checkpoint ID."""
        content = f"{step}:{policy_version}:{node_id}:{time.time()}"
        hash_str = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"ckpt_step{step}_v{policy_version}_{hash_str}"

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []

        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        checkpoints.append(CheckpointMetadata(**data))

        # Sort by step
        checkpoints.sort(key=lambda x: x.step, reverse=True)
        return checkpoints
