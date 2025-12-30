"""Experience buffer and data structures for RLHF training."""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import time
import threading


@dataclass
class Experience:
    """Single experience from RLHF rollout."""

    # Core data
    prompt: str
    response: str
    reward: float = 0.0

    # Log probabilities
    log_probs: List[float] = field(default_factory=list)
    ref_log_probs: List[float] = field(default_factory=list)

    # Values (from critic)
    values: List[float] = field(default_factory=list)

    # Computed advantages
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    # Metadata
    policy_version: int = 0
    timestamp: float = field(default_factory=time.time)
    prompt_id: str = ""
    response_id: str = ""
    worker_id: str = ""

    # Token information
    prompt_tokens: List[int] = field(default_factory=list)
    response_tokens: List[int] = field(default_factory=list)

    # Additional info
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt_id:
            import hashlib
            self.prompt_id = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        if not self.response_id:
            import hashlib
            self.response_id = hashlib.sha256(
                f"{self.prompt}:{self.response}".encode()
            ).hexdigest()[:8]

    @property
    def kl_divergence(self) -> float:
        """Compute KL divergence between policy and reference."""
        if not self.log_probs or not self.ref_log_probs:
            return 0.0

        kl = 0.0
        for log_p, ref_log_p in zip(self.log_probs, self.ref_log_probs):
            kl += log_p - ref_log_p

        return kl / max(len(self.log_probs), 1)

    @property
    def response_length(self) -> int:
        """Get response length in tokens."""
        if self.response_tokens:
            return len(self.response_tokens)
        return len(self.response.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "log_probs": self.log_probs,
            "ref_log_probs": self.ref_log_probs,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
            "policy_version": self.policy_version,
            "timestamp": self.timestamp,
            "prompt_id": self.prompt_id,
            "response_id": self.response_id,
            "worker_id": self.worker_id,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create from dictionary."""
        return cls(**data)


class ExperienceBuffer:
    """
    Buffer for storing RLHF experiences.

    Supports on-policy (fresh experiences only) and off-policy
    (replay buffer) training modes.
    """

    def __init__(
        self,
        max_size: int = 10000,
        min_size: int = 1000,
        on_policy: bool = True,
    ):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum buffer capacity
            min_size: Minimum samples before training
            on_policy: If True, clear buffer after each training batch
        """
        self.max_size = max_size
        self.min_size = min_size
        self.on_policy = on_policy

        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # Statistics
        self._total_added = 0
        self._total_sampled = 0

    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        with self._lock:
            self._buffer.append(experience)
            self._total_added += 1

    def add_batch(self, experiences: List[Experience]) -> None:
        """Add batch of experiences."""
        with self._lock:
            for exp in experiences:
                self._buffer.append(exp)
                self._total_added += 1

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        with self._lock:
            if len(self._buffer) < batch_size:
                batch = list(self._buffer)
            else:
                batch = random.sample(list(self._buffer), batch_size)

            self._total_sampled += len(batch)

            if self.on_policy:
                # Clear sampled experiences
                for exp in batch:
                    try:
                        self._buffer.remove(exp)
                    except ValueError:
                        pass

            return batch

    def get_all(self) -> List[Experience]:
        """Get all experiences (for on-policy training)."""
        with self._lock:
            experiences = list(self._buffer)
            if self.on_policy:
                self._buffer.clear()
            return experiences

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= self.min_size

    def filter_by_version(self, policy_version: int) -> List[Experience]:
        """Get experiences from specific policy version."""
        with self._lock:
            return [
                exp for exp in self._buffer
                if exp.policy_version == policy_version
            ]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Experience]:
        with self._lock:
            return iter(list(self._buffer))

    @property
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if not self._buffer:
                return {
                    "size": 0,
                    "max_size": self.max_size,
                    "total_added": self._total_added,
                    "total_sampled": self._total_sampled,
                }

            rewards = [exp.reward for exp in self._buffer]
            return {
                "size": len(self._buffer),
                "max_size": self.max_size,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "avg_reward": sum(rewards) / len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
            }


class BatchIterator:
    """Iterator for creating training batches from experiences."""

    def __init__(
        self,
        experiences: List[Experience],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize batch iterator.

        Args:
            experiences: Experiences to iterate over
            batch_size: Batch size
            shuffle: Whether to shuffle experiences
            drop_last: Whether to drop last incomplete batch
        """
        self.experiences = experiences
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._indices: List[int] = []
        self._position = 0
        self._reset()

    def _reset(self) -> None:
        """Reset iterator."""
        self._indices = list(range(len(self.experiences)))
        if self.shuffle:
            random.shuffle(self._indices)
        self._position = 0

    def __iter__(self) -> "BatchIterator":
        self._reset()
        return self

    def __next__(self) -> List[Experience]:
        if self._position >= len(self._indices):
            raise StopIteration

        end = min(self._position + self.batch_size, len(self._indices))

        if self.drop_last and end - self._position < self.batch_size:
            raise StopIteration

        batch_indices = self._indices[self._position:end]
        batch = [self.experiences[i] for i in batch_indices]
        self._position = end

        return batch

    def __len__(self) -> int:
        n_batches = len(self.experiences) // self.batch_size
        if not self.drop_last and len(self.experiences) % self.batch_size != 0:
            n_batches += 1
        return n_batches


def collate_experiences(
    experiences: List[Experience],
    tokenizer: Any = None,
    max_length: int = 2048,
    padding: bool = True,
) -> Dict[str, Any]:
    """
    Collate experiences into training batch.

    Args:
        experiences: List of experiences
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        padding: Whether to pad sequences

    Returns:
        Collated batch dictionary
    """
    import torch

    batch = {
        "prompts": [exp.prompt for exp in experiences],
        "responses": [exp.response for exp in experiences],
        "rewards": torch.tensor([exp.reward for exp in experiences]),
    }

    # Add log probs if available
    if experiences[0].log_probs:
        # Pad log probs to same length
        max_len = max(len(exp.log_probs) for exp in experiences)
        padded_log_probs = []
        for exp in experiences:
            padded = exp.log_probs + [0.0] * (max_len - len(exp.log_probs))
            padded_log_probs.append(padded)
        batch["old_log_probs"] = torch.tensor(padded_log_probs)

    # Add reference log probs
    if experiences[0].ref_log_probs:
        max_len = max(len(exp.ref_log_probs) for exp in experiences)
        padded_ref_log_probs = []
        for exp in experiences:
            padded = exp.ref_log_probs + [0.0] * (max_len - len(exp.ref_log_probs))
            padded_ref_log_probs.append(padded)
        batch["ref_log_probs"] = torch.tensor(padded_ref_log_probs)

    # Add advantages if available
    if experiences[0].advantages:
        max_len = max(len(exp.advantages) for exp in experiences)
        padded_advantages = []
        for exp in experiences:
            padded = exp.advantages + [0.0] * (max_len - len(exp.advantages))
            padded_advantages.append(padded)
        batch["advantages"] = torch.tensor(padded_advantages)

    # Add returns if available
    if experiences[0].returns:
        max_len = max(len(exp.returns) for exp in experiences)
        padded_returns = []
        for exp in experiences:
            padded = exp.returns + [0.0] * (max_len - len(exp.returns))
            padded_returns.append(padded)
        batch["returns"] = torch.tensor(padded_returns)

    # Tokenize if tokenizer provided
    if tokenizer is not None:
        # Encode prompts
        prompt_encodings = tokenizer(
            batch["prompts"],
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        batch["prompt_input_ids"] = prompt_encodings["input_ids"]
        batch["prompt_attention_mask"] = prompt_encodings["attention_mask"]

        # Encode full sequences (prompt + response)
        full_texts = [
            p + r for p, r in zip(batch["prompts"], batch["responses"])
        ]
        full_encodings = tokenizer(
            full_texts,
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        batch["input_ids"] = full_encodings["input_ids"]
        batch["attention_mask"] = full_encodings["attention_mask"]

    return batch
