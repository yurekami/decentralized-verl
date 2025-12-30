"""Critic worker for value estimation."""

import logging
from typing import Any, Dict, List, Optional
import time

import torch
import torch.nn as nn

from decentralized_verl.core.config import DecentralizedConfig
from decentralized_verl.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class CriticWorker(BaseWorker):
    """
    Critic worker for estimating state values.

    Used for advantage computation in actor-critic algorithms.
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        block_indices: Optional[List[int]] = None,
    ):
        """Initialize critic worker."""
        super().__init__(config, block_indices)
        self.value_head: Optional[nn.Module] = None

    async def initialize(self) -> None:
        """Initialize critic worker."""
        await self._load_model()

        # Add value head if not present
        if not hasattr(self.model, "value_head"):
            hidden_size = self.config.model.hidden_size
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            ).to(self.device)
        else:
            self.value_head = self.model.value_head

        self._initialized = True
        logger.info("Critic worker initialized")

    async def process(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs,
    ) -> List[float]:
        """Process value estimation request."""
        return await self.get_values(prompts, responses)

    async def get_values(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Get value estimates for prompt-response pairs.

        Args:
            prompts: Input prompts
            responses: Responses

        Returns:
            List of value estimates
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")

        async with self._lock:
            start_time = time.time()
            values = []

            for prompt, response in zip(prompts, responses):
                full_text = prompt + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                    truncation=True,
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Get hidden states
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                    )

                    # Use last hidden state
                    hidden_states = outputs.hidden_states[-1]
                    last_hidden = hidden_states[:, -1, :]  # Last token

                    # Get value
                    if self.value_head is not None:
                        value = self.value_head(last_hidden).squeeze().item()
                    else:
                        value = 0.0

                    values.append(value)

            # Update stats
            latency = time.time() - start_time
            self.stats.total_requests += len(prompts)
            self.stats.avg_latency = 0.9 * self.stats.avg_latency + 0.1 * latency

            return values

    async def get_values_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimates for a batch of encoded inputs.

        Args:
            input_ids: Encoded input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Value estimates [batch]
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]

            # Get last non-padding token for each sequence
            batch_size = input_ids.shape[0]
            seq_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(batch_size, device=input_ids.device),
                seq_lengths,
            ]

            if self.value_head is not None:
                values = self.value_head(last_hidden).squeeze(-1)
            else:
                values = torch.zeros(batch_size, device=input_ids.device)

            return values
