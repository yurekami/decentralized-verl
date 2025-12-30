"""Reference worker for KL divergence computation."""

import logging
from typing import Any, Dict, List, Optional
import time

import torch

from decentralized_verl.core.config import DecentralizedConfig
from decentralized_verl.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ReferenceWorker(BaseWorker):
    """
    Reference worker for computing reference policy log probabilities.

    Used for KL divergence penalty in RLHF training.
    The reference model is typically frozen and represents the initial policy.
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        block_indices: Optional[List[int]] = None,
    ):
        """Initialize reference worker."""
        super().__init__(config, block_indices)

    async def initialize(self) -> None:
        """Initialize reference worker."""
        await self._load_model()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self._initialized = True
        logger.info("Reference worker initialized (frozen)")

    async def process(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs,
    ) -> List[List[float]]:
        """Process log probability request."""
        return await self.get_log_probs(prompts, responses)

    async def get_log_probs(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[List[float]]:
        """
        Get reference log probabilities for prompt-response pairs.

        Args:
            prompts: Input prompts
            responses: Generated responses

        Returns:
            List of log probability sequences
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")

        async with self._lock:
            start_time = time.time()
            all_log_probs = []

            for prompt, response in zip(prompts, responses):
                log_probs = self._compute_log_probs(prompt, response)
                all_log_probs.append(log_probs)

            # Update stats
            latency = time.time() - start_time
            self.stats.total_requests += len(prompts)
            self.stats.avg_latency = 0.9 * self.stats.avg_latency + 0.1 * latency

            return all_log_probs

    def _compute_log_probs(
        self,
        prompt: str,
        response: str,
    ) -> List[float]:
        """Compute log probs for single prompt-response pair."""
        # Encode full sequence
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prompt length to know where response starts
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Compute log probs for response tokens
        log_probs = []
        for i in range(prompt_len - 1, inputs["input_ids"].shape[1] - 1):
            token_id = inputs["input_ids"][0, i + 1].item()
            log_prob = torch.log_softmax(logits[0, i], dim=-1)[token_id].item()
            log_probs.append(log_prob)

        return log_probs

    async def get_log_probs_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_indices: List[int],
    ) -> torch.Tensor:
        """
        Get reference log probabilities for a batch.

        Args:
            input_ids: Encoded sequences [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start_indices: Start index of response for each sequence

        Returns:
            Log probabilities tensor [batch, max_response_len]
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        batch_size, seq_len, vocab_size = logits.shape

        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probs for actual tokens (shifted by 1)
        token_log_probs = log_probs[:, :-1].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # Mask out prompt tokens
        max_response_len = max(
            seq_len - idx for idx in response_start_indices
        )
        response_log_probs = torch.zeros(
            batch_size, max_response_len,
            device=input_ids.device,
        )

        for i, start_idx in enumerate(response_start_indices):
            response_len = seq_len - start_idx - 1
            if response_len > 0:
                response_log_probs[i, :response_len] = token_log_probs[i, start_idx - 1:start_idx + response_len - 1]

        return response_log_probs

    def compute_kl_divergence(
        self,
        policy_log_probs: List[float],
        reference_log_probs: List[float],
    ) -> float:
        """
        Compute KL divergence between policy and reference.

        Args:
            policy_log_probs: Log probs from policy
            reference_log_probs: Log probs from reference

        Returns:
            KL divergence value
        """
        if not policy_log_probs or not reference_log_probs:
            return 0.0

        # KL(policy || reference) = sum(policy * (log(policy) - log(reference)))
        # = sum(exp(log_policy) * (log_policy - log_reference))
        # Approximation: sum(log_policy - log_reference)
        kl = 0.0
        min_len = min(len(policy_log_probs), len(reference_log_probs))

        for i in range(min_len):
            kl += policy_log_probs[i] - reference_log_probs[i]

        return kl / max(min_len, 1)
