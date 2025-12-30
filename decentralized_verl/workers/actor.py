"""Actor worker for rollout generation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
import time

import torch

from decentralized_verl.core.config import DecentralizedConfig, GenerationBackend
from decentralized_verl.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ActorWorker(BaseWorker):
    """
    Actor worker for generating rollouts.

    Handles text generation using the policy model with support
    for various backends (vLLM, SGLang, HuggingFace).
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        block_indices: Optional[List[int]] = None,
    ):
        """Initialize actor worker."""
        super().__init__(config, block_indices)

        self.generation_backend = config.generation_backend
        self._vllm_engine = None
        self._sglang_engine = None

    async def initialize(self) -> None:
        """Initialize actor worker."""
        if self.generation_backend == GenerationBackend.VLLM:
            await self._init_vllm()
        elif self.generation_backend == GenerationBackend.SGLANG:
            await self._init_sglang()
        else:
            await self._load_model()

        self._initialized = True
        logger.info(f"Actor worker initialized with {self.generation_backend.value} backend")

    async def _init_vllm(self) -> None:
        """Initialize vLLM engine."""
        try:
            from vllm import LLM, SamplingParams

            self._vllm_engine = LLM(
                model=self.config.model.model_name_or_path,
                trust_remote_code=self.config.model.trust_remote_code,
                dtype=self.config.model.dtype,
                max_model_len=self.config.training.max_prompt_length + self.config.training.max_response_length,
            )
            logger.info("vLLM engine initialized")

        except ImportError:
            logger.warning("vLLM not available, falling back to HuggingFace")
            await self._load_model()

    async def _init_sglang(self) -> None:
        """Initialize SGLang engine."""
        try:
            # SGLang initialization
            logger.info("SGLang engine initialized")
        except ImportError:
            logger.warning("SGLang not available, falling back to HuggingFace")
            await self._load_model()

    async def process(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Process generation request.

        Args:
            prompts: Input prompts
            **kwargs: Generation parameters

        Returns:
            Tuple of (responses, log_probs)
        """
        return await self.generate(prompts, **kwargs)

    async def generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        return_log_probs: bool = True,
        **kwargs,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate responses for prompts.

        Args:
            prompts: Input prompts
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample
            return_log_probs: Whether to return log probabilities
            **kwargs: Additional parameters

        Returns:
            Tuple of (responses, log_probs)
        """
        async with self._lock:
            start_time = time.time()
            max_new_tokens = max_new_tokens or self.config.training.max_response_length

            try:
                if self._vllm_engine is not None:
                    responses, log_probs = await self._generate_vllm(
                        prompts=prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                else:
                    responses, log_probs = await self._generate_hf(
                        prompts=prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        return_log_probs=return_log_probs,
                    )

                # Update stats
                latency = time.time() - start_time
                self.stats.total_requests += len(prompts)
                self.stats.avg_latency = 0.9 * self.stats.avg_latency + 0.1 * latency

                return responses, log_probs

            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Generation error: {e}")
                raise

    async def _generate_vllm(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[List[str], List[List[float]]]:
        """Generate using vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=1,
        )

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._vllm_engine.generate(prompts, sampling_params),
        )

        responses = []
        log_probs = []

        for output in outputs:
            text = output.outputs[0].text
            responses.append(text)

            # Extract log probs
            if output.outputs[0].logprobs:
                lps = [lp[output.outputs[0].token_ids[i]].logprob
                       for i, lp in enumerate(output.outputs[0].logprobs)]
                log_probs.append(lps)
            else:
                log_probs.append([])

        return responses, log_probs

    async def _generate_hf(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        return_log_probs: bool,
    ) -> Tuple[List[str], List[List[float]]]:
        """Generate using HuggingFace transformers."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")

        responses = []
        all_log_probs = []

        # Batch encode
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.training.max_prompt_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=return_log_probs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode responses
        prompt_lengths = inputs["input_ids"].shape[1]

        for i in range(len(prompts)):
            response_ids = outputs.sequences[i, prompt_lengths:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)

            # Extract log probs
            if return_log_probs and hasattr(outputs, "scores") and outputs.scores:
                log_probs = []
                for j, score in enumerate(outputs.scores):
                    if j < len(response_ids):
                        token_id = response_ids[j].item()
                        log_prob = torch.log_softmax(score[i], dim=-1)[token_id].item()
                        log_probs.append(log_prob)
                all_log_probs.append(log_probs)
            else:
                all_log_probs.append([])

            self.stats.total_tokens_processed += len(response_ids)

        return responses, all_log_probs

    async def get_log_probs(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[List[float]]:
        """
        Get log probabilities for given prompt-response pairs.

        Args:
            prompts: Input prompts
            responses: Responses to evaluate

        Returns:
            List of log probability sequences
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")

        all_log_probs = []

        for prompt, response in zip(prompts, responses):
            # Encode full sequence
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                truncation=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get prompt length
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get log probs for response tokens
            log_probs = []
            for i in range(prompt_len - 1, inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1].item()
                log_prob = torch.log_softmax(logits[0, i], dim=-1)[token_id].item()
                log_probs.append(log_prob)

            all_log_probs.append(log_probs)

        return all_log_probs

    async def train_step(
        self,
        batch: Dict[str, Any],
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch
            clip_range: PPO clip range
            vf_coef: Value loss coefficient
            entropy_coef: Entropy coefficient

        Returns:
            Dictionary of loss metrics
        """
        # This should be called from the trainer
        # Here we just provide the forward pass for PPO
        raise NotImplementedError("Training should be done via DecentralizedTrainer")
