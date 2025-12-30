"""Reward worker for computing rewards."""

import logging
from typing import Any, Callable, Dict, List, Optional
import time

import torch
import torch.nn as nn

from decentralized_verl.core.config import DecentralizedConfig
from decentralized_verl.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class RewardWorker(BaseWorker):
    """
    Reward worker for computing rewards.

    Supports both model-based rewards and function-based (verifiable) rewards.
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        reward_function: Optional[Callable] = None,
    ):
        """
        Initialize reward worker.

        Args:
            config: Configuration
            reward_function: Optional callable for verifiable rewards
        """
        super().__init__(config)
        self.reward_function = reward_function
        self.use_model = config.algorithm.reward_model_name is not None

    async def initialize(self) -> None:
        """Initialize reward worker."""
        if self.use_model and self.config.algorithm.reward_model_name:
            # Load reward model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = self.config.algorithm.reward_model_name

            logger.info(f"Loading reward model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
            else:
                self.device = torch.device("cpu")

            self.model.eval()

        self._initialized = True
        logger.info("Reward worker initialized")

    async def process(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs,
    ) -> List[float]:
        """Process reward computation request."""
        return await self.compute_rewards(prompts, responses)

    async def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Compute rewards for prompt-response pairs.

        Args:
            prompts: Input prompts
            responses: Generated responses

        Returns:
            List of reward scores
        """
        async with self._lock:
            start_time = time.time()

            if self.reward_function is not None:
                # Use verifiable reward function
                rewards = self._compute_function_rewards(prompts, responses)
            elif self.model is not None:
                # Use reward model
                rewards = self._compute_model_rewards(prompts, responses)
            else:
                # Default: simple length-based reward
                rewards = [len(r.split()) / 100.0 for r in responses]

            # Apply scaling and clipping
            rewards = [r * self.config.algorithm.reward_scale for r in rewards]

            if self.config.algorithm.reward_clip is not None:
                clip = self.config.algorithm.reward_clip
                rewards = [max(-clip, min(clip, r)) for r in rewards]

            # Update stats
            latency = time.time() - start_time
            self.stats.total_requests += len(prompts)
            self.stats.avg_latency = 0.9 * self.stats.avg_latency + 0.1 * latency

            return rewards

    def _compute_function_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """Compute rewards using verifiable function."""
        rewards = []
        for prompt, response in zip(prompts, responses):
            try:
                reward = self.reward_function(prompt, response)
                rewards.append(float(reward))
            except Exception as e:
                logger.warning(f"Reward function error: {e}")
                rewards.append(0.0)
        return rewards

    def _compute_model_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """Compute rewards using reward model."""
        rewards = []

        for prompt, response in zip(prompts, responses):
            full_text = prompt + response

            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.training.max_prompt_length + self.config.training.max_response_length,
                truncation=True,
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Handle different output formats
                if hasattr(outputs, "logits"):
                    # Classification model
                    if outputs.logits.shape[-1] == 1:
                        reward = outputs.logits.squeeze().item()
                    else:
                        # Multi-class: use positive class probability
                        probs = torch.softmax(outputs.logits, dim=-1)
                        reward = probs[0, -1].item()  # Last class as "good"
                else:
                    reward = outputs[0].item()

                rewards.append(reward)

        return rewards


class VerifiableRewardFunctions:
    """Collection of verifiable reward functions."""

    @staticmethod
    def math_correctness(prompt: str, response: str) -> float:
        """
        Reward for math problem correctness.

        Extracts answer from response and compares to expected answer.
        """
        import re

        # Extract expected answer from prompt (e.g., "Answer: 42")
        expected_match = re.search(r"Answer:\s*(\d+)", prompt)
        if not expected_match:
            return 0.0

        expected = expected_match.group(1)

        # Extract answer from response
        # Look for boxed answer or final number
        boxed_match = re.search(r"\\boxed{([^}]+)}", response)
        if boxed_match:
            answer = boxed_match.group(1).strip()
        else:
            # Find last number in response
            numbers = re.findall(r"\d+", response)
            answer = numbers[-1] if numbers else ""

        return 1.0 if answer == expected else 0.0

    @staticmethod
    def code_execution(prompt: str, response: str) -> float:
        """
        Reward for code correctness via execution.

        Extracts code from response and checks if it runs without errors.
        """
        import re

        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)

        if not code_match:
            return 0.0

        code = code_match.group(1)

        try:
            # Execute in restricted environment
            exec(compile(code, "<string>", "exec"), {"__builtins__": {}})
            return 1.0
        except:
            return 0.0

    @staticmethod
    def length_penalty(prompt: str, response: str, target_length: int = 100) -> float:
        """
        Reward based on response length.

        Penalizes responses that are too short or too long.
        """
        length = len(response.split())
        deviation = abs(length - target_length) / target_length
        return max(0.0, 1.0 - deviation)

    @staticmethod
    def format_adherence(prompt: str, response: str, required_sections: List[str] = None) -> float:
        """
        Reward for following required format.

        Checks if response contains required sections.
        """
        if required_sections is None:
            required_sections = []

        if not required_sections:
            return 1.0

        found = sum(1 for section in required_sections if section.lower() in response.lower())
        return found / len(required_sections)
