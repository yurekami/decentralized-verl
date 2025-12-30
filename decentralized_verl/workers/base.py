"""Base worker class for decentralized model serving."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn

from decentralized_verl.core.config import DecentralizedConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    total_requests: int = 0
    total_tokens_processed: int = 0
    avg_latency: float = 0.0
    errors: int = 0
    uptime: float = 0.0
    start_time: float = field(default_factory=time.time)


class BaseWorker(ABC):
    """
    Base class for model workers in decentralized network.

    Handles common functionality like model loading, device management,
    and statistics tracking.
    """

    def __init__(
        self,
        config: DecentralizedConfig,
        block_indices: Optional[List[int]] = None,
    ):
        """
        Initialize base worker.

        Args:
            config: Configuration
            block_indices: Transformer block indices this worker serves
        """
        self.config = config
        self.block_indices = block_indices or []

        self.model: Optional[nn.Module] = None
        self.tokenizer: Any = None
        self.device: torch.device = torch.device("cpu")

        self.stats = WorkerStats()
        self._initialized = False
        self._lock = asyncio.Lock()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the worker and load model."""
        pass

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Process a request."""
        pass

    async def _load_model(self) -> nn.Module:
        """Load the model based on configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_config = self.config.model

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Loading model {model_config.model_name_or_path} on {self.device}")

        # Quantization config
        quantization_config = None
        if model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif model_config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(model_config.dtype, torch.bfloat16)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Apply LoRA if configured
        if model_config.use_lora:
            model = self._apply_lora(model, model_config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model
        self._initialized = True

        logger.info(f"Model loaded successfully: {model_config.model_name_or_path}")
        return model

    def _apply_lora(
        self,
        model: nn.Module,
        config: ModelConfig,
    ) -> nn.Module:
        """Apply LoRA to model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            logger.info(f"Applied LoRA with rank={config.lora_rank}")
            return model

        except ImportError:
            logger.warning("peft not installed, skipping LoRA")
            return model

    async def get_checkpoint(self) -> Dict[str, Any]:
        """Get model checkpoint."""
        if self.model is None:
            return {}

        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "block_indices": self.block_indices,
        }

    async def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load model checkpoint."""
        if self.model is None:
            await self.initialize()

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model checkpoint")

    async def apply_gradient(self, gradients: Dict[str, Any]) -> None:
        """Apply gradient update to model."""
        if self.model is None:
            return

        # Decompress and apply gradients
        for name, param in self.model.named_parameters():
            if name in gradients and param.requires_grad:
                grad_data = gradients[name]
                if isinstance(grad_data, bytes):
                    import numpy as np
                    grad_tensor = torch.from_numpy(
                        np.frombuffer(grad_data, dtype=np.float32)
                    ).reshape(param.shape)
                else:
                    grad_tensor = grad_data

                param.grad = grad_tensor.to(param.device)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        self.stats.uptime = time.time() - self.stats.start_time
        return {
            "total_requests": self.stats.total_requests,
            "total_tokens": self.stats.total_tokens_processed,
            "avg_latency": self.stats.avg_latency,
            "errors": self.stats.errors,
            "uptime": self.stats.uptime,
            "initialized": self._initialized,
            "device": str(self.device),
            "block_indices": self.block_indices,
        }
