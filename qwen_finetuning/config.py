from enum import Enum


class ThinkingMode(Enum):
    """Enumeration for thinking mode options."""

    DISABLED = "disabled"  # No thinking tokens (default, backwards compatible)
    ENABLED = "enabled"  # All examples use thinking tokens
    MIXED = "mixed"  # Mix of thinking and non-thinking examples


class QwenFineTuningConfig:
    """
    Configuration for Qwen fine-tuning.
    Optimised for quality over speed, matching successful QLoRA parameters.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results",
        batch_size: int = 12,  # Conservative for better convergence
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 4e-5,  # Slightly lower for full precision
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        num_epochs: int = 2,
        max_length: int = 512,
        lora_r: int = 24,  # Balanced capacity
        lora_alpha: int = 48,  # 2*r
        lora_dropout: float = 0.1,  # More regularisation without quantisation
        target_modules: list | None = None,
        dataloader_num_workers: int = 4,
        gradient_checkpointing: bool = False,
        thinking_mode: ThinkingMode = ThinkingMode.DISABLED,  # Backwards compatible default
    ) -> None:
        self.model_name: str = model_name
        self.train_file: str = train_file
        self.output_dir: str = output_dir
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.warmup_ratio: float = warmup_ratio
        self.lr_scheduler_type: str = lr_scheduler_type
        self.num_epochs: int = num_epochs
        self.lora_r: int = lora_r
        self.lora_alpha: int = lora_alpha
        self.lora_dropout: float = lora_dropout
        self.dataloader_num_workers: int = dataloader_num_workers
        self.gradient_checkpointing: bool = gradient_checkpointing
        self.thinking_mode: ThinkingMode = thinking_mode

        # Adjust hyperparameters based on thinking mode
        if thinking_mode in [ThinkingMode.ENABLED, ThinkingMode.MIXED]:
            # Qwen3 recommendations for thinking mode training
            self.batch_size: int = max(
                1, batch_size // 2
            )  # Reduce batch size for longer sequences
            self.learning_rate: float = (
                learning_rate * 0.8
            )  # Slightly lower LR for stability
            # Auto-adjust max_length for thinking mode if using default
            if max_length == 512:
                self.max_length: int = 2048  # Qwen3 recommended for thinking
            else:
                self.max_length: int = max_length
        else:
            # Standard settings for non-thinking mode
            self.batch_size: int = batch_size
            self.learning_rate: float = learning_rate
            self.max_length: int = max_length

        # Standard target modules for Qwen/Llama architecture
        if target_modules is None:
            self.target_modules: list[str] = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            self.target_modules = target_modules

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def enable_thinking(self) -> bool:
        """Backwards compatibility property."""
        return self.thinking_mode != ThinkingMode.DISABLED

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size} (effective: {self.effective_batch_size})")
        print(
            f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}"
        )
        print(f"Thinking mode: {self.thinking_mode.value}")
        print(f"Max length: {self.max_length}")
