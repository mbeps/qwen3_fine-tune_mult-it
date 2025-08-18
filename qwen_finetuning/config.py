from enum import Enum


class ThinkingMode(Enum):
    """
    Enumeration for thinking mode options.
    DISABLED: No thinking tokens (default, backwards compatible)
    ENABLED: All examples use thinking tokens
    MIXED: Mix of thinking and non-thinking examples
    """

    DISABLED = "disabled"
    ENABLED = "enabled"
    MIXED = "mixed"


class QwenFineTuningConfig:
    """
    Configuration class for Qwen fine-tuning.
    Sets all hyperparameters and options for training, including model, data, LoRA, and thinking mode.
    Adjusts batch size, learning rate, and max_length based on thinking mode for optimal training.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results",
        batch_size: int = 12,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 4e-5,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        num_epochs: int = 2,
        max_length: int = 512,
        lora_r: int = 24,
        lora_alpha: int = 48,
        lora_dropout: float = 0.1,
        target_modules: list | None = None,
        dataloader_num_workers: int = 4,
        gradient_checkpointing: bool = False,
        thinking_mode: ThinkingMode = ThinkingMode.DISABLED,
    ) -> None:
        """
        Initialise the configuration for Qwen fine-tuning.
        Adjusts batch size, learning rate, and max_length if thinking mode is enabled or mixed.
        Args:
            model_name (str): Model name or path.
            train_file (str): Path to training data file.
            output_dir (str): Directory to save results.
            batch_size (int): Batch size per device.
            gradient_accumulation_steps (int): Gradient accumulation steps.
            learning_rate (float): Learning rate.
            warmup_ratio (float): Warmup ratio for scheduler.
            lr_scheduler_type (str): Scheduler type.
            num_epochs (int): Number of epochs.
            max_length (int): Max sequence length.
            lora_r (int): LoRA rank.
            lora_alpha (int): LoRA alpha.
            lora_dropout (float): LoRA dropout.
            target_modules (list|None): Target modules for LoRA.
            dataloader_num_workers (int): DataLoader workers.
            gradient_checkpointing (bool): Enable gradient checkpointing.
            thinking_mode (ThinkingMode): Thinking mode option.
        """
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

        if thinking_mode in [ThinkingMode.ENABLED, ThinkingMode.MIXED]:
            self.batch_size: int = max(1, batch_size // 2)
            self.learning_rate: float = learning_rate * 0.8
            if max_length == 512:
                self.max_length: int = 2048
            else:
                self.max_length: int = max_length
        else:
            self.batch_size: int = batch_size
            self.learning_rate: float = learning_rate
            self.max_length: int = max_length

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
        """
        Calculate the effective batch size (batch_size * gradient_accumulation_steps).
        Returns:
            int: Effective batch size.
        """
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def enable_thinking(self) -> bool:
        """
        Backwards compatibility property for enabling thinking mode.
        Returns:
            bool: True if thinking mode is enabled or mixed, False otherwise.
        """
        return self.thinking_mode != ThinkingMode.DISABLED

    def print_config(self) -> None:
        """
        Print a summary of the current configuration for inspection.
        """
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size} (effective: {self.effective_batch_size})")
        print(
            f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}"
        )
        print(f"Thinking mode: {self.thinking_mode.value}")
        print(f"Max length: {self.max_length}")
