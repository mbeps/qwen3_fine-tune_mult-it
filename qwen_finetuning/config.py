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
        enable_thinking: bool = False,  # Enable Chain-of-Thought training
    ) -> None:
        self.model_name: str = model_name
        self.train_file: str = train_file
        self.output_dir: str = output_dir
        self.batch_size: int = batch_size
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.learning_rate: float = learning_rate
        self.warmup_ratio: float = warmup_ratio
        self.lr_scheduler_type: str = lr_scheduler_type
        self.num_epochs: int = num_epochs
        self.lora_r: int = lora_r
        self.lora_alpha: int = lora_alpha
        self.lora_dropout: float = lora_dropout
        self.dataloader_num_workers: int = dataloader_num_workers
        self.gradient_checkpointing: bool = gradient_checkpointing
        self.enable_thinking: bool = enable_thinking

        # Adjust max_length for thinking mode - Qwen3 recommends longer sequences for reasoning
        if enable_thinking and max_length == 512:  # Only auto-adjust if using default
            self.max_length: int = 2048
        else:
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

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size} (effective: {self.effective_batch_size})")
        print(
            f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}"
        )
        print(f"Thinking mode: {'Enabled' if self.enable_thinking else 'Disabled'}")
        print(f"Max length: {self.max_length}")
