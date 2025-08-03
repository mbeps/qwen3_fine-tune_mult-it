class QwenFineTuningConfig:
    """Configuration class for Qwen fine-tuning experiments."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results_clean",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dataset_num_proc: int = 4,
        cache_writer_batch_size: int = 500,
        dataloader_num_workers: int = 4,
        dataloader_pin_memory: bool = True,
        dataloader_persistent_workers: bool = True,
        torch_empty_cache_steps: int = 4,
    ):
        self.model_name = model_name
        self.train_file = train_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.dataset_num_proc = dataset_num_proc
        self.cache_writer_batch_size = cache_writer_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.torch_empty_cache_steps = torch_empty_cache_steps

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps

    def print_config(self):
        """Print current configuration."""
        print(f"✓ Configuration set")
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Effective batch size: {self.effective_batch_size}")
        print(f"Dataset processing cores: {self.dataset_num_proc}")
        print(f"Cache writer batch size: {self.cache_writer_batch_size}")
        print(f"DataLoader workers: {self.dataloader_num_workers}")
        print(
            f"DataLoader optimizations: pin_memory={self.dataloader_pin_memory}, persistent_workers={self.dataloader_persistent_workers}"
        )
        print(f"GPU cache management: empty every {self.torch_empty_cache_steps} steps")
