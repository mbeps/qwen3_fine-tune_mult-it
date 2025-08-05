class QwenFineTuningConfig:
    """
    Configuration class for Qwen fine-tuning experiments.

    Args:
        model_name (str): Model name or path.
        train_file (str): Path to training data file.
        output_dir (str): Output directory for results.
        batch_size (int): Training batch size.
        gradient_accumulation_steps (int): Gradient accumulation steps.
        learning_rate (float): Learning rate.
        warmup_ratio (float): Warmup ratio for learning rate scheduling.
        lr_scheduler_type (str): Learning rate scheduler type.
        num_cycles (int): Number of cycles for cosine with restarts scheduler.
        num_epochs (int): Number of training epochs.
        max_length (int): Max sequence length.
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA alpha.
        lora_dropout (float): LoRA dropout.
        use_rslora (bool): Whether to use Rank-Stabilized LoRA.
        target_modules (list): Target modules for LoRA adaptation.
        dataset_num_proc (int): Number of processes for dataset formatting.
        cache_writer_batch_size (int): Batch size for cache writing.
        dataloader_num_workers (int): DataLoader worker count.
        dataloader_pin_memory (bool): Pin memory for DataLoader.
        dataloader_persistent_workers (bool): Persistent workers for DataLoader.
        torch_empty_cache_steps (int): Steps between GPU cache clearing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results_clean",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,  # NEW: Optimal warmup for large datasets
        lr_scheduler_type: str = "cosine_with_restarts",  # NEW: Better than linear
        num_cycles: int = 2,  # NEW: For cosine restarts
        num_epochs: int = 3,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,  # CHANGED: From 0.05 to 0.1 for better regularization
        use_rslora: bool = True,  # NEW: Rank-Stabilized LoRA for better performance
        target_modules: list = None,  # Will be set to specific modules in __init__
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
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.num_cycles = num_cycles
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        
        # Set specific target modules
        if target_modules is None:
            self.target_modules = [
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
            
        self.dataset_num_proc = dataset_num_proc
        self.cache_writer_batch_size = cache_writer_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.torch_empty_cache_steps = torch_empty_cache_steps

    @property
    def effective_batch_size(self) -> int:
        """
        Calculate effective batch size.

        Returns:
            int: Effective batch size.
        """
        return self.batch_size * self.gradient_accumulation_steps

    def print_config(self):
        """
        Print current configuration settings.
        """
        print(f"✓ Configuration set with optimizations")
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"LR scheduler: {self.lr_scheduler_type} (warmup: {self.warmup_ratio})")
        print(f"Batch size: {self.batch_size}")
        print(f"Effective batch size: {self.effective_batch_size}")
        print(f"LoRA optimizations:")
        print(f"  - RSLoRA enabled: {self.use_rslora}")
        print(f"  - Target modules: {self.target_modules}")
        print(f"  - Rank: {self.lora_r}, Alpha: {self.lora_alpha}, Dropout: {self.lora_dropout}")
        print(f"Dataset processing cores: {self.dataset_num_proc}")
        print(f"Cache writer batch size: {self.cache_writer_batch_size}")
        print(f"DataLoader workers: {self.dataloader_num_workers}")
        print(
            f"DataLoader optimizations: pin_memory={self.dataloader_pin_memory}, persistent_workers={self.dataloader_persistent_workers}"
        )
        print(f"GPU cache management: empty every {self.torch_empty_cache_steps} steps")