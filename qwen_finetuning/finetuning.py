import json
import torch
import re
import os
import hashlib
import psutil
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from dotenv import load_dotenv
from .config import QwenFineTuningConfig


# Global tokenizer cache for multiprocessing
_tokenizer_cache = {}


def _get_tokenizer(model_name: str):
    """
    Get or create a tokenizer for multiprocessing.

    Args:
        model_name (str): Model name or path.
    Returns:
        AutoTokenizer: Tokenizer instance.
    """
    if model_name not in _tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        _tokenizer_cache[model_name] = tokenizer
    return _tokenizer_cache[model_name]


def format_prompt_multiprocess(example, model_name: str):
    """
    Format a single example for training using chat template (multiprocessing).

    Args:
        example (dict): Data example with 'question', 'options', 'answer'.
        model_name (str): Model name or path.
    Returns:
        dict: Formatted text dict for dataset.
    """
    tokenizer = _get_tokenizer(model_name)

    question = example["question"]
    options = example["options"]
    answer = example["answer"]

    # Format options text
    options_text = "\n".join(
        [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
    )

    # Training format with answer
    messages = [
        {"role": "user", "content": f"Domanda: {question}\n\n{options_text}"},
        {"role": "assistant", "content": answer},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    return {"text": formatted_text}


class QwenFineTuning:
    """
    Main class for Qwen fine-tuning with LoRA and gradient clipping.

    Args:
        config (QwenFineTuningConfig): Configuration object.
    """

    def __init__(self, config: QwenFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._setup_environment()

    def _setup_environment(self):
        """
        Load environment variables and set Hugging Face token.
        Raises:
            ValueError: If HF_TOKEN is not found.
        """
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env file")

        print(f"✓ Environment loaded, HF token available")

    @staticmethod
    def _get_memory_usage():
        """
        Get current memory usage in GB.

        Returns:
            float: Used memory in GB.
        """
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.used / (1024**3)  # Convert to GB
        except:
            return 0.0

    @staticmethod
    def _format_memory(gb):
        """
        Format memory usage for display.

        Args:
            gb (float): Memory in GB.
        Returns:
            str: Formatted string.
        """
        return f"{gb:.1f}GB"

    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """
        Load data from a JSONL file.

        Args:
            file_path (str): Path to JSONL file.
        Returns:
            list: List of loaded examples.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def format_prompt(self, question: str, options: list, answer: str = None) -> str:
        """
        Format prompt using Qwen3 chat template for training or inference.

        Args:
            question (str): Question text.
            options (list): List of option dicts.
            answer (str, optional): Answer text. If None, formats for inference.
        Returns:
            str: Formatted prompt string.
        """
        options_text = "\n".join(
            [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
        )

        if answer is not None:
            # Training format
            messages = [
                {"role": "user", "content": f"Domanda: {question}\n\n{options_text}"},
                {"role": "assistant", "content": answer},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        else:
            # Inference format
            messages = [
                {"role": "user", "content": f"Domanda: {question}\n\n{options_text}"}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

    def _get_cache_path(self, data_file: str) -> str:
        """
        Generate cache path for processed dataset.

        Args:
            data_file (str): Path to source data file.
        Returns:
            str: Cache directory path.
        """
        # Get source file info
        source_path = Path(data_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {data_file}")

        # Create cache key from source file, modification time, and model
        source_mtime = str(source_path.stat().st_mtime)
        cache_components = [
            str(source_path.absolute()),
            source_mtime,
            self.config.model_name,
            "v1",  # Format version, increment if we change formatting logic
        ]

        # Generate hash
        cache_key = hashlib.md5("|".join(cache_components).encode()).hexdigest()

        # Create cache directory path
        cache_dir = Path("./cache/processed_datasets") / cache_key
        return str(cache_dir)

    def _is_cache_valid(self, cache_path: str, source_file: str) -> bool:
        """
        Check if cache is valid and up-to-date.

        Args:
            cache_path (str): Path to cache directory.
            source_file (str): Path to source file.
        Returns:
            bool: True if cache is valid, else False.
        """
        cache_dir = Path(cache_path)

        # Check if cache directory exists
        if not cache_dir.exists():
            return False

        # Check if cache contains required files
        if not (cache_dir / "dataset_info.json").exists():
            return False

        # Cache is valid if it exists and source file hasn't changed
        # (modification time is already included in cache path hash)
        return True

    def prepare_dataset_cached(self, data: list, data_file: str) -> Dataset:
        """
        Prepare dataset with memory-efficient caching optimization.

        Args:
            data (list): List of examples.
            data_file (str): Path to source data file.
        Returns:
            Dataset: HuggingFace Dataset object.
        """
        try:
            cache_path = self._get_cache_path(data_file)

            if self._is_cache_valid(cache_path, data_file):
                print(f"✓ Loading cached dataset from: {cache_path}")
                memory_before = self._get_memory_usage()

                # Load with memory-mapped access (default behavior)
                dataset = load_from_disk(cache_path)

                memory_after = self._get_memory_usage()
                print(
                    f"✓ Dataset loaded efficiently (memory: {self._format_memory(memory_before)} → {self._format_memory(memory_after)})"
                )
                return dataset
            else:
                print(f"✓ Processing and caching dataset with memory optimization...")
                memory_start = self._get_memory_usage()

                # Process dataset using existing method
                dataset = self.prepare_dataset(data)

                memory_after_processing = self._get_memory_usage()
                print(
                    f"✓ Processing complete (memory: {self._format_memory(memory_start)} → {self._format_memory(memory_after_processing)})"
                )

                # Save to cache with memory-efficient parameters
                cache_dir = Path(cache_path)
                cache_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"✓ Saving to cache with optimized batch size ({self.config.cache_writer_batch_size})..."
                )
                dataset.save_to_disk(
                    cache_path,
                    num_proc=1,  # Single process for cache writing to avoid memory pressure
                    num_shards=1,  # Single shard for simplicity
                )

                memory_final = self._get_memory_usage()
                print(f"✓ Dataset cached efficiently to: {cache_path}")
                print(
                    f"✓ Total memory usage: {self._format_memory(memory_start)} → {self._format_memory(memory_final)}"
                )

                return dataset

        except Exception as e:
            print(f"⚠ Caching failed ({e}), falling back to non-cached processing")
            return self.prepare_dataset(data)

    def prepare_dataset(self, data: list) -> Dataset:
        """
        Prepare dataset for training with multiprocessing and memory optimization.

        Args:
            data (list): List of examples.
        Returns:
            Dataset: Formatted HuggingFace Dataset.
        """
        dataset = Dataset.from_list(data)

        memory_before = self._get_memory_usage()
        print(
            f"Formatting dataset with {self.config.dataset_num_proc} processes (memory: {self._format_memory(memory_before)})..."
        )

        # Use multiprocessing for dataset formatting with memory-efficient batching
        formatted_dataset = dataset.map(
            format_prompt_multiprocess,
            fn_kwargs={"model_name": self.config.model_name},
            num_proc=self.config.dataset_num_proc,
            desc="Formatting with chat templates",
            remove_columns=dataset.column_names,  # Remove original columns, keep only 'text'
            writer_batch_size=self.config.cache_writer_batch_size,  # Memory-efficient batch size
        )

        memory_after = self._get_memory_usage()
        print(
            f"✓ Formatting complete (memory: {self._format_memory(memory_before)} → {self._format_memory(memory_after)})"
        )

        return formatted_dataset

    @staticmethod
    def analyze_data(data: list, name: str):
        """
        Analyse dataset distribution by category and answer.

        Args:
            data (list): List of examples.
            name (str): Dataset name for display.
        """
        categories = {}
        answers = {}

        for item in data:
            cat = item.get("category", "unknown")
            ans = item.get("answer", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            answers[ans] = answers.get(ans, 0) + 1

        print(f"{name} Dataset: {len(data)} examples")
        print(f"Categories: {', '.join(f'{k}({v})' for k, v in categories.items())}")
        print(
            f"Answer distribution: {', '.join(f'{k}({v})' for k, v in sorted(answers.items()))}"
        )

    def setup_model(self):
        """
        Load model and tokenizer, set up LoRA configuration.
        """
        print("Loading model and tokeniser...")

        # Load model with trust_remote_code for Qwen3
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=self.hf_token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, token=self.hf_token
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        print("Trainable parameters:")
        self.model.print_trainable_parameters()

    def setup_trainer(self, train_data: list):
        """
        Set up trainer for fine-tuning with optimized DataLoader configuration and gradient clipping.

        Args:
            train_data (list): List of training examples.
        """
        print(
            "Setting up trainer with optimized DataLoader configuration and gradient clipping..."
        )

        # Use cached dataset preparation
        train_dataset = self.prepare_dataset_cached(train_data, self.config.train_file)

        # Report optimization settings
        print(f"✓ Training optimizations enabled:")
        print(
            f"  - DataLoader workers: {self.config.dataloader_num_workers} (parallel data loading)"
        )
        print(
            f"  - Pin memory: {self.config.dataloader_pin_memory} (faster GPU transfer)"
        )
        print(
            f"  - Persistent workers: {self.config.dataloader_persistent_workers} (reduced startup overhead)"
        )
        print(
            f"  - GPU cache clearing: every {self.config.torch_empty_cache_steps} steps"
        )
        print(
            f"  - Gradient clipping: max_grad_norm={self.config.max_grad_norm} (training stability)"
        )

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,  # NEW: Gradient clipping
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            save_strategy="epoch",
            seed=42,
            bf16=True,
            max_length=self.config.max_length,
            packing=True,
            dataset_text_field="text",
            # DataLoader optimizations
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_persistent_workers=self.config.dataloader_persistent_workers,
            # Memory management optimization
            torch_empty_cache_steps=self.config.torch_empty_cache_steps,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
        )

        print(f"✓ Trainer configured with gradient clipping for improved stability")

    def train(self):
        """
        Start training using the configured trainer.
        Raises:
            ValueError: If trainer is not set up.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")

        print("Starting training with gradient clipping enabled...")
        self.trainer.train()

    def save_model(self):
        """
        Save the trained model.
        Raises:
            ValueError: If trainer is not available.
        """
        if self.trainer is None:
            raise ValueError("Trainer not available. Complete training first.")

        print("Saving model...")
        self.trainer.save_model()
        print("✓ Training completed")

    @staticmethod
    def extract_answer(output: str) -> str:
        """
        Extract answer (A-E) from model output string.

        Args:
            output (str): Model output string.
        Returns:
            str: Extracted answer letter or empty string.
        """
        if not output:
            return ""
        match = re.search(r"\b([ABCDE])\b", output.upper())
        return match.group(1) if match else ""

    def _format_options_text(self, options: list) -> str:
        """
        Format options text for evaluation.

        Args:
            options (list): List of option dicts.
        Returns:
            str: Formatted options string.
        """
        return "\n".join(
            [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
        )

    def evaluate_model(self, test_data: list) -> float:
        """
        Evaluate model accuracy on test data.

        Args:
            test_data (list): List of test examples.
        Returns:
            float: Accuracy score.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        correct = 0
        total = len(test_data)
        failed_extractions = 0

        for example in tqdm(test_data, desc="Evaluating"):
            # Use chat template for evaluation
            messages = [
                {
                    "role": "user",
                    "content": f"Domanda: {example['question']}\n\n{self._format_options_text(example['options'])}",
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs.to(self.model.device),
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            predicted = self.extract_answer(response)

            if not predicted:
                failed_extractions += 1
            elif predicted == example["answer"]:
                correct += 1

        accuracy = correct / total
        print(f"Results: {correct}/{total} correct ({accuracy:.4f})")
        if failed_extractions > 0:
            print(f"✗ Failed to extract answer: {failed_extractions}/{total}")
        else:
            print(f"✓ Successfully extracted all answers")

        return accuracy

    def run_complete_finetuning(self, train_data: list):
        """
        Run complete fine-tuning pipeline with all optimizations including gradient clipping.

        Args:
            train_data (list): List of training examples.
        """
        # Analyse data
        self.analyze_data(train_data, "Train")

        # Set up model and trainer
        self.setup_model()

        # Show example format (after tokenizer is loaded)
        print(f"\nExample prompt format:")
        example = self.format_prompt(
            train_data[0]["question"][:80] + "...",
            train_data[0]["options"][:2],
            train_data[0]["answer"],
        )
        print(example[:150] + "...")

        print(f"\nOptimizations enabled:")
        print(f"  - Dataset processing: {self.config.dataset_num_proc} CPU cores")
        print(
            f"  - Memory-efficient caching: batch size {self.config.cache_writer_batch_size}"
        )
        print(
            f"  - Optimized DataLoader: {self.config.dataloader_num_workers} workers, pin_memory, persistent_workers"
        )
        print(
            f"  - GPU memory management: cache clearing every {self.config.torch_empty_cache_steps} steps"
        )
        print(
            f"  - Gradient clipping: max_grad_norm={self.config.max_grad_norm} for training stability"
        )

        self.setup_trainer(train_data)

        # Train and save
        self.train()
        self.save_model()
