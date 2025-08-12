import json
import torch
import re
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from dotenv import load_dotenv
from .config import QwenFineTuningConfig


class QwenFineTuning:
    """
    Simplified Qwen fine-tuning class.
    Removes all unnecessary complexity and focuses on what works.
    """

    def __init__(self, config: QwenFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._setup_environment()

    def _setup_environment(self):
        """Load environment variables."""
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env file")

    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """Load data from JSONL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def format_prompt(self, example: dict) -> dict:
        """
        Format a single example using chat template.
        Supports both standard QA and Chain-of-Thought formats.

        Args:
            example: Dict with 'question', 'options', 'answer' and optionally 'thinking'
        Returns:
            Dict with formatted 'text' field
        """
        question = example["question"]
        options = example["options"]
        answer = example["answer"]
        thinking = example.get("thinking", None)

        # Format options
        options_text = "\n".join(
            [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
        )

        # Create user message
        user_content = f"Domanda: {question}\n\n{options_text}"

        # Create assistant response based on thinking mode and data availability
        if self.config.enable_thinking and thinking:
            # Thinking mode: include reasoning in <think> tags
            assistant_content = f"<think>\n{thinking}\n</think>\n\n{answer}"
        else:
            # Standard mode: direct answer only
            assistant_content = answer

        # Create messages for chat template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        # Apply chat template with appropriate thinking mode
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.config.enable_thinking,
        )

        return {"text": text}

    def prepare_dataset(self, data: list) -> Dataset:
        """
        Prepare dataset for training.
        Simple, single-process approach that works.
        """
        formatted_data = []
        thinking_count = 0

        for example in tqdm(data, desc="Formatting"):
            formatted = self.format_prompt(example)
            formatted_data.append(formatted)

            # Count examples with thinking content
            if example.get("thinking"):
                thinking_count += 1

        # Print dataset statistics
        print(f"Dataset prepared: {len(formatted_data)} examples")
        if self.config.enable_thinking:
            print(
                f"Examples with thinking content: {thinking_count}/{len(formatted_data)}"
            )

        dataset = Dataset.from_list(formatted_data)
        return dataset

    @staticmethod
    def analyze_data(data: list, name: str):
        """Analyze dataset distribution."""
        categories = {}
        answers = {}
        thinking_examples = 0

        for item in data:
            cat = item.get("category", "unknown")
            ans = item.get("answer", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            answers[ans] = answers.get(ans, 0) + 1

            if item.get("thinking"):
                thinking_examples += 1

        print(f"{name}: {len(data)} examples, {len(categories)} categories")
        print(
            f"Answer distribution: {', '.join(f'{k}:{v}' for k, v in sorted(answers.items()))}"
        )
        if thinking_examples > 0:
            print(f"Examples with thinking: {thinking_examples}")

    def setup_model(self):
        """
        Load model and tokenizer, configure LoRA.
        No quantization as per requirements.
        """
        # Load model without quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Use bfloat16 for RTX 6000
            trust_remote_code=True,
            token=self.hf_token,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, token=self.hf_token
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

        # Configure LoRA - simple and proven configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def setup_trainer(self, train_data: list):
        """
        Set up trainer with proven configuration.
        Simple, effective settings without overengineering.
        """
        # Prepare dataset
        train_dataset = self.prepare_dataset(train_data)

        # Training arguments - optimised for RTX 6000
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if self.config.gradient_checkpointing
            else None,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            seed=42,
            bf16=True,  # Use bf16 for RTX 6000
            max_length=self.config.max_length,
            packing=True,  # Keep packing for efficiency
            dataset_text_field="text",
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
        )

        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
        )

        print(
            f"✓ Trainer ready ({len(train_dataset)} samples, {len(train_dataset) // self.config.effective_batch_size * self.config.num_epochs} steps)"
        )
        print(
            f"✓ Thinking mode: {'Enabled' if self.config.enable_thinking else 'Disabled'}"
        )

    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")

        print("\nStarting training...")
        self.trainer.train()
        print("✓ Training completed")

    def save_model(self):
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"✓ Model saved to {self.config.output_dir}")

    @staticmethod
    def extract_answer(output: str) -> str:
        """Extract answer letter from model output."""
        if not output:
            return ""
        match = re.search(r"\b([ABCDE])\b", output.upper())
        return match.group(1) if match else ""

    def evaluate_model(self, test_data: list) -> float:
        """
        Evaluate model on test data.

        Args:
            test_data: List of test examples
        Returns:
            Accuracy score
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model() first.")

        correct = 0
        total = len(test_data)
        failed_extractions = 0

        # Put model in eval mode
        self.model.eval()

        # Set generation parameters based on thinking mode (Qwen3 recommendations)
        if self.config.enable_thinking:
            gen_params = {
                "max_new_tokens": 2048,  # More tokens for thinking
                "do_sample": True,
                "temperature": 0.6,  # Qwen3 recommended for thinking
                "top_p": 0.95,  # Qwen3 recommended for thinking
                "top_k": 20,
            }
        else:
            gen_params = {
                "max_new_tokens": 256,  # Fewer tokens for direct answers
                "do_sample": True,
                "temperature": 0.7,  # Qwen3 recommended for non-thinking
                "top_p": 0.8,  # Qwen3 recommended for non-thinking
                "top_k": 20,
            }

        for example in tqdm(test_data, desc="Evaluating"):
            # Format prompt for inference
            options_text = "\n".join(
                [
                    f"{list(opt.keys())[0]}) {list(opt.values())[0]}"
                    for opt in example["options"]
                ]
            )

            messages = [
                {
                    "role": "user",
                    "content": f"Domanda: {example['question']}\n\n{options_text}",
                }
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs.to(self.model.device),
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_params,
                )

            # Decode and extract answer
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            predicted = self.extract_answer(response)

            if not predicted:
                failed_extractions += 1
            elif predicted == example["answer"]:
                correct += 1

        accuracy = correct / total

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        if failed_extractions > 0:
            print(f"Failed extractions: {failed_extractions}")

        return accuracy

    def run_complete_pipeline(self, train_data: list, test_data: list = None):
        """
        Run the complete training pipeline.

        Args:
            train_data: Training examples
            test_data: Optional test examples for evaluation
        """
        # Analyze data
        self.analyze_data(train_data, "Training")
        if test_data:
            self.analyze_data(test_data, "Test")

        # Setup and train
        self.setup_model()
        self.setup_trainer(train_data)
        self.train()
        self.save_model()

        # Evaluate if test data provided
        if test_data:
            accuracy = self.evaluate_model(test_data)
            return accuracy

        return None
