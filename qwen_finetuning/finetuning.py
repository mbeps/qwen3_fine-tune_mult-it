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
from .config import QwenFineTuningConfig, ThinkingMode


class QwenFineTuning:
    """
    QwenFineTuning provides an interface for fine-tuning Qwen models with support for mixed thinking/non-thinking training modes.

    This class handles environment setup, data formatting, model and tokenizer loading, LoRA configuration, training, evaluation, and saving.
    It is designed to be minimal, robust, and compatible with Qwen3 recommendations for both standard and chain-of-thought (thinking) training.
    """

    def __init__(self, config: QwenFineTuningConfig):
        """
        Initialise the QwenFineTuning instance with a configuration object.
        Loads environment variables and prepares placeholders for model, tokenizer, and trainer.
        Args:
            config (QwenFineTuningConfig): Configuration for fine-tuning.
        """
        self.config: QwenFineTuningConfig = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._setup_environment()

    def _setup_environment(self) -> None:
        """
        Load environment variables required for Hugging Face authentication.
        Raises an error if the HF_TOKEN is not found in the .env file.
        """
        load_dotenv()
        self.hf_token: str | None = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env file")

    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """
        Load a JSONL (JSON Lines) file into a list of dictionaries.
        Args:
            file_path (str): Path to the JSONL file.
        Returns:
            list: List of parsed JSON objects.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def format_prompt(self, example: dict, force_thinking: bool = None) -> dict:
        """
        Format a single example into a chat prompt for the model, supporting both standard QA and chain-of-thought (thinking) formats.
        Args:
            example (dict): Example with 'question', 'options', 'answer', and optionally 'thinking'.
            force_thinking (bool, optional): If set, overrides the default thinking mode for this example.
        Returns:
            dict: Dictionary with a single 'text' field containing the formatted prompt.
        """
        question = example["question"]
        options = example["options"]
        answer = example["answer"]
        thinking = example.get("thinking", None)

        use_thinking: bool = (
            force_thinking
            if force_thinking is not None
            else (
                self.config.thinking_mode == ThinkingMode.ENABLED
                or (
                    self.config.thinking_mode == ThinkingMode.MIXED
                    and thinking is not None
                )
            )
        )

        options_text: str = "\n".join(
            [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
        )
        user_content: str = f"Domanda: {question}\n\n{options_text}"

        if use_thinking and thinking:
            assistant_content = f"<think>\n{thinking}\n</think>\n\n{answer}"
        else:
            assistant_content = answer

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=use_thinking,
        )

        return {"text": text}

    def prepare_dataset(self, data: list) -> Dataset:
        """
        Prepare and format a dataset for training, counting thinking/non-thinking examples and printing statistics.
        Args:
            data (list): List of raw data examples.
        Returns:
            Dataset: Hugging Face Dataset object with formatted prompts.
        """
        formatted_data = []
        thinking_count = 0
        mixed_thinking_count = 0

        for example in tqdm(data, desc="Formatting"):
            formatted = self.format_prompt(example)
            formatted_data.append(formatted)
            if example.get("thinking"):
                thinking_count += 1
                if self.config.thinking_mode == ThinkingMode.MIXED:
                    if "<think>" in formatted["text"]:
                        mixed_thinking_count += 1

        print(f"Dataset prepared: {len(formatted_data)} examples")
        if self.config.thinking_mode == ThinkingMode.ENABLED:
            print("All examples using thinking mode")
        elif self.config.thinking_mode == ThinkingMode.MIXED:
            print(
                f"Mixed training: {mixed_thinking_count} thinking, {len(formatted_data) - mixed_thinking_count} non-thinking"
            )
            print(f"Original data had {thinking_count} examples with thinking content")
        else:
            print("All examples using non-thinking mode")

        dataset = Dataset.from_list(formatted_data)
        return dataset

    @staticmethod
    def analyze_data(data: list, name: str):
        """
        Analyse and print statistics about a dataset, including category and answer distribution and thinking example count.
        Args:
            data (list): List of data examples.
            name (str): Name to display for the dataset (e.g., 'Training', 'Test').
        """
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
            print(
                f"Examples with thinking: {thinking_examples} ({thinking_examples / len(data) * 100:.1f}%)"
            )

    def setup_model(self) -> None:
        """
        Load the model and tokenizer from Hugging Face Hub, configure LoRA for parameter-efficient fine-tuning, and set padding tokens as needed.
        """
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
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def setup_trainer(self, train_data: list):
        """
        Set up the SFTTrainer for supervised fine-tuning with the provided training data and configuration.
        Prepares the dataset, configures training arguments, and initialises the trainer.
        Args:
            train_data (list): List of training examples.
        """
        train_dataset = self.prepare_dataset(train_data)
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
            bf16=True,
            max_length=self.config.max_length,
            packing=True,
            dataset_text_field="text",
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
        )
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
        )
        print(
            f"✓ Trainer ready ({len(train_dataset)} samples, {len(train_dataset) // self.config.effective_batch_size * self.config.num_epochs} steps)"
        )
        print(f"✓ Thinking mode: {self.config.thinking_mode.value}")
        print(f"✓ Effective batch size: {self.config.effective_batch_size}")

    def train(self) -> None:
        """
        Start the training process using the configured trainer.
        Raises an error if the trainer is not set up.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
        print(f"\nStarting training with {self.config.thinking_mode.value} mode...")
        self.trainer.train()
        print("✓ Training completed")

    def save_model(self) -> None:
        """
        Save the trained model and tokenizer to the output directory specified in the configuration.
        Raises an error if training has not been performed.
        """
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"✓ Model saved to {self.config.output_dir}")

    @staticmethod
    def extract_answer(output: str) -> str:
        """
        Extract a single answer letter (A-E) from the model's output string.
        Args:
            output (str): Model output text.
        Returns:
            str: Extracted answer letter, or empty string if not found.
        """
        if not output:
            return ""
        match = re.search(r"\b([ABCDE])\b", output.upper())
        return match.group(1) if match else ""

    def evaluate_model(self, test_data: list) -> float:
        """
        Evaluate the fine-tuned model on a test dataset, reporting accuracy and extraction failures.
        Args:
            test_data (list): List of test examples.
        Returns:
            float: Accuracy score (correct/total).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model() first.")
        correct = 0
        total = len(test_data)
        failed_extractions = 0
        self.model.eval()
        if self.config.thinking_mode in [ThinkingMode.ENABLED, ThinkingMode.MIXED]:
            gen_params = {
                "max_new_tokens": 2048,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            }
            enable_thinking = True
        else:
            gen_params = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
            enable_thinking = False
        for example in tqdm(test_data, desc="Evaluating"):
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
            if self.config.thinking_mode == ThinkingMode.MIXED:
                enable_thinking = bool(example.get("thinking"))
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_params,
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
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        if failed_extractions > 0:
            print(f"Failed extractions: {failed_extractions}")
        return accuracy

    def run_complete_pipeline(
        self, train_data: list, test_data: list = None
    ) -> float | None:
        """
        Run the complete fine-tuning pipeline: analyze data, set up model and trainer, train, save, and optionally evaluate.
        Args:
            train_data (list): Training examples.
            test_data (list, optional): Test examples for evaluation.
        Returns:
            float or None: Accuracy if test_data is provided, else None.
        """
        self.analyze_data(train_data, "Training")
        if test_data:
            self.analyze_data(test_data, "Test")
        self.setup_model()
        self.setup_trainer(train_data)
        self.train()
        self.save_model()
        if test_data:
            accuracy = self.evaluate_model(test_data)
            return accuracy
        return None
