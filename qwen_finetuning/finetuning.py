import json
import torch
import re
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from dotenv import load_dotenv
from .config import QwenFineTuningConfig


class QwenFineTuning:
    """Main class for Qwen fine-tuning with LoRA."""
    
    def __init__(self, config: QwenFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up environment variables."""
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env file")
        
        print(f"✓ Environment loaded, HF token available")
    
    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """Load data from JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def format_prompt(self, question: str, options: list, answer: str = None) -> str:
        """Format using Qwen3 chat template."""
        options_text = "\n".join([
            f"{list(opt.keys())[0]}) {list(opt.values())[0]}" 
            for opt in options
        ])
        
        if answer is not None:
            # Training format
            messages = [
                {"role": "user", "content": f"Domanda: {question}\n\n{options_text}"},
                {"role": "assistant", "content": answer}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
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
                enable_thinking=False
            )
    
    def prepare_dataset(self, data: list) -> Dataset:
        """Prepare dataset for training."""
        formatted_data = []
        for example in data:
            text = self.format_prompt(example['question'], example['options'], example['answer'])
            formatted_data.append({"text": text})
        return Dataset.from_list(formatted_data)
    
    @staticmethod
    def analyze_data(data: list, name: str):
        """Analyse dataset distribution."""
        categories = {}
        answers = {}
        
        for item in data:
            cat = item.get('category', 'unknown')
            ans = item.get('answer', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            answers[ans] = answers.get(ans, 0) + 1
        
        print(f"{name} Dataset: {len(data)} examples")
        print(f"Categories: {', '.join(f'{k}({v})' for k, v in categories.items())}")
        print(f"Answer distribution: {', '.join(f'{k}({v})' for k, v in sorted(answers.items()))}")
    
    def setup_model(self):
        """Set up model and tokeniser."""
        print("Loading model and tokeniser...")
        
        # Load model with trust_remote_code for Qwen3
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=self.hf_token
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True,
            token=self.hf_token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        print("Trainable parameters:")
        self.model.print_trainable_parameters()
    
    def setup_trainer(self, train_data: list):
        """Set up trainer for fine-tuning."""
        print("Setting up trainer...")
        
        train_dataset = self.prepare_dataset(train_data)
        
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            save_strategy="epoch",
            seed=42,
            bf16=True,
            max_length=self.config.max_length,
            packing=True,
            dataset_text_field="text"
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
        )
    
    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
        
        print("Starting training...")
        self.trainer.train()
    
    def save_model(self):
        """Save trained model."""
        if self.trainer is None:
            raise ValueError("Trainer not available. Complete training first.")
        
        print("Saving model...")
        self.trainer.save_model()
        print("✓ Training completed")
    
    @staticmethod
    def extract_answer(output: str) -> str:
        """Extract answer from model output."""
        if not output:
            return ""
        match = re.search(r'\b([ABCDE])\b', output.upper())
        return match.group(1) if match else ""
    
    def _format_options_text(self, options: list) -> str:
        """Helper method for evaluation."""
        return "\n".join([
            f"{list(opt.keys())[0]}) {list(opt.values())[0]}" 
            for opt in options
        ])
    
    def evaluate_model(self, test_data: list) -> float:
        """Evaluate model on test data."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        
        correct = 0
        total = len(test_data)
        failed_extractions = 0
        
        for example in tqdm(test_data, desc="Evaluating"):
            # Use chat template for evaluation
            messages = [
                {"role": "user", "content": f"Domanda: {example['question']}\n\n{self._format_options_text(example['options'])}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs.to(self.model.device),
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = self.extract_answer(response)
            
            if not predicted:
                failed_extractions += 1
            elif predicted == example['answer']:
                correct += 1
        
        accuracy = correct / total
        print(f"Results: {correct}/{total} correct ({accuracy:.4f})")
        if failed_extractions > 0:
            print(f"✗ Failed to extract answer: {failed_extractions}/{total}")
        else:
            print(f"✓ Successfully extracted all answers")
        
        return accuracy
    
    def run_complete_finetuning(self, train_data: list):
        """Run complete fine-tuning pipeline."""
        # Analyse data
        self.analyze_data(train_data, "Train")
        
        # Set up model and trainer
        self.setup_model()
        
        # Show example format (after tokenizer is loaded)
        print(f"\nExample prompt format:")
        example = self.format_prompt(
            train_data[0]['question'][:80] + "...", 
            train_data[0]['options'][:2], 
            train_data[0]['answer']
        )
        print(example[:150] + "...")
        
        self.setup_trainer(train_data)
        
        # Train and save
        self.train()
        self.save_model()