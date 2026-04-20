#!/usr/bin/env python3
"""
CLEAR Mistral LoRA Finetune - HKU HPC Optimized
Joanne's FYP: Chest X-ray lay summaries (Prompts 1-4)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer
)
from trl import SFTTrainer
from dataclasses import dataclass
from typing import Dict, Any
import torch

# === CONFIG (UPDATE DATA_PATH!) ===
DATA_PATH = "/home/huisinyu/fyp/train_real/CLEAR_mistral_dev_cleaned.csv" 
OUTPUT_DIR = "./clear_mistral_lora_final"
MAX_LENGTH = 512
BATCH_SIZE = 1  # Conservative for HPC
EPOCHS = 1.5
LR = 1e-5

print(f"🚀 Starting CLEAR Mistral LoRA on {DATA_PATH}")

# Load + prep data
df = pd.read_csv(os.path.expanduser(DATA_PATH))
print(f"Loaded {len(df)} rows. Split: {df['split'].value_counts().to_dict() if 'split' in df else 'No split'}")

# Use train split OR 80% sample
if 'split' in df.columns:
    train_df = df[df['split'] == 'train']
else:
    train_df = df.sample(frac=0.8, random_state=42)
print(f"Training: {len(train_df)} samples")

def format_instruction(row):
    """Prompt 1-4 instruction format"""
    return {
        "text": f"""<s>[INST] Rewrite chest X-ray report using Prompt {row['prompt_id']}:

{row.get('report', row.get('Exam_Taken', 'N/A'))}

Follow safety rules: describe all findings only. [/INST]

{row['mistral_lay_en_clean']} </s>"""
    }

dataset = Dataset.from_pandas(train_df).map(format_instruction)
dataset = dataset.remove_columns(train_df.columns.tolist())

# Tokenizer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Model + LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    warmup_steps=20,
    learning_rate=LR,
    weight_decay=0.01,
    fp16=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
   # dataset_text_field="text",
    args=args,
   # tokenizer=tokenizer,
   # packing=False,
   # max_seq_length=512
)

print("🎯 Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ SAVED: {OUTPUT_DIR}")

# Quick test
print("\n🧪 Test inference:")
model.eval()
test_input = "<s>[INST] Chest X-ray Prompt 2 (elderly): Heart normal. Lungs clear. [/INST]"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))

