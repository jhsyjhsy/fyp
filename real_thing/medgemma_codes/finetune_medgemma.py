# Paste this entire block into vim as your new finetune_medgemma.py
# Then :wq

#!/usr/bin/env python3
"""
MedGemma Radiology Simplification Fine-Tuning
- Uses cleaned CLEARtraindevpseudolabels.csv
- LoRA on google/medgemma-4b-it
- Noisy-robust setup for pseudo-labels
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from transformers import Trainer
import argparse
from tqdm import tqdm

# Configs (HKU/HPC friendly)
MODEL_NAME = "google/medgemma-4b-it"
CACHE_DIR = "/data/users/huisinyu/hf_cache"  # Your path from pseudolabel script
DATA_FILE = "CLEARtraindevpseudolabels_cleaned.csv"  # From our cleaning
OUTPUT_DIR = "./medgemma-simplified-lora"

SAFETY_PROMPT_TEMPLATE = """<s>[INST] Rewrite the radiology report as simple language for patients. Neutral tone only.

Safety Rules (must follow):
- ONLY the simplified report. No "I am AI", no "Thank you", no examples, no notes.
- No emoji, no code, no symbols/garbage.
- 3-4 short sentences max. Medical facts only.
- Chinese: Pure 中文, normal punctuation only. English: Simple words.
- Do not state any causes, risks, symptoms, or treatments not explicitly in the report.
- Do not say a finding is "common", "not serious", "benign", or "not cancer" unless the report uses those words.
- Do not suggest monitoring, follow-up tests, or treatments.
- Only describe what the X-ray shows. Stay neutral about seriousness

Report: {report}

Simplified: [/INST]"""

def format_example(row):
    """Format as chat/instruction pair"""
    report = row['Report_Content']
    simplified = row['cleaned']  # Or 'medgemma_pseudolabel' if not cleaned
    prompt = SAFETY_PROMPT_TEMPLATE.format(report=report)
    full_text = f"{prompt}{simplified}</s>"  # Complete response
    return {'text': full_text}

def main(data_file=DATA_FILE, output_dir=OUTPUT_DIR):
    print("Loading data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows. Sample cleaned: {df['cleaned'].iloc[0][:100]}...")

    # Format dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example)

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    dataset = dataset.train_test_split(test_size=0.1)

    # NOW tokenize (tokenizer exists)
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True, max_length=512, padding=False)
        tokenized['token_type_ids'] = tokenized['input_ids']

        return tokenized

    dataset['train'] = dataset['train'].map(tokenize_function, batched=True)
    dataset['test'] = dataset['test'].map(tokenize_function, batched=True)

    # Clean cols
    # Clean ALL raw cols (safer)
    cols_to_remove = ['Case no. Identifier', 'Exam_Taken', 'Sex', 'Exam_Age', 
                  'In/Out/AE_Patient', 'Urgency', 'Report_Content', 'split', 
                  'is_elderly', 'prompt_id', 'medgemma_pseudolabel', 'cleaned', 'text']
    dataset['train'] = dataset['train'].remove_columns([c for c in cols_to_remove if c in dataset['train'].column_names])
    dataset['test'] = dataset['test'].remove_columns([c for c in cols_to_remove if c in dataset['test'].column_names])

    
    # LoRA config (efficient)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args (noisy-robust)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust for GPU (biomed1?)
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        gradient_checkpointing=True,        # 50% mem ↓
        remove_unused_columns=True,
        weight_decay=0.01,
        label_smoothing_factor=0.1,  # Key for noisy labels
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        fp16=False,  # Use bf16=True if A100+
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",  # No wandb unless set
        dataloader_num_workers=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA model to {output_dir}")

    # Quick test inference
   # test_report = df['Report_Content'].iloc[0]
    #inputs = tokenizer(SAFETY_PROMPT_TEMPLATE.format(report=test_report), return_tensors="pt").to(model.device)
   # with torch.no_grad():
       # outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
   # print("Test generation:", tokenizer.decode(outputs[0], skip_special_tokens=True))
print("✅ Training complete! LoRA saved to ./medgemma-simplified-lora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma on cleaned pseudo-labels")
    parser.add_argument("--data", default=DATA_FILE, help="Cleaned CSV path")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output dir")
    args = parser.parse_args()
    main(args.data, args.output)

