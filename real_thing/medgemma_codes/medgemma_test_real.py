#!/usr/bin/env python3
# MedGemma Pseudo-Label (Finetuned) - Uses finetuned LoRA model, adds safety template from finetune script

import os
import pandas as pd
import torch
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# From finetune_medgemma-2.py - exact SAFETY_PROMPT_TEMPLATE
SAFETY_PROMPT_TEMPLATE = """<s>[INST] Rewrite the radiology report as simple language for patients. Neutral tone only. Safety Rules must follow:
- ONLY the simplified report. No 'I am AI', no 'Thank you', no examples, no notes.
- No emoji, no code, no symbols/garbage.
- 3-4 short sentences max. Medical facts only.
- Chinese: Pure, normal punctuation only. English: Simple words.
- Do not state any causes, risks, symptoms, or treatments not explicitly in the report.
- Do not say a finding is common, not serious, benign, or not cancer unless the report uses those words.
- Do not suggest monitoring, follow-up tests, or treatments.
- Only describe what the X-ray shows. Stay neutral about seriousness.
Report: {report}
Simplified: [/INST]"""

MODELNAME = "google/medgemma-4b-it"
CACHEDIR = "data/users/huisinyuhf/cache"  # Your path from original
FINETUNED_PATH = "/data/users/huisinyu/pseudo/medgemma-simplified-lora"  # Change if your finetuned path differs

SAFETY_RULES = """SAFETY RULES must follow:
- Do not state any causes, risks, symptoms, or treatments not explicitly in the report.
- Do not say a finding is common, not serious, benign, or not cancer unless the report uses those words.
- Do not suggest monitoring, follow-up tests, or treatments.
- Only describe what the X-ray shows. Stay neutral about seriousness.
- Use short sentences. Limit to 2-3 sentences."""

PROMPT1 = f"""{SAFETY_RULES}
Rewrite this chest X-ray report in simple English for a secondary school student (Form 3-5 level) to understand on their own. Use-
- Short sentences (10-15 words).
- Everyday words. Explain medical terms in brackets once.
- Active voice.
- Clear structure: findings -> what it means simply.
- If compares with prior X-ray, briefly mention comparison.
There is a shadow in the lower right lung (right lung base). Do not add extra."""

PROMPT2 = f"""{SAFETY_RULES}
Rewrite this chest X-ray report in very simple English for primary school reading level. Use-
- Very short sentences (5-10 words).
- Very common words only. Say 'heart' not 'cardiac', 'lung bottom' not 'costophrenic angle'.
- Repeat key words if needed.
- One main idea per sentence.
Heart is normal size. Lungs look clear. Do not add extra."""

PROMPT3 = PROMPT1.replace("simple English", "simplified Chinese, written style")
PROMPT4 = PROMPT2.replace("very simple English", "simplified Chinese, written style")
PROMPT5 = PROMPT1.replace("simple English", "Hong Kong wordings (traditional Chinese), written style")
PROMPT6 = PROMPT2.replace("very simple English", "Hong Kong wordings (traditional Chinese), written style")

PROMPT_MAP = {1: PROMPT1, 2: PROMPT2, 3: PROMPT3, 4: PROMPT4, 5: PROMPT5, 6: PROMPT6}

def build_messages(report_text: str, prompt_id: int):
    base_prompt = PROMPT_MAP.get(int(prompt_id), PROMPT1)
    return [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": f"Original chest X-ray report: {report_text}"}
    ]

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print(f"Loaded {len(df)} rows")
    text_col = 'Report_Content'  # Matches your CSV
    prompt_col = 'prompt_id'
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELNAME, cache_dir=f"{CACHEDIR}/local", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODELNAME, cache_dir=f"{CACHEDIR}/local", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    print("Loading finetuned PEFT model...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)
    model.eval()

    outputs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating pseudo-labels"):
        text = str(row[text_col]).strip()
        pid = int(row[prompt_col])
        if len(text) < 10:
            outputs.append("N/A")
            continue
        messages = build_messages(text, pid)
        # Apply safety template to full prompt for generation (prevents truncation issues)
        safe_prompt = SAFETY_PROMPT_TEMPLATE.format(report=text)
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_input = safe_prompt + chat_text  # Combines safety + prompt
        input_ids = tokenizer(full_input, return_tensors="pt")["input_ids"].to(model.device)
        input_len = input_ids.shape[1]
        print(f"Input tokens: {input_len}")
        input_ids_batched = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        attention_mask = torch.ones_like(input_ids_batched)
        tokenizer.pad_token_id = 128000  # From original
        with torch.no_grad():
            gen = model.generate(
                input_ids_batched, attention_mask=attention_mask,
                max_new_tokens=280, repetition_penalty=1.5, temperature=0.7, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )[0, input_len:]
        raw_decoded = tokenizer.decode(gen, skip_special_tokens=False)
        decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
        decoded = " ".join(decoded.split())  # Normalize spaces
        decoded = decoded.replace("endofturn", "").replace("<indicator>", "").replace("\n", " ")
        decoded = decoded.strip()[:500]
        outputs.append(decoded if decoded else raw_decoded.strip()[:500])

    df['medgemma_pseudolabel'] = outputs
    df.to_csv(output_csv, index=False)
    print(f"Saved {sum(len(x) for x in outputs if x):,} chars to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma pseudo-labeling with finetuned model")
    parser.add_argument("input_csv", nargs="?", default="CLEAR_test_use.csv", help="Input CSV")
    parser.add_argument("--output", "-o", default="CLEAR_test_finetuned.csv", help="Output CSV")
    args = parser.parse_args()
    main(args.input_csv, args.output)

