import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --------- CONFIG ---------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "reports_500.csv"
TEXT_COLUMN = "sentence_en"

OUTPUT_CSV = INPUT_CSV.replace(".csv", "_mistral_zh.csv")
COL_ZH_CN = "mistral_zh_cn"
COL_ZH_HK = "mistral_zh_hk"

MAX_NEW_TOKENS = 128
BATCH_SIZE = 4          # try 4–8 depending on VRAM
# --------------------------


INSTR_ZH_CN = (
    "You are a helpful medical translator. "
    "Translate the following chest X-ray sentence into simple Chinese "
    "(Simplified, Mainland style) that a person with secondary education and basic medical "
    "knowledge can understand, in 1–3 short sentences. "
    "Keep all important clinical information and do not add new facts or treatment suggestions."
)

INSTR_ZH_HK = (
    "You are a helpful medical translator. "
    "Translate the following chest X-ray sentence into simple Chinese "
    "(Traditional, Hong Kong style) that a patient with secondary education and basic medical "
    "knowledge can understand, in 1–3 short sentences. "
    "Use natural Hong Kong written Chinese, keep all important clinical information, "
    "and do not add new facts or treatment suggestions."
)


def build_prompt_zh_cn(sentence_en: str) -> str:
    return (
        f"<s>[INST] {INSTR_ZH_CN} "
        f'Sentence:\n""\"{sentence_en}""\" [/INST]'
    )


def build_prompt_zh_hk(sentence_en: str) -> str:
    return (
        f"<s>[INST] {INSTR_ZH_HK} "
        f'Sentence:\n""\"{sentence_en}""\" [/INST]'
    )


def load_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # ensure a *GPU* is seen
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts, max_new_tokens: int):
    """
    prompts: list[str]
    returns: list[str]
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Strip echoed prompt if needed
    cleaned = []
    for t in texts:
        if "[/INST]" in t:
            t = t.split("[/INST]", 1)[-1].strip()
        cleaned.append(t.strip())
    return cleaned


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not in CSV. Columns: {list(df.columns)}")

    model, tokenizer = load_mistral()

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    zh_cn_all, zh_hk_all = [], []

    n = len(texts)
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Mistral EN → ZH (CN + HK, batched)"):
        batch = texts[i : i + BATCH_SIZE]

        # Handle empty rows
        non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
        if not non_empty_indices:
            zh_cn_all.extend([""] * len(batch))
            zh_hk_all.extend([""] * len(batch))
            continue

        batch_cn_prompts = []
        batch_hk_prompts = []
        for t in batch:
            if t.strip():
                batch_cn_prompts.append(build_prompt_zh_cn(t))
                batch_hk_prompts.append(build_prompt_zh_hk(t))
            else:
                batch_cn_prompts.append("")  # placeholder
                batch_hk_prompts.append("")

        # Filter out placeholders for generation
        real_cn_prompts = [p for p in batch_cn_prompts if p]
        real_hk_prompts = [p for p in batch_hk_prompts if p]

        cn_out = generate_batch(model, tokenizer, real_cn_prompts, MAX_NEW_TOKENS)
        hk_out = generate_batch(model, tokenizer, real_hk_prompts, MAX_NEW_TOKENS)

        # Reinsert blanks in the right positions
        cn_iter = iter(cn_out)
        hk_iter = iter(hk_out)
        for t in batch:
            if t.strip():
                zh_cn_all.append(next(cn_iter))
                zh_hk_all.append(next(hk_iter))
            else:
                zh_cn_all.append("")
                zh_hk_all.append("")

    df[COL_ZH_CN] = zh_cn_all
    df[COL_ZH_HK] = zh_hk_all
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

