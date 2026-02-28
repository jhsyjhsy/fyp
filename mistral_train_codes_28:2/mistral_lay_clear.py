import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# --------- CONFIG ----------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
CACHE_DIR = "/data/users/huisinyu/hf_cache"

INPUT_CSV = "CLEAR_train_only_mistral_lay.csv"          # update if different
OUTPUT_CSV = "CLEAR_with_prompts_mistral_lay_V1.csv"
TEXT_COL = "Report_Content"
PROMPT_ID_COL = "prompt_id"
# ---------------------------

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

df = pd.read_csv(INPUT_CSV)

SAFETY_RULES = """SAFETY RULES (must follow):
- Do not state any causes, risks, symptoms, or treatments that are not explicitly written in the report.
- Do not say a finding is "common", "not serious", "benign", or "not cancer" unless the report uses those words.
- Do not suggest monitoring, follow-up tests, or treatments.
- Only describe what the X-ray report says. Stay neutral about seriousness.
- Use short sentences (2–3 sentences total)."""

PROMPT_1 = f"""{SAFETY_RULES}

TASK: Rewrite this chest X-ray report in simple English for a secondary school student (Form 3–5 level) to understand on their own.

Use:
- Short sentences (about 10–15 words).
- Everyday words. Explain medical terms in brackets once.
- Clear structure: describe what the X-ray shows in simple terms."""
PROMPT_2 = f"""{SAFETY_RULES}

TASK: Rewrite this chest X-ray report in very simple English for an elderly person (Primary 6 level or easier) to read alone.

Use:
- Very short sentences (5–10 words).
- Very common words only.
- One idea per sentence."""
PROMPT_3 = f"""{SAFETY_RULES}

TASK: Rewrite this chest X-ray report in the simplest English possible for a caregiver to read to an elderly person.

Use:
- Very short sentences (under 10 words).
- Easiest words you can.
- Number sentences if there are several findings."""
PROMPT_4 = f"""{SAFETY_RULES}

TASK: Rewrite this chest X-ray report in simple English (Form 3–5 level) for a caregiver helping an elderly person.

Use:
- Short sentences (10–15 words).
- Everyday words with one short explanation in brackets if needed.
- You may mention if things are similar or changed compared with previous X-rays, but do not guess the reason for change."""

def select_prompt(prompt_id: int) -> str:
    if prompt_id == 1:
        return PROMPT_1
    elif prompt_id == 2:
        return PROMPT_2
    elif prompt_id == 3:
        return PROMPT_3
    else:
        return PROMPT_4

def make_prompt(report_text: str, prompt_id: int) -> str:
    base = select_prompt(prompt_id)
    return f"""<s>[INST] {base}

Original chest X-ray report:
\"\"\"{report_text}\"\"\"

/INST]"""

lay_outputs = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating lay summaries"):
    text = str(row.get(TEXT_COL, "") or "").strip()
    pid = int(row.get(PROMPT_ID_COL, 1))

    if not text:
        lay_outputs.append("")
        continue

    prompt = make_prompt(text, pid)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,   # a bit longer than 96 to allow 2–3 short sentences
            temperature=0.7,
            do_sample=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lay_outputs.append(decoded)


# Length safety
lay_outputs = lay_outputs[:len(df)] + [""] * (len(df) - len(lay_outputs))

df["mistral_lay_en"] = lay_outputs
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to {OUTPUT_CSV}")

