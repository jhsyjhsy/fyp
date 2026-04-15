import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from peft import PeftModel

# --------- CONFIG ----------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_REPO = "jhsyjhsy/mistral_finetune_v1"
CACHE_DIR = "/data/users/huisinyu/hf_cache"

INPUT_CSV = "CLEAR_test_use.csv"
OUTPUT_CSV = "CLEAR_test_use_mistral_finetuned.csv"
TEXT_COL = "Report_Content"
PROMPT_ID_COL = "prompt_id"
# ---------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

print("Loading finetuned LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows, prompt dist: {df[PROMPT_ID_COL].value_counts().to_dict()}")

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

def select_prompt(prompt_id: int) -> str:
    if prompt_id in [1, 3, 5]:
        return PROMPT_1
    elif prompt_id in [2, 4, 6]:
        return PROMPT_2
    else:
        return PROMPT_1

def make_prompt(report_text: str, prompt_id: int) -> str:
    base = select_prompt(prompt_id)
    return f"""[INST] {base}

Original chest X-ray report:
\"\"\"
{report_text}
\"\"\"

[/INST]"""

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
            max_new_tokens=160,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    lay_outputs.append(decoded)

lay_outputs = lay_outputs[:len(df)] + [""] * (len(df) - len(lay_outputs))

df["mistral_finetuned_lay_en"] = lay_outputs
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to {OUTPUT_CSV}")
