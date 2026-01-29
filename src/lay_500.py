import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
cache_dir = "/data/users/huisinyu/hf_cache"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

df = pd.read_csv("reports_500.csv")

def make_prompt(report_text: str) -> str:
    return f"""<s>[INST] You are a helpful medical translator. Translate this chest X-ray sentence into simple language a patient of secondary education background, basic medical background can understand, in 1–3 short sentences.

Sentence:
\"\"\"{report_text}\"\"\"

[/INST]"""

translations = []

for text in tqdm(df["sentence_en"].fillna(""), desc="Translating"):
    if not text.strip():
        translations.append("")
        continue

    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            temperature=0.7,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translations.append(result)

df["lay_translation"] = translations
df.to_csv("reports_500_translated.csv", index=False)

print("Saved to reports_500_translated.csv")

