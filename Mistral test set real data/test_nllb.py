import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1) Model setup
MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading NLLB model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# NLLB language codes
SRC_LANG = "eng_Latn"
TGT_ZH_SIMPLIFIED = "zho_Hans"
TGT_ZH_TRADITIONAL = "zho_Hant"

def translate_with_nllb(text: str, target_lang: str) -> str:
    if not text or not text.strip():
        return ""

    tokenizer.src_lang = SRC_LANG
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256,
    )

    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated

# 2) Load your CSV
in_path = "/Users/joannehui/Desktop/fyp/test_real/CLEAR_test_use_mistral_finetuned.csv"
df = pd.read_csv(in_path)

if "mistral_finetuned_lay_en" not in df.columns:
    raise ValueError("Column 'mistral_finetuned_lay_en' not found in CSV.")

if "prompt_id" not in df.columns:
    raise ValueError("Column 'prompt_id' not found in CSV.")

lay_en = df["mistral_finetuned_lay_en"].fillna("")
prompt_ids = df["prompt_id"].fillna(0).astype(int)

zh_cn_list = []
zh_tw_list = []

# 3) Loop and translate
for text, pid in tqdm(zip(lay_en, prompt_ids), total=len(df), desc="Translating with NLLB"):
    if pid in [3, 4]:
        zh_cn = translate_with_nllb(text, TGT_ZH_SIMPLIFIED)
        zh_tw = ""
    elif pid in [5, 6]:
        zh_cn = ""
        zh_tw = translate_with_nllb(text, TGT_ZH_TRADITIONAL)
    else:
        zh_cn = ""
        zh_tw = ""

    zh_cn_list.append(zh_cn)
    zh_tw_list.append(zh_tw)

# 4) Save new CSV
df["nllb_lay_zh_cn"] = zh_cn_list
df["nllb_lay_zh_tw"] = zh_tw_list

out_path = "/Users/joannehui/Desktop/fyp/test_real/CLEAR_test_use_mistral_finetuned_nllb.csv"
df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")
