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
TGT_ZH_SIMPLIFIED = "zho_Hans"  # Simplified Chinese
TGT_ZH_TRADITIONAL = "zho_Hant"  # Traditional Chinese

def translate_with_nllb(text: str, target_lang: str) -> str:
    if not text or not text.strip():
        return ""
    
    tokenizer.src_lang = SRC_LANG
    
    # NEW: Replace lang_code_to_id entirely
    if target_lang == "zho_Hans":
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("zho_Hans")
    elif target_lang == "zho_Hant":
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("zho_Hant")
    else:
        forced_bos_token_id = None

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256,
    )
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated

# 2) Load your CSV
in_path = "/Users/joannehui/Desktop/fyp/padchest/Reports_public/mistral_clean_lay.csv"
df = pd.read_csv(in_path)

if "lay_translation" not in df.columns:
    raise ValueError("Column 'lay_translation' not found in CSV.")

lay_en = df["lay_translation"].fillna("")

zh_cn_list = []
zh_tw_list = []

# 3) Loop and translate
for text in tqdm(lay_en, desc="Translating with NLLB"):
    zh_cn = translate_with_nllb(text, TGT_ZH_SIMPLIFIED)
    zh_tw = translate_with_nllb(text, TGT_ZH_TRADITIONAL)
    zh_cn_list.append(zh_cn)
    zh_tw_list.append(zh_tw)

# 4) Save new CSV
df["lay_zh_cn"] = zh_cn_list
df["lay_zh_tw"] = zh_tw_list

out_path = "/Users/joannehui/Desktop/fyp/padchest/reports_500_mistral_nllb_zh_clean.csv"
df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")

