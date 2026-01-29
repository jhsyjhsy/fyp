import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import MarianMTModel, MarianTokenizer  # [web:126][web:158]

INPUT_CSV = "/Users/joannehui/Desktop/fyp/padchest/reports_500_medgemma.csv"
TEXT_COLUMN = "medgemma_lay_en"
OUTPUT_CSV = "/Users/joannehui/Desktop/fyp/padchest/medgemma/reports_500_medgemma_opusmt_bi.csv"

# Two Simplified-Chinese outputs for research alignment with Mistral+Opus
COL_ZH_CN_1 = "opusmt_zh_cn_medgemma_v1"
COL_ZH_CN_2 = "opusmt_zh_cn_medgemma_v2"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"  # en → zh (Simplified) [web:123][web:163]

def load_model(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    return tok, mdl, device

def translate_one(text, tokenizer, model, device, max_length=256):
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    generated = model.generate(**batch, max_length=max_length)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not in CSV. Columns: {list(df.columns)}")

    print("Loading Opus-MT model...")
    tok, model, device = load_model(MODEL_NAME)

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    out_v1, out_v2 = [], []

    for text in tqdm(texts, desc="Opus-MT (two Simplified outputs)"):
        if not text.strip():
            out_v1.append("")
            out_v2.append("")
            continue

        zh1 = translate_one(text, tok, model, device)
        zh2 = translate_one(text, tok, model, device)  # second pass, same model
        out_v1.append(zh1)
        out_v2.append(zh2)

    df[COL_ZH_CN_1] = out_v1
    df[COL_ZH_CN_2] = out_v2
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

