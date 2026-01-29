import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

INPUT_CSV = "/Users/joannehui/Desktop/fyp/padchest/reports_500_medgemma.csv"
TEXT_COLUMN = "medgemma_lay_en"
OUTPUT_CSV = "/Users/joannehui/Desktop/fyp/padchest/reports_500_medgemma_nllb_bi.csv"
COL_SIMPLIFIED = "nllb_zh_Hans_medgemma"  
COL_TRADITIONAL = "nllb_zh_Hant_medgemma"  

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not in CSV. Columns: {list(df.columns)}")

    device = 0 if torch.cuda.is_available() else -1

 # One NLLB model, two different target langs
    nllb_model = "facebook/nllb-200-distilled-600M"  # same as before [web:6][web:12][web:35]

    pipe_zh_Hans = pipeline(
        task="translation",
        model=nllb_model,
        src_lang="eng_Latn",
        tgt_lang="zho_Hans",
        device=device,
    )

    pipe_zh_Hant = pipeline(
        task="translation",
        model=nllb_model,
        src_lang="eng_Latn",
        tgt_lang="zho_Hant",
        device=device,
    )  # [web:22][web:23][web:27]

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    out_simp = []
    out_trad = []

    for text in tqdm(texts, desc="Translating (NLLB: Hans + Hant)"):
        if not text.strip():
            out_simp.append("")
            out_trad.append("")
            continue

        r1 = pipe_zh_Hans(text, max_length=256)
        r2 = pipe_zh_Hant(text, max_length=256)

        out_simp.append(r1[0]["translation_text"])
        out_trad.append(r2[0]["translation_text"])

    df[COL_SIMPLIFIED] = out_simp
    df[COL_TRADITIONAL] = out_trad
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
