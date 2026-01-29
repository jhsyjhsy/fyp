import os
import pandas as pd
from tqdm import tqdm

from medgemma_simplify_one import load_model, simplify_report


INPUT_CSV = "/home/huisinyu/reports_500.csv"
TEXT_COLUMN = "sentence_en"         
OUTPUT_CSV = "/home/huisinyu/reports_500_medgemma.csv"
OUTPUT_COLUMN = "medgemma_lay_en"        # new simplified-English column


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in CSV. Columns: {list(df.columns)}")

    model, processor = load_model()

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    simplified_texts = []

    for text in tqdm(texts, desc="Simplifying with MedGemma"):
        if not text.strip():
            simplified_texts.append("")
            continue

        simplified = simplify_report(model, processor, text, max_new_tokens=96)
        simplified_texts.append(simplified)

    df[OUTPUT_COLUMN] = simplified_texts

    print(f"Saving to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

