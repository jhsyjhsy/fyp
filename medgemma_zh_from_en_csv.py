import csv
from pathlib import Path

from medgemma_simplify_one import (
    load_model_and_processor,
    simplify_to_zh_cn,
    simplify_to_zh_hk,
)


INPUT_CSV = Path("reports_500.csv")      # change to your actual file
OUTPUT_CSV = Path("output_medgemma_zh.csv")  # change if you want
MAX_NEW_TOKENS = 96
LIMIT = 500  # or len(csv) if you want full


def main():
    model, processor = load_model_and_processor()

    rows = []
    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= LIMIT:
                break

            # <-- use sentence_en here
            text = row.get("sentence_en", "").strip()

            if not text:
                row["zh_cn"] = ""
                row["zh_hk"] = ""
            else:
                zh_cn = simplify_to_zh_cn(
                    model, processor, text, max_new_tokens=MAX_NEW_TOKENS
                )
                zh_hk = simplify_to_zh_hk(
                    model, processor, text, max_new_tokens=MAX_NEW_TOKENS
                )
                row["zh_cn"] = zh_cn
                row["zh_hk"] = zh_hk

            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["sentence_en", "zh_cn", "zh_hk"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

