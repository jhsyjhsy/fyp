# FYP: Lay Radiology Explanation + Chinese Translation

## Stage 1 – HPC (Mistral)

- Input: `reports_X.csv` with column `sentence_en`
- Output: `reports_X_translated.csv` with column `lay_translation`

Run on HPC:
conda activate llm_proj
python stage1_lay_en.py reports_X.csv


## Stage 2 – Local (NLLB)

- Input: `reports_X_translated.csv` (from HPC)
- Output: `reports_X_translated_zh.csv` with `lay_zh_cn`, `lay_zh_tw`

Run locally:
source translate_env/bin/activate
python stage2_nllb_zh_local.py /path/to/reports_X_translated.csv

undefined

