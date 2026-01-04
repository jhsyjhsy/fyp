# FYP: Lay Radiology Explanation + Chinese Translation

This project simplifies chest X‑ray report sentences into lay English and then translates them into Simplified and Traditional Chinese for patients in Hong Kong and other Chinese‑speaking regions.

## Data

- **Input CSV**: `reports_X.csv`
- **Core input column**: `sentence_en` (original English sentence)
- **Main output columns**:
  - `lay_translation` (Mistral lay English)
  - `medgemma_lay_en` (MedGemma lay English)
  - Chinese columns (NLLB / OPUS‑MT), for example:
    - `nllb_zh_Hans_mistral`, `nllb_zh_Hant_mistral`
    - `nllb_zh_Hans_medgemma`, `nllb_zh_Hant_medgemma`
    - `opusmt_zh_mistral`, `opusmt_zh_medgemma`

---

## Stage 1 – HPC (Mistral lay English)

- **Input**: `reports_X.csv` with column `sentence_en`  
- **Output**: `reports_X_translated.csv` with column `lay_translation`  
- **Example command** (on HPC):

```bash
conda activate llm_proj
python stage1_lay_en.py reports_X.csv
```

## Stage 1b – HPC (MedGemma lay English)

- **Input**: reports_500.csv with column sentence_en
- **Output**: reports_500_medgemma.csv with column medgemma_lay_en

Example command (on HPC):

```bash
conda activate /usersdata/huisinyu/conda_envs/medgemma_cpu
python medgemma_simplify_csv.py
```

## Stage 2 – Local (NLLB Chinese)
- **Input (Mistral path):** reports_X_translated.csv
- **Input (MedGemma path):** reports_500_medgemma.csv
- **Outputs (examples):**
reports_X_translated_nllb_bi.csv with:
nllb_zh_Hans_mistral
nllb_zh_Hant_mistral

reports_500_medgemma_nllb_bi.csv with:
nllb_zh_Hans_medgemma
nllb_zh_Hant_medgemma

Example commands (local):

```bash
conda activate nllb_env
python nllb_translate_mistral_local.py
python nllb_translate_medgemma_local.py
```

## Stage 3 – Local / HPC (OPUS‑MT Chinese)
- **Input (Mistral):** lay_translation from reports_X_translated.csv
- **Input (MedGemma):** medgemma_lay_en from reports_500_medgemma.csv
- **Outputs (examples):**
reports_X_opusmt.csv with column opusmt_zh_mistral
reports_500_medgemma_opusmt.csv with column opusmt_zh_medgemma
Example commands:

```bash
python opus_mt_mistral.py
python opus_mt_medgemma.py
``

