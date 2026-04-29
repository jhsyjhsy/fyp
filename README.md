### Usage (Inference Example)
To run the 2-step pipeline (Mistral simplification followed by NLLB translation):
1. **Simplify:** Run `python mistral_real_train/mistral_lay_clear.py` to generate English lay summaries.
2. **Translate:** Run `python src/nllb_translate_medgemma_local.py` to generate Traditional/Simplified Chinese versions.

---

## 🛡️ Privacy Note
The QEH dataset used in the refinement phase is anonymized. Patient identifiers, HKID numbers, and HA case numbers have been removed to comply with patient-data confidentiality standards.

***

# AI-Assisted Radiology Report Simplification and Translation

This repository contains the source code, prompts, and evaluation pipelines for research into simplifying and translating Chest X-ray (CXR) reports. The goal is to bridge the health literacy gap for patients in multilingual clinical settings (specifically Hong Kong) by transforming technical medical jargon into patient-friendly English, Traditional Chinese, and Simplified Chinese.

## 🔗 Model Links (Hugging Face)
The fine-tuned models developed for this project are available on Hugging Face:
* **Mistral Finetune v1:** [jhsyjhsy/mistral_finetune_v1](https://huggingface.co/jhsyjhsy/mistral_finetune_v1)
* **MedGemma-4B-IT Finetuned:** [jhsyjhsy/medgemma-4b-it_fyp_finetuned](https://huggingface.co/jhsyjhsy/medgemma-4b-it_fyp_finetuned)

---

## 📖 Project Overview
Radiology reports serve as the primary communication tool between clinicians, but their complexity often leads to patient distress and misinterpretation. This project leverages Large Language Models (LLMs) to automate the creation of lay summaries.

### Two-Phase Methodology
1.  **Screening Phase:** Comparison of 6 candidate pipelines using the **PadChest-GR** dataset to identify the most promising architectures.
2.  **Refinement Phase:** Fine-tuning models using **LoRA (Low-Rank Adaptation)** on a dataset of 993 anonymized CXR reports from Queen Elizabeth Hospital (QEH), collected in January 2026.

### Patient-Centric Design
Prompts were engineered to target specific health literacy levels in Hong Kong:
*   **Form 3-5:** Optimized for the general adult population (9 years of compulsory education).
*   **Primary 6:** Optimized for older adults (65+) who may have lower formal educational backgrounds.
---

## 📂 Repository Structure

The repository is organized to facilitate the reproduction of both the 1-step (direct generation) and 2-step (simplification then translation) pipelines.

| Directory / File | Description |
| :--- | :--- |
| **`real_thing/medgemma_codes`** | Core refinement scripts for the 1-step MedGemma pipeline. |
| ↳ `finetune_medgemma.py` | Implementation of LoRA SFT for the MedGemma-4B-IT backbone. |
| ↳ `medgemma_pseudolabel_clear.py` | Script for generating and cleaning pseudo-labels for training. |
| **`mistral_real_train`** | Training and layout scripts for the Mistral-7B-Instruct-v0.3 backbone. |
| ↳ `finetune_clear_mistral_lora.py` | LoRA training implementation for Mistral English simplification. |
| **`Mistral test set real data`** | Evaluation and inference scripts for the final held-out test set ($N=100$). |
| ↳ `inference_test.py` | Main script for generating test set outputs. |
| **`Padchest_pipelines`** | Phase 1 screening scripts including NLLB and Opus-MT integration. |
| **`src`** | Global utility scripts for translation and CSV data cleaning. |
| **`2nd round_eval`** | Scripts for English readability analysis (FK/GF metrics). |
| ↳ `FK_GF_both.ipynb` | Notebook for Flesch-Kincaid and Gunning Fog index calculation. |
| **`upload`** | General evaluation tools and Chinese readability metrics. |
| ↳ `CRIE_NLLB+mistral_new.py` | Main evaluation script for CRIE 3.0 Chinese readability. |

---

## ⚙️ Technical Specifications

### Training Parameters (LoRA)
The models were refined to balance factual accuracy with readability using the following hyperparameters:

*   **MedGemma-4B-IT:**
    *   Rank $r = 16$, $\alpha = 32$, Dropout $= 0.1$.
    *   3 Epochs, Learning Rate $1\times10^{-5}$.
*   **Mistral-7B-Instruct-v0.3:**
    *   Rank $r = 8$, $\alpha = 16$, Dropout $= 0.05$.
    *   1.5 Epochs, Learning Rate $1\times10^{-5}$.

---

## 📊 Evaluation & Safety
The project employs a multi-dimensional evaluation strategy:
1.  **Readability:** Flesch-Kincaid (FK) and Gunning Fog (GF) for English; CRIE 3.0 for Chinese.
2.  **Safety (CRIMSON):** A clinically grounded LLM metric used to screen for hallucinations and omissions.
3.  **Clinician Review:** Manual safety assessment by experienced radiologists for the final 100-report test set.

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/jhsyjhsy/codes.git](https://github.com/jhsyjhsy/codes.git)
cd codes
pip install -r requirements.txt

