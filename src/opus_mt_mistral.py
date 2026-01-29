import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch

model_name = "Helsinki-NLP/opus-mt-en-zh"

print("Loading Opus-MT model...")
device = "cpu"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

df = pd.read_csv("/home/huisinyu/fyp/fyp/reports_500_translated.csv")  # Mistral output
lay_en = df["lay_translation"].fillna("").astype(str)


def translate_batch(texts, batch_size=4, max_length=256):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if all((not t.strip()) for t in batch):
            outputs.extend([""] * len(batch))
            continue

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_length=max_length,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


print("Translating Mistral lay English to Chinese with Opus-MT...")
texts = list(lay_en)
opus_zh = translate_batch(texts, batch_size=8, max_length=256)

df["opus_lay_zh_tw_like"] = opus_zh
df["opus_lay_zh_cn_like"] = opus_zh

output_csv = "reports_500_translated_zh_opus.csv"
df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

