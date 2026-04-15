import pandas as pd
import re

# Load your mistral_zh CSV (adjust path as needed, e.g., for HPC: /data/users/huisinyu/.../mistral_zh.csv)
df = pd.read_csv('/Users/joannehui/Desktop/fyp/padchest/Reports_public/reports_500_mistral_zh.csv')  # Replace with exact filename/path

def clean_mistral_zh(text):
    """
    Clean Mistral ZH outputs (cn/hk variants):
    - Remove [INST]..[/INST] prompts (full blocks, DOTALL for multiline)
    - Strip leading "Sentence: " or numbered prefixes
    - Keep first translation only (cut before "Or," or "。或，")
    - Remove remaining numbered lists (1., 2., etc.)
    Handles your prior contamination patterns.
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 1. Remove full prompt blocks
    text = re.sub(r'\[INST\].*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    
    # 2. Remove leading "Sentence: " (English/Chinese variants)
    text = re.sub(r'^Sentence:\s*', '', text)
    text = re.sub(r'^句子：\s*', '', text)
    
    # 3. Cut before first "Or," / "。或，" (keep first variant)
    if 'Or,' in text:
        text = text.split('Or,', 1)[0]
    if '。或，' in text:
        text = text.split('。或，', 1)[0]
    
    # 4. Strip numbered lists remaining
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # 5. Clean whitespace/newlines
    text = re.sub(r'\n+', ' ', text).strip()
    
    return text

# Apply to ZH columns (from your prior chats: mistral_zh_cn, mistral_zh_hk)
df['mistral_zh_cn_clean'] = df['mistral_zh_cn'].apply(clean_mistral_zh)
df['mistral_zh_hk_clean'] = df['mistral_zh_hk'].apply(clean_mistral_zh)

# Stats: Expect ~70% char drop (e.g., 250→75 chars mean)
print("Before cleaning:")
print("CN chars mean:", df['mistral_zh_cn'].str.len().mean())
print("HK chars mean:", df['mistral_zh_hk'].str.len().mean())
print("\nAfter cleaning:")
print("CN chars mean:", df['mistral_zh_cn_clean'].str.len().mean())
print("HK chars mean:", df['mistral_zh_hk_clean'].str.len().mean())
print("Rows with '或':", (~df['mistral_zh_cn_clean'].str.contains('或', na=False)).sum())

# Save cleaned (keep originals + cleans for comparison/CRIE)
df[['StudyID', 'ImageID', 'sentence_en', 'mistral_zh_cn', 'mistral_zh_cn_clean', 
    'mistral_zh_hk', 'mistral_zh_hk_clean']].to_csv('mistral_zh_clean.csv', index=False)
print("\n✅ Saved: mistral_zh_clean.csv")
print("\nSample cleans:")
print(df[['mistral_zh_cn', 'mistral_zh_cn_clean']].head(3))
