import pandas as pd
import os
import numpy as np
import tabulate

base = "/Users/joannehui/Desktop/fyp/padchest/Reports_public"
out_dir = os.path.join(base, "crie_inputs")

# Mistral NLLB clean only (both variants)
mistral_nllb_df = pd.read_csv(os.path.join(base, "reports_500_translated_mistral_nllb_zh.csv"))
for variant in ['cn', 'tw']:
    col = f'lay_zh_{variant}'
    texts = mistral_nllb_df[col].fillna('').astype(str)
    
    # Clean/filter
    lines = [t.strip().replace('\n', ' ') for t in texts if t.strip()]
    tag = f'mistral_nllb_clean_{variant}'
    
    out_path = os.path.join(out_dir, f'{tag}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"✅ {tag}: {len(lines)} lines → {out_path}")
    print(f"   Chars/mean: {np.mean([len(l) for l in lines]):.0f}")

print("\nUpload mistral_nllb_clean_cn.txt + mistral_nllb_clean_tw.txt to CRIE!")


