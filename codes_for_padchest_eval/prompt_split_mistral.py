import pandas as pd
import numpy as np

df = pd.read_csv("CLEAR_anonymised_with_split_70_20_10.csv")

# Elderly = >=65
df["is_elderly"] = df["Exam_Age"] >= 65

# Prompt assignment
df["prompt_id"] = np.where(
    df["is_elderly"],
    np.random.choice([2, 3, 4], size=len(df), p=[0.4, 0.3, 0.3]),  # Random 2/3/4 for elderly
    1  # Always 1 for <65
)

print("Prompt distribution:")
print(df["prompt_id"].value_counts().sort_index())

df.to_csv("CLEAR_with_prompts.csv", index=False)
