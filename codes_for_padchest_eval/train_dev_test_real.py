import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("CLEAR_anonymised(Sheet1).csv")

df = df.dropna(subset=["Report_Content"])

# 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# 20% dev, 10% test (i.e. 2/3 and 1/3 of temp)
dev_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,       # 1/3 of 30% = 10% of total
    random_state=42,
    shuffle=True
)

print("Sizes:", len(train_df), len(dev_df), len(test_df))

train_df["split"] = "train"
dev_df["split"] = "dev"
test_df["split"] = "test"

full = pd.concat([train_df, dev_df, test_df], axis=0)
full.to_csv("CLEAR_anonymised_with_split_70_20_10.csv", index=False)

