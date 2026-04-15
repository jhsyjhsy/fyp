import pandas as pd

df = pd.read_csv('CLEAR_with_prompts.csv')

# Adjust column name based on your split (check above)
train_df = df[df['split'] == 'train']  # or df['train_dev_test'] == 'train'
train_df.to_csv('CLEAR_train_only_mistral_lay.csv', index=False)

print(f'Train only: {len(train_df)} rows')
print(train_df.head())

