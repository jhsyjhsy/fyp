import pandas as pd

df = pd.read_csv('CLEAR_with_prompts.csv')

# Adjust column name based on your split (check above)
dev_df = df[df['split'] == 'dev']  
dev_df.to_csv('CLEAR_dev_only_mistral_lay.csv', index=False)

print(f'Dev only: {len(dev_df)} rows')
print(dev_df.head())