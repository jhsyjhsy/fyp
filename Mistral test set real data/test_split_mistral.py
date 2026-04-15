import pandas as pd

df = pd.read_csv('CLEAR_with_prompts.csv')

# Adjust column name based on your split (check above)
test_df = df[df['split'] == 'test']  
test_df.to_csv('CLEAR_test_only_mistral_lay.csv', index=False)

print(f'Test only: {len(test_df)} rows')
print(test_df.head())