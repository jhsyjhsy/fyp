import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.read_csv('CLEAR_test_only_mistral_lay.csv')
prompt_ids = np.repeat([1,2,3,4,5,6], [17,17,17,16,17,16])
np.random.shuffle(prompt_ids)
df['prompt_id'] = prompt_ids
df.to_csv('CLEAR_test_use.csv', index=False)
print("done")
print(pd.Series(prompt_ids).value_counts().sort_index())
