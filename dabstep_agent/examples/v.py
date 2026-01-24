import pandas as pd
fees_df = pd.read_json('data/context/fees.json', orient='records')
print(fees_df.head())
#print(fees_df['account_type'].unique()[:20])
