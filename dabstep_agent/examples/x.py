import pandas as pd

# Load fees data
fees_df = pd.read_json('data/context/fees.json')

globalcard_credit = fees_df[(fees_df['card_scheme'] == 'GlobalCard') & ((fees_df['is_credit'] == True) | (fees_df['is_credit'].isnull()))]
globalcard_credit_clean = globalcard_credit#.dropna(subset=['fixed_amount', 'rate'])

# Compute fee for transaction_value = 10
transaction_value = 10
globalcard_credit_clean['fee'] = globalcard_credit_clean['fixed_amount'] + globalcard_credit_clean['rate'] * transaction_value / 10000

# Compute average fee
average_fee = globalcard_credit_clean['fee'].mean()
print(average_fee)
