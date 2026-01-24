import pandas as pd

# Load the MCC → description mapping
mcc_df = pd.read_csv('data/context/merchant_category_codes.csv')

# Load the fees data
fees_df = pd.read_json('data/context/fees.json')

# Identify MCC codes for the description "Eating Places and Restaurants"
target_mccs = mcc_df.loc[mcc_df['description'] == 'Eating Places and Restaurants', 'mcc'].tolist()

# Filter fees for:
#   - card_scheme == 'GlobalCard'
#   - account_type == 'H' or null (wildcard)
#   - merchant_category_code belonging to the target MCC(s)
filtered_fees = fees_df[
    (fees_df['card_scheme'] == 'GlobalCard') &
    (fees_df['account_type'].apply(lambda x: x == 'H' or pd.isnull(x))) &
    (fees_df['merchant_category_code'].isin(target_mccs))
]

# Remove rows where required numeric fields are missing
filtered_fees = filtered_fees.dropna(subset=['fixed_amount', 'rate'])

# Transaction value in EUR
transaction_value = 10  # EUR

# Compute fee for each matching record: fee = fixed_amount + rate * transaction_value / 10000
filtered_fees['fee'] = filtered_fees['fixed_amount'] + filtered_fees['rate'] * transaction_value / 10000

# Calculate the average fee across all matching records
average_fee = filtered_fees['fee'].mean()

# Output the average fee rounded to 6 decimal places
print(f"{average_fee:.6f}")
