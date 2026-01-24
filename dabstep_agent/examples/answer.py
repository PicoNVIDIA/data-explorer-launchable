"""
Calculate the average fee for GlobalCard credit transactions with a 10 EUR transaction value.

Based on terms_research.json definitions:
- credit transactions: is_credit=True OR is_credit=null (null applies to all)
- card_scheme: Filter for 'GlobalCard'
- fee calculation: fee = fixed_amount + rate * transaction_value / 10000
- average fee: arithmetic mean of all applicable fee calculations
"""

import json

# Load fees data
with open('data/context/fees.json', 'r') as f:
    fees = json.load(f)

# Filter for GlobalCard credit transactions
# Per terms_research.json: is_credit=True OR is_credit=null (null matches all values)
globalcard_credit_fees = [
    fee for fee in fees
    if fee['card_scheme'] == 'GlobalCard'
    and (fee['is_credit'] == True or fee['is_credit'] is None)
]

print(f'Number of applicable GlobalCard credit fee rules: {len(globalcard_credit_fees)}')

# Calculate fee for each rule with transaction_value = 10 EUR
# Formula from manual.md line 92: fee = fixed_amount + rate * transaction_value / 10000
transaction_value = 10  # EUR as specified in the question
calculated_fees = []

for fee in globalcard_credit_fees:
    fixed_amount = fee['fixed_amount']
    rate = fee['rate']
    # Apply fee calculation formula
    calculated_fee = fixed_amount + (rate * transaction_value / 10000)
    calculated_fees.append(calculated_fee)

print(f'\nTransaction value: {transaction_value} EUR')
print(f'Fees calculated: {len(calculated_fees)}')
print(f'Fee range: {min(calculated_fees):.6f} to {max(calculated_fees):.6f} EUR')

# Calculate average (arithmetic mean)
average_fee = sum(calculated_fees) / len(calculated_fees)

print(f'\nAverage fee for GlobalCard credit transactions (10 EUR): {average_fee:.6f} EUR')
print(f'Answer: {average_fee:.6f}')
