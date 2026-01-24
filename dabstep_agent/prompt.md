You are analyzing payment transaction data for a data science benchmark.

Available data files in 'data/context/':
- merchant_category_codes.csv (CSV):
    Columns: Unnamed: 0, mcc, description
    Sample row: {"Unnamed: 0": 0, "mcc": 1520, "description": "General Contractors - Residential and Commercial"}
- merchant_data.json (JSON, array_of_objects):
    Keys: merchant, capture_delay, acquirer, merchant_category_code, account_type
    Sample record: {"merchant": "Crossfit_Hanna", "capture_delay": "manual", "acquirer": ["gringotts", "the_savings_and_loan_bank", "bank_of_springfield", "dagoberts_vault"], "merchant_category_code": 7997, "account_type": "F"}
- fees.json (JSON, array_of_objects):
    Keys: ID, card_scheme, account_type, capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code, is_credit, aci, fixed_amount, rate, intracountry
    Sample record: {"ID": 1, "card_scheme": "TransactPlus", "account_type": [], "capture_delay": null, "monthly_fraud_level": null, "monthly_volume": null, "merchant_category_code": [8000, 8011, 8021, 8031, 8041, 7299, 9399, 8742], "is_credit": false, "aci": ["C", "B"], "fixed_amount": 0.1, "rate": 19, "intracountry": null}
- payments.csv (CSV):
    Columns: psp_reference, merchant, card_scheme, year, hour_of_day, minute_of_hour, day_of_year, is_credit, eur_amount, ip_country, issuing_country, device_type, ip_address, email_address, card_number, shopper_interaction, card_bin, has_fraudulent_dispute, is_refused_by_adyen, aci, acquirer_country
    Sample row: {"psp_reference": 20034594130, "merchant": "Crossfit_Hanna", "card_scheme": "NexPay", "year": 2023, "hour_of_day": 16, "minute_of_hour": 21, "day_of_year": 12, "is_credit": false, "eur_amount": 151.74, "ip_country": "SE", "issuing_country": "SE", "device_type": "Windows", "ip_address": "pKPYzJqqwB8TdpY0jiAeQw", "email_address": "0AKXyaTjW7H4m1hOWmOKBQ", "card_number": "uRofX46FuLUrSOTz8AW5UQ", "shopper_interaction": "Ecommerce", "card_bin": 4802, "has_fraudulent_dispute": false, "is_refused_by_adyen": false, "aci": "F", "acquirer_country": "NL"}
- acquirer_countries.csv (CSV):
    Columns: Unnamed: 0, acquirer, country_code
    Sample row: {"Unnamed: 0": 0, "acquirer": "gringotts", "country_code": "GB"}

RELEVANT DOCUMENTATION (from research phase):
- fee: fee = fixed_amount + rate * transaction_value / 10000  
- rate: integer variable rate multiplied by transaction value and divided by 10000  
- fixed_amount: float fixed fee in euros per transaction for the rule  
- scheme: card networks such as Visa or Mastercard referred to as schemes

RELEVANT FILES AND COLUMNS (from explore phase):
The following code shows how to access the relevant data:
```python
import json
import pandas as pd

# Load the fees data (already available as `fees_df` but shown here for completeness)
with open('data/context/fees.json', 'r') as f:
    fees_data = json.load(f)
fees_df = pd.json_normalize(fees_data)

# Select the columns needed for the query
relevant_columns = ['card_scheme', 'is_credit', 'fixed_amount', 'rate']
print(fees_df[relevant_columns])
```


Regarding field filtering:
* when filtering, treat null as a wild card.
* Example: To filter where 'column' matches 'value', you should check 'column' matches 'value' OR is null (wildcard):
  before: df[(df['column'] == 'value') 
  after: df[(df['column'] == 'value') | (df['column'].isnull())]

QUESTION: For credit transactions, what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR?

GUIDELINES: Answer must be just a number expressed in EUR rounded to 6 decimals. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'


INSTRUCTIONS:
1. Use the provided documentation information above to understand the terms and definitions
2. Use the explore code above as a starting point for loading the relevant data
3. Use execute_python_code to analyze the data and answer the question
4. Provide the final answer following the guidelines exactly