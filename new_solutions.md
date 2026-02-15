# Insights and Examples

**Key insight:** When a question asks for the "most expensive" or "cheapest" MCC/card_scheme "in general", use the **average** fee across all applicable fee rules (mean), not the maximum.

## Helper Module (`helper.py`)

Shared functions for fee calculation tasks.

### Function Signatures

```python
# Data Loading
def load_fees() -> List[Dict]
def load_payments() -> pd.DataFrame
def get_merchant_info(merchant_name: str) -> Optional[Dict]

# Field Matching (null/empty = applies to all)
def matches_list_field(field_value: Any, target: Any) -> bool

# Fee Calculation
def calculate_fee(fixed_amount: float, rate: float, transaction_value: float) -> float

# Transaction Filtering
def filter_merchant_transactions(df: pd.DataFrame, merchant: str, year: int, month: int) -> pd.DataFrame
def calculate_monthly_metrics(df: pd.DataFrame) -> Dict[str, float]
def add_intracountry_flag(df: pd.DataFrame, acquirer_country: Optional[str] = None) -> pd.DataFrame

# Fee Rule Matching
def find_matching_fees(fees: List[Dict], card_scheme: str, account_type: str, mcc: int, is_credit: bool, aci: str, intracountry: bool, capture_delay: Any, monthly_vol: float, fraud_pct: float) -> List[Dict]
```

---

## Which issuing country has the highest number of transactions?

**Data Source:** `data/context/payments.csv` → `issuing_country` column

**Approach:** Count transactions per issuing country, sort descending.

**Code:**
```python
import pandas as pd
df = pd.read_csv('data/context/payments.csv')
df['issuing_country'].value_counts().idxmax()
```

---

## What is the top country (ip_country) for fraud?

**Data Source:** `data/context/payments.csv` → `ip_country`, `eur_amount`, `has_fraudulent_dispute` columns

**Approach:**
According to manual.md
Fraud = fraudulent volume / total volume.
All volumes are specified in euros.

**Code:**
```python
import pandas as pd
df = pd.read_csv('data/context/payments.csv')
fraud_vol = df[df['has_fraudulent_dispute']].groupby('ip_country')['eur_amount'].sum()
total_vol = df.groupby('ip_country')['eur_amount'].sum()
(fraud_vol / total_vol).idxmax()
```

---

## Is Martinis_Fine_Steakhouse in danger of getting a high-fraud rate fine?

**Data Source:** `data/context/manual.md`

**Approach:** Search manual.md for "fine" definition. Manual only defines "fees", not "fines". No high-fraud rate fine is defined.

**Code:**
```python
# No code needed - conceptual lookup
# grep -i "fine" data/context/manual.md shows no fine definitions
```

---

## For credit transactions, what would be the average fee that GlobalCard would charge for 10 EUR?

**Data Source:** `data/context/fees.json`

**Approach:**
Filter GlobalCard fees where is_credit is True or null (null matches all per fee_matching_guide.md).
Fee formula: `fixed_amount + rate * transaction_value / 10000`

**Code:**
```python
import pandas as pd
fees = pd.read_json('data/context/fees.json')
mask = (fees['card_scheme'] == 'GlobalCard') & ((fees['is_credit'].isna()) | (fees['is_credit'] == True))
(fees[mask]['fixed_amount'] + fees[mask]['rate'] * 10 / 10000).mean()
```

---

## Average GlobalCard fee for account type H, MCC 5812, 10 EUR

**Data Source:** `data/context/fees.json`, `data/context/merchant_category_codes.csv`

**Approach:** Filter GlobalCard fees where account_type includes 'H' (or empty) AND merchant_category_code includes 5812 (or empty).

**Code:**
```python
from helper import load_fees, matches_list_field, calculate_fee

fees = [f for f in load_fees()
        if f['card_scheme'] == "GlobalCard"
        and matches_list_field(f['account_type'], "H")
        and matches_list_field(f['merchant_category_code'], 5812)]

calculated_fees = [calculate_fee(f['fixed_amount'], f['rate'], 10.0) for f in fees]
avg = sum(calculated_fees) / len(calculated_fees)
```

---

## Fee IDs for account_type=R and aci=B

**Data Source:** `data/context/fees.json`

**Approach:** Filter fees where account_type includes 'R' (or empty) AND aci includes 'B' (or empty).

**Code:**
```python
from helper import load_fees, matches_list_field

matching = sorted([f['ID'] for f in load_fees()
                   if matches_list_field(f.get('account_type'), 'R')
                   and matches_list_field(f.get('aci'), 'B')])
```

---

## Fee IDs for Belles_cookbook_store on day 10 of January 2023

**Data Source:** `data/context/payments.csv`, `data/context/fees.json`, `data/context/merchant_data.json`

**Approach:**
1. Get merchant info (account_type=R, mcc=5942, capture_delay=1)
2. Calculate January monthly metrics (volume=113,260, fraud_pct=10.31%)
3. Get unique transaction combinations on day 10
4. Match fees using all criteria

**Code:**
```python
from helper import (
    load_fees, load_payments, get_merchant_info, find_matching_fees,
    filter_merchant_transactions, calculate_monthly_metrics, add_intracountry_flag
)

fees, payments = load_fees(), load_payments()
m = get_merchant_info('Belles_cookbook_store')

metrics = calculate_monthly_metrics(filter_merchant_transactions(payments, 'Belles_cookbook_store', 2023, 1))
day_10 = add_intracountry_flag(payments[(payments['merchant'] == 'Belles_cookbook_store') &
                                         (payments['year'] == 2023) &
                                         (payments['day_of_year'] == 10)])
combos = day_10[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()

ids = set()
for _, txn in combos.iterrows():
    for fee in find_matching_fees(fees, txn['card_scheme'], m['account_type'], m['merchant_category_code'],
                                   txn['is_credit'], txn['aci'], txn['intracountry'],
                                   m['capture_delay'], metrics['volume'], metrics['fraud_rate_pct']):
        ids.add(fee['ID'])
ids = sorted(ids)
```

---

## Fee delta if fee ID=384 rate changed to 1

**Data Source:** `data/context/payments.csv`, `data/context/fees.json`

**Approach:**
1. Fee 384: card_scheme=NexPay, is_credit=True, aci=['C','B'], rate=14
2. Find matching January transactions
3. Delta = (1 - 14) * total_amount / 10000

**Code:**
```python
from decimal import Decimal, getcontext
from helper import load_payments, filter_merchant_transactions

getcontext().prec = 50
txns = filter_merchant_transactions(load_payments(), 'Belles_cookbook_store', 2023, 1)

matching = txns[((txns['card_scheme'] == 'NexPay') | txns['card_scheme'].isnull()) &
                ((txns['is_credit'] == True) | txns['is_credit'].isnull()) &
                (txns['aci'].isin(['C', 'B']) | txns['aci'].isnull())]

total = sum(Decimal(str(a)) for a in matching['eur_amount'])
delta = -Decimal('13') * total / Decimal('10000')
```

---

## For Belles_cookbook_store in January, which ACI should fraudulent transactions be shifted to minimize fees?

**Data Source:** `data/context/payments.csv`, `data/context/fees.json`, `data/context/merchant_data.json`

**Approach:**
1. Get January fraudulent transactions (94 txns)
2. For each ACI (A-G), calculate total fees if fraud txns used that ACI
3. Exclude the current ACI(s) already used by fraudulent transactions (the question asks for a *different* ACI)
4. Select ACI with lowest total fee among the remaining alternatives

**Code:**
```python
from helper import (
    load_fees, load_payments, get_merchant_info, find_matching_fees,
    filter_merchant_transactions, calculate_monthly_metrics, add_intracountry_flag, calculate_fee
)

fees, payments = load_fees(), load_payments()
m = get_merchant_info('Belles_cookbook_store')
mcc, acct, cap_del = m['merchant_category_code'], m['account_type'], int(m['capture_delay'])

txns = filter_merchant_transactions(payments, 'Belles_cookbook_store', 2023, 1)
txns = add_intracountry_flag(txns)
fraud_txns = txns[txns['has_fraudulent_dispute'] == True]

metrics = calculate_monthly_metrics(txns)
vol, fraud_lvl = metrics['volume'], metrics['fraud_rate_pct']

# Get current ACIs used by fraudulent transactions
current_acis = set(fraud_txns['aci'].unique())

results = {}
for aci in payments.aci.unique():
    total = 0.0
    for _, t in fraud_txns.iterrows():
        matching = find_matching_fees(fees, t['card_scheme'], acct, mcc, t['is_credit'], aci,
                                       t['intracountry'], cap_del, vol, fraud_lvl)
        for f in matching:
            total += calculate_fee(f['fixed_amount'], f['rate'], t['eur_amount'])
    results[aci] = round(total, 2)

# Exclude current ACI(s) - the question asks to move to a DIFFERENT ACI
other_results = {k: v for k, v in results.items() if k not in current_acis}
best = min(other_results, key=other_results.get)
```

---

## Which card scheme should a merchant steer traffic to for minimum fees in January 2023?

**Data Source:** `data/context/payments.csv`, `data/context/fees.json`, `data/context/merchant_data.json`

**Approach:**
1. Get all merchant transactions for the target month
2. For each card scheme, simulate all transactions as if processed under that scheme
3. Sum all matching fees per transaction
4. Exclude the current card scheme(s) already used by the transactions (the question asks to steer to a *different* scheme)
5. Select the card scheme with the lowest total fee among the remaining alternatives

**Code:**
```python
from helper import (
    load_fees, load_payments, get_merchant_info, find_matching_fees,
    filter_merchant_transactions, calculate_monthly_metrics, add_intracountry_flag, calculate_fee
)

fees, payments = load_fees(), load_payments()
m = get_merchant_info('MERCHANT_NAME')
mcc, acct, cap_del = m['merchant_category_code'], m['account_type'], int(m['capture_delay'])

txns = filter_merchant_transactions(payments, 'MERCHANT_NAME', 2023, 1)
txns = add_intracountry_flag(txns)
metrics = calculate_monthly_metrics(txns)
vol, fraud_lvl = metrics['volume'], metrics['fraud_rate_pct']

# Get current card schemes used by the transactions
current_schemes = set(txns['card_scheme'].unique())

results = {}
for scheme in ['GlobalCard', 'NexPay', 'SwiftCharge', 'TransactPlus']:
    total = 0.0
    for _, t in txns.iterrows():
        matching = find_matching_fees(fees, card_scheme=scheme, account_type=acct, mcc=mcc,
                                       is_credit=t['is_credit'], aci=t['aci'],
                                       intracountry=t['intracountry'], capture_delay=cap_del,
                                       monthly_vol=vol, fraud_pct=fraud_lvl)
        for f in matching:
            total += calculate_fee(f['fixed_amount'], f['rate'], t['eur_amount'])
    results[scheme] = total

# Exclude current card scheme(s) - the question asks to steer to a DIFFERENT scheme
other_results = {k: v for k, v in results.items() if k not in current_schemes}
best = min(other_results, key=other_results.get)
```
