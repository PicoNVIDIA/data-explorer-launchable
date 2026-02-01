# Dev.jsonl Solutions

## Task 5: Which issuing country has the highest number of transactions?

**Data Source:** `data/payments.csv` → `issuing_country` column

**Approach:** Count transactions per issuing country, sort descending.

**Code:**
```python
import pandas as pd
df = pd.read_csv('data/payments.csv')
df['issuing_country'].value_counts().idxmax()
```

**Results:**
| Country | Transactions |
|---------|--------------|
| NL      | 29,622       |
| IT      | 28,329       |
| BE      | 23,040       |
| SE      | 21,716       |
| FR      | 14,175       |

**Answer:** `NL`

---

## Task 49: What is the top country (ip_country) for fraud?

**Data Source:** `data/payments.csv` → `ip_country`, `eur_amount`, `has_fraudulent_dispute` columns

**Approach:** 
According to manual.md
Fraud = fraudulent volume / total volume.
All volumes are specified in euros. 

**Code:**
```python
import pandas as pd
df = pd.read_csv('data/payments.csv')
fraud_vol = df[df['has_fraudulent_dispute']].groupby('ip_country')['eur_amount'].sum()
total_vol = df.groupby('ip_country')['eur_amount'].sum()
(fraud_vol / total_vol).idxmax()
```

**Results:**
| Country | Fraud Rate |
|---------|------------|
| BE      | 12.27%     |
| NL      | 12.18%     |
| SE      | 8.49%      |
| FR      | 6.90%      |

**Answer:** `B. BE`

---

## Task 70: Is Martinis_Fine_Steakhouse in danger of getting a high-fraud rate fine?

**Data Source:** `data/manual.md`

**Approach:** Search manual.md for "fine" definition. Manual only defines "fees", not "fines". No high-fraud rate fine is defined.

**Code:**
```python
# No code needed - conceptual lookup
# grep -i "fine" data/manual.md shows no fine definitions
```

**Answer:** `Not Applicable`

---

## Task 1273: For credit transactions, what would be the average fee that GlobalCard would charge for 10 EUR?

**Data Source:** `data/fees.json`

**Approach:**
Filter GlobalCard fees where is_credit is True or null (null matches all per fee_matching_guide.md).
Fee formula: `fixed_amount + rate * transaction_value / 10000`

**Code:**
```python
import pandas as pd
fees = pd.read_json('data/fees.json')
mask = (fees['card_scheme'] == 'GlobalCard') & ((fees['is_credit'].isna()) | (fees['is_credit'] == True))
(fees[mask]['fixed_amount'] + fees[mask]['rate'] * 10 / 10000).mean()
```

**Answer:** `0.120132`

---

## Task 1305: For account type H and MCC "Eating Places and Restaurants", average GlobalCard fee for 10 EUR?

**Data Source:** `data/fees.json`, `data/merchant_category_codes.csv`

**Approach:**
1. Find MCC code for "Eating Places and Restaurants" → 5812
2. Filter GlobalCard fees where account_type includes 'H' (or empty) AND merchant_category_code includes 5812 (or empty)

**Code:**
```python
import pandas as pd
fees = pd.read_json('data/fees.json')
mcc_code = 5812  # from merchant_category_codes.csv

def matches(row):
    if row['card_scheme'] != 'GlobalCard':
        return False
    if row['account_type'] and 'H' not in row['account_type']:
        return False
    if row['merchant_category_code'] and mcc_code not in row['merchant_category_code']:
        return False
    return True

matching = fees[fees.apply(matches, axis=1)]
(matching['fixed_amount'] + matching['rate'] * 10 / 10000).mean()
```

**Answer:** `0.123217`

---

## Task 1464: What fee IDs apply to account_type = R and aci = B?

**Data Source:** `data/fees.json`

**Approach:**
Filter fees where account_type includes 'R' (or empty) AND aci includes 'B' (or empty). Empty list matches all.

**Code:**
```python
import pandas as pd
fees = pd.read_json('data/fees.json')

def matches(row):
    if row['account_type'] and 'R' not in row['account_type']:
        return False
    if row['aci'] and 'B' not in row['aci']:
        return False
    return True

matching = fees[fees.apply(matches, axis=1)]
', '.join(map(str, sorted(matching['ID'].tolist())))
```

**Answer:** `1, 2, 5, 6, 8, 9, 10, 12, 14, 15, 20, 21, ...` (485 fee IDs total)

---

## Task 1681: For the 10th day of 2023, what Fee IDs apply to Belles_cookbook_store?

**Data Source:** `data/payments.csv`, `data/fees.json`, `data/merchant_data.json`

**Approach:**
1. Get merchant info (account_type=R, mcc=5942, capture_delay=1)
2. Calculate January monthly metrics (volume=113,260, fraud_pct=10.31%)
3. Get all transactions on day 10
4. For each transaction, find matching fees using all criteria

**Code:**
```python
import pandas as pd
import json

payments = pd.read_csv('data/payments.csv')
fees = pd.read_json('data/fees.json')
with open('data/merchant_data.json') as f:
    merchants = {m['merchant']: m for m in json.load(f)}

merchant = merchants['Belles_cookbook_store']
merchant_delay = int(merchant['capture_delay'])  # "1" -> 1

# January metrics
jan = payments[(payments['merchant'] == 'Belles_cookbook_store') &
               (payments['day_of_year'] <= 31)]
monthly_vol = jan['eur_amount'].sum()
fraud_pct = jan[jan['has_fraudulent_dispute']]['eur_amount'].sum() / monthly_vol * 100

# Day 10 transactions
day10 = payments[(payments['merchant'] == 'Belles_cookbook_store') &
                 (payments['day_of_year'] == 10)]

# Match fees using criteria from fee_matching_guide.md (null/empty = matches all)
# ... (full matching logic)
```

**Answer:** `286, 381, 454, 473, 477, 536, 572, 709, 741, 813`

---

## Task 1753: What are the applicable fee IDs for Belles_cookbook_store in March 2023?

**Data Source:** `data/payments.csv`, `data/fees.json`, `data/merchant_data.json`

**Approach:**
Same as Task 1681, but for all transactions in March (days 60-90).
Monthly metrics: volume=116,436, fraud_pct=10.25%

**Code:**
```python
import pandas as pd
from fee_matcher import *

# March 2023: days 60-90
start_day, end_day = 60, 90
mar_txns = payments[
    (payments['merchant'] == 'Belles_cookbook_store') &
    (payments['day_of_year'] >= start_day) &
    (payments['day_of_year'] <= end_day)
]
# Use same matching logic as Task 1681, applied to all March transactions
```

**Answer:** `36, 51, 53, 64, 107, 123, 150, 163, 231, 249, 276, 286, 347, 381, 384, 394, 428, 454, 473, 477, 536, 556, 572, 595, 608, 626, 680, 709, 725, 741, 813, 868, 939, 960`

---

## Task 1871: In January 2023, what delta if fee ID=384's rate changed to 1?

**Data Source:** `data/payments.csv`, `data/fees.json`

**Approach:**
1. Fee 384: card_scheme=NexPay, is_credit=True, aci=['C','B'], rate=14
2. Find all January transactions matching fee 384 criteria
3. Delta = (new_rate - old_rate) * total_matching_amount / 10000 = (1-14) * total / 10000

**Code:**
```python
import pandas as pd
payments = pd.read_csv('data/payments.csv')
fees = pd.read_json('data/fees.json')

jan_txns = payments[
    (payments['merchant'] == 'Belles_cookbook_store') &
    (payments['year'] == 2023) &
    (payments['day_of_year'] <= 31) &
    (payments['card_scheme'] == 'NexPay') &
    (payments['is_credit'] == True) &
    (payments['aci'].isin(['C', 'B']))
]
total = jan_txns['eur_amount'].sum()  # 729.31
delta = (1 - 14) * total / 10000      # -0.948103
```

**Answer:** `-0.94810300000000`

---

## Task 2697: For Belles_cookbook_store in January, move fraud txns to different ACI for lowest fees?

**Data Source:** `data/payments.csv`, `data/fees.json`, `data/merchant_data.json`

**Approach:**
1. Find all January fraudulent transactions (94 txns, all have ACI=G)
2. For each possible ACI, calculate total fees if fraud txns were moved to that ACI
3. Select ACI with lowest total fee

**Code:**
```python
import pandas as pd
from fee_matcher import *

jan_fraud = payments[
    (payments['merchant'] == 'Belles_cookbook_store') &
    (payments['day_of_year'] <= 31) &
    (payments['has_fraudulent_dispute'] == True)
]

# Calculate min fee per txn for each target ACI
for aci in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    total = sum(get_min_fee(txn, aci) for txn in jan_fraud)
    print(f'{aci}: {total:.2f}')
```

**Note:** Not all card schemes have matching fees for all ACIs. GlobalCard with ACI E = 13.56.

**Answer:** `E:13.57`

---
