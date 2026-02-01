# Fee Matching Rules Reference

Quick reference for matching payment transactions to fee rules.

## Data Files

| File | Description |
|------|-------------|
| `data/payments.csv` | Transaction data (card_scheme, is_credit, eur_amount, issuing_country, acquirer_country, aci, etc.) |
| `data/fees.json` | Fee rules with matching criteria and fee calculation parameters |
| `data/merchant_data.json` | Merchant info (account_type, capture_delay, merchant_category_code) |

---

## Matching Rules & Functions

### 1. Card Scheme
**Rule:** Must match exactly.
```python
if fee['card_scheme'] != txn['card_scheme']:
    continue
```

---

### 2. Account Type
**Rule:** Empty list `[]` matches all. Otherwise merchant's account_type must be in the list.
```python
if fee['account_type'] and account_type not in fee['account_type']:
    continue
```

---

### 3. Merchant Category Code (MCC)
**Rule:** Empty list `[]` matches all. Otherwise merchant's MCC must be in the list.
```python
if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']:
    continue
```

---

### 4. Is Credit
**Rule:** `null` matches all. Otherwise must match transaction's is_credit.
```python
if fee['is_credit'] is not None and fee['is_credit'] != is_credit:
    continue
```

---

### 5. ACI (Authorization Characteristics Indicator)
**Rule:** Empty list `[]` matches all. Otherwise transaction's ACI must be in the list.

| ACI | Description |
|-----|-------------|
| A | Card present - Non-authenticated |
| B | Card Present - Authenticated |
| C | Tokenized card with mobile device |
| D | Card Not Present - Card On File |
| E | Card Not Present - Recurring Bill Payment |
| F | Card Not Present - 3-D Secure |
| G | Card Not Present - Non-3-D Secure |

```python
if fee['aci'] and aci not in fee['aci']:
    continue
```

---

### 6. Intracountry
**Rule:** `null` matches all. `1.0` = domestic (issuer country == acquirer country), `0.0` = international.
```python
# Calculate from transaction
intracountry = 1.0 if issuing_country == acquirer_country else 0.0

if fee['intracountry'] is not None and fee['intracountry'] != intracountry:
    continue
```

---

### 7. Capture Delay
**Rule:** `null` matches all. Compare merchant's capture_delay (as integer) against rule ranges.

| Rule Value | Matches |
|------------|---------|
| `immediate` | merchant_delay == 0 |
| `<3` | merchant_delay < 3 |
| `3-5` | 3 <= merchant_delay <= 5 |
| `>5` | merchant_delay > 5 |
| `manual` | merchant_delay == 'manual' |

```python
def matches_capture_delay(rule_delay, merchant_delay):
    if rule_delay is None:
        return True
    if rule_delay == '<3' and merchant_delay < 3:
        return True
    if rule_delay == '3-5' and 3 <= merchant_delay <= 5:
        return True
    if rule_delay == '>5' and merchant_delay > 5:
        return True
    if rule_delay == 'immediate' and merchant_delay == 0:
        return True
    if rule_delay == 'manual':
        return merchant_delay == 'manual'
    return False
```

---

### 8. Monthly Volume
**Rule:** `null` matches all. Compare merchant's monthly volume (EUR) against rule ranges.

| Rule Value | Matches |
|------------|---------|
| `<100k` | volume < 100,000 |
| `100k-1m` | 100,000 <= volume < 1,000,000 |
| `1m-5m` | 1,000,000 <= volume < 5,000,000 |
| `>5m` | volume >= 5,000,000 |

```python
def matches_monthly_volume(rule_vol, monthly_vol):
    if rule_vol is None:
        return True
    if rule_vol == '<100k':
        return monthly_vol < 100000
    if rule_vol == '100k-1m':
        return 100000 <= monthly_vol < 1000000
    if rule_vol == '1m-5m':
        return 1000000 <= monthly_vol < 5000000
    if rule_vol == '>5m':
        return monthly_vol >= 5000000
    return False
```

---

### 9. Monthly Fraud Level
**Rule:** `null` matches all. Fraud level = (fraud_volume / total_volume) * 100.

**How to compute:**
```python
# Filter payments for the merchant and time period
merchant_payments = payments[
    (payments['merchant'] == merchant_name) &
    (payments['year'] == 2023) &
    (payments['day_of_year'] >= 1) &
    (payments['day_of_year'] <= 31)  # January
]

# Monthly volume = sum of all transaction amounts
monthly_vol = merchant_payments['eur_amount'].sum()

# Fraud volume = sum of amounts where has_fraudulent_dispute is True
fraud_vol = merchant_payments[
    merchant_payments['has_fraudulent_dispute'] == True
]['eur_amount'].sum()

# Fraud percentage
fraud_pct = (fraud_vol / monthly_vol) * 100
```

| Rule Value | Matches |
|------------|---------|
| `<7.2%` | fraud_pct < 7.2 |
| `7.2%-7.7%` | 7.2 <= fraud_pct < 7.7 |
| `7.7%-8.3%` | 7.7 <= fraud_pct < 8.3 |
| `>8.3%` | fraud_pct >= 8.3 |

```python
def matches_fraud_level(rule_fraud, fraud_pct):
    if rule_fraud is None:
        return True
    if rule_fraud == '<7.2%':
        return fraud_pct < 7.2
    if rule_fraud == '7.2%-7.7%':
        return 7.2 <= fraud_pct < 7.7
    if rule_fraud == '7.7%-8.3%':
        return 7.7 <= fraud_pct < 8.3
    if rule_fraud == '>8.3%':
        return fraud_pct >= 8.3
    return False
```

---

## Fee Calculation

```python
fee = fixed_amount + (rate * transaction_value / 10000)
```

## Rule Selection

When multiple rules match a transaction, the aggregation method depends on the question:
- **Minimize fees:** Select the rule with the **lowest** calculated fee
- **Maximize fees:** SUM all matching fees 

Always read the question carefully to determine the appropriate approach.
