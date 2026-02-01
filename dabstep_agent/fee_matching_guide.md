# Transaction Fee Calculation Guide

Each payment transaction incurs fees based on rules in `fees.json`. To calculate a transaction's fee:

1. **Find matching fee rules** - check transaction fields, merchant attributes, and monthly metrics against each rule's criteria
2. **Calculate fee for each match** - `fixed_amount + (rate * transaction_value / 10000)`
3. **Aggregate** - select min, max, or sum depending on the question

## Data Required

| Source | Fields Used |
|--------|-------------|
| `payments.csv` (transaction) | card_scheme, is_credit, aci, eur_amount, issuing_country, acquirer_country, has_fraudulent_dispute |
| `merchant_data.json` (merchant) | account_type, merchant_category_code, capture_delay |
| `payments.csv` (aggregated) | monthly_volume (sum of eur_amount), fraud_pct (fraud_volume / total_volume * 100) |
| `fees.json` | fee rules with matching criteria and fee parameters (fixed_amount, rate) |

---

## Matching Rules

Use a `rule_matches()` function that returns `True` if a fee rule matches the transaction:

```python
def rule_matches(fee, card_scheme, txn, merchant, monthly_volume, fraud_pct):
    """Return True if fee rule matches the transaction criteria."""

    # 1. Card Scheme - must match exactly
    if fee['card_scheme'] != card_scheme:
        return False

    # 2. Account Type - empty list matches all
    if fee['account_type'] and merchant['account_type'] not in fee['account_type']:
        return False

    # 3. MCC - empty list matches all
    if fee['merchant_category_code'] and merchant['merchant_category_code'] not in fee['merchant_category_code']:
        return False

    # 4. Is Credit - null/nan matches all
    if not pd.isna(fee['is_credit']) and bool(fee['is_credit']) != bool(txn['is_credit']):
        return False

    # 5. ACI - empty list matches all
    if fee['aci'] and txn['aci'] not in fee['aci']:
        return False

    # 6. Intracountry - null/nan matches all (1.0 = domestic, 0.0 = international)
    intracountry = 1.0 if txn['issuing_country'] == txn['acquirer_country'] else 0.0
    if not pd.isna(fee['intracountry']) and fee['intracountry'] != intracountry:
        return False

    # 7. Capture Delay - null matches all
    if not matches_capture_delay(fee.get('capture_delay'), merchant['capture_delay']):
        return False

    # 8. Monthly Volume - null matches all
    if not matches_monthly_volume(fee.get('monthly_volume'), monthly_volume):
        return False

    # 9. Monthly Fraud Level - null matches all
    if not matches_fraud_level(fee.get('monthly_fraud_level'), fraud_pct):
        return False

    return True
```

Then find all matching rules with:

```python
matching_rules = [fee for fee in fees if rule_matches(fee, card_scheme, txn, merchant, monthly_volume, fraud_pct)]
```

---

## ACI (Authorization Characteristics Indicator) Reference

| ACI | Description |
|-----|-------------|
| A | Card present - Non-authenticated |
| B | Card Present - Authenticated |
| C | Tokenized card with mobile device |
| D | Card Not Present - Card On File |
| E | Card Not Present - Recurring Bill Payment |
| F | Card Not Present - 3-D Secure |
| G | Card Not Present - Non-3-D Secure |

---

## Helper Functions

### matches_capture_delay

| Rule Value | Matches |
|------------|---------|
| `null` | all |
| `immediate` | merchant_delay == 0 |
| `<3` | merchant_delay < 3 |
| `3-5` | 3 <= merchant_delay <= 5 |
| `>5` | merchant_delay > 5 |
| `manual` | merchant_delay == 'manual' |

```python
def matches_capture_delay(rule_delay, merchant_delay):
    if rule_delay is None:
        return True
    if rule_delay == 'immediate':
        return merchant_delay == 0
    if rule_delay == 'manual':
        return merchant_delay == 'manual'
    if rule_delay == '<3':
        return merchant_delay < 3
    if rule_delay == '3-5':
        return 3 <= merchant_delay <= 5
    if rule_delay == '>5':
        return merchant_delay > 5
    return False
```

### matches_monthly_volume

| Rule Value | Matches |
|------------|---------|
| `null` | all |
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

### matches_fraud_level

Fraud level = (fraud_volume / total_volume) * 100

| Rule Value | Matches |
|------------|---------|
| `null` | all |
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

## Computing Monthly Metrics

```python
# Filter payments for the merchant and time period
merchant_payments = payments[
    (payments['merchant'] == merchant_name) &
    (payments['year'] == 2023) &
    (payments['day_of_year'] >= start_day) &
    (payments['day_of_year'] <= end_day)
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
