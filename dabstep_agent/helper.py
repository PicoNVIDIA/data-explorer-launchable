"""
Common helper functions for fee calculation tasks.
"""

import json
import pandas as pd
from typing import Optional, List, Tuple, Any, Dict

# Data directory
DATA_DIR = '/raid/jiwei/gitlab/llmtech/data-explorer-agent/data/context'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fees() -> List[Dict]:
    """Load fee rules from fees.json"""
    with open(f'{DATA_DIR}/fees.json', 'r') as f:
        return json.load(f)


def load_payments() -> pd.DataFrame:
    """Load payments data from payments.csv"""
    return pd.read_csv(f'{DATA_DIR}/payments.csv')


def load_merchants() -> List[Dict]:
    """Load merchant data from merchant_data.json"""
    with open(f'{DATA_DIR}/merchant_data.json', 'r') as f:
        return json.load(f)


def load_acquirer_countries() -> pd.DataFrame:
    """Load acquirer countries from acquirer_countries.csv"""
    return pd.read_csv(f'{DATA_DIR}/acquirer_countries.csv')


def get_merchant_info(merchant_name: str) -> Optional[Dict]:
    """Get merchant info by name"""
    merchants = load_merchants()
    for m in merchants:
        if m['merchant'] == merchant_name:
            return m
    return None


# =============================================================================
# FIELD MATCHING - List fields (null/empty = applies to all)
# =============================================================================

def matches_list_field(field_value: Any, target: Any) -> bool:
    """
    Check if a list field matches the target value.
    According to manual.md: null or empty list means applies to ALL values.

    Args:
        field_value: The field value from the fee rule (can be None, list, or other)
        target: The target value to match

    Returns:
        True if the field applies to the target value
    """
    if field_value is None:
        return True  # null means applies to all
    if isinstance(field_value, list):
        if len(field_value) == 0:
            return True  # empty list means applies to all
        return target in field_value
    return False


def matches_bool_field(field_value: Any, target: bool) -> bool:
    """
    Check if a boolean field matches the target value.
    Null means applies to all.

    Args:
        field_value: The field value from the fee rule (can be None or bool/numeric)
        target: The target boolean value

    Returns:
        True if the field applies to the target value
    """
    if field_value is None:
        return True  # null means applies to all
    return bool(field_value) == target


# =============================================================================
# RANGE PARSING
# =============================================================================

def parse_volume_value(val_str: str) -> float:
    """Convert volume string like '100k' or '1m' to number"""
    val_str = val_str.strip().lower()
    if val_str.endswith('m'):
        return float(val_str[:-1]) * 1_000_000
    elif val_str.endswith('k'):
        return float(val_str[:-1]) * 1_000
    else:
        return float(val_str)


def parse_volume_range(volume_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Parse volume range string like '100k-1m', '<100k', '>10m'

    Returns:
        Tuple of (min, max) or None if input is None
    """
    if volume_str is None:
        return None

    volume_str = volume_str.strip()

    if volume_str.startswith('<'):
        upper = parse_volume_value(volume_str[1:])
        return (0, upper)
    elif volume_str.startswith('>'):
        lower = parse_volume_value(volume_str[1:])
        return (lower, float('inf'))
    elif '-' in volume_str:
        parts = volume_str.split('-')
        lower = parse_volume_value(parts[0])
        upper = parse_volume_value(parts[1])
        return (lower, upper)
    else:
        val = parse_volume_value(volume_str)
        return (val, val)


def parse_fraud_value(val_str: str) -> float:
    """Convert fraud percentage string to float"""
    return float(val_str.strip().rstrip('%'))


def parse_fraud_range(fraud_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Parse fraud range string like '7.7%-8.3%', '<7.7%', '>8.3%'

    Returns:
        Tuple of (min, max) as percentages or None if input is None
    """
    if fraud_str is None:
        return None

    fraud_str = fraud_str.strip()

    if fraud_str.startswith('<'):
        upper = parse_fraud_value(fraud_str[1:])
        return (0, upper)
    elif fraud_str.startswith('>'):
        lower = parse_fraud_value(fraud_str[1:])
        return (lower, float('inf'))
    elif '-' in fraud_str:
        parts = fraud_str.split('-')
        lower = parse_fraud_value(parts[0])
        upper = parse_fraud_value(parts[1])
        return (lower, upper)
    else:
        val = parse_fraud_value(fraud_str)
        return (val, val)


# =============================================================================
# CAPTURE DELAY MATCHING
# =============================================================================

def matches_capture_delay(fee_delay: Optional[str], merchant_delay: Any) -> bool:
    """
    Check if merchant's capture delay matches fee rule's capture delay.

    Args:
        fee_delay: The capture_delay field from fee rule
        merchant_delay: The merchant's capture_delay (can be int, str like 'immediate', 'manual')

    Returns:
        True if the fee rule applies to the merchant's capture delay
    """
    if fee_delay is None:
        return True  # null means applies to all

    # Try to convert merchant delay to integer
    try:
        delay_days = int(merchant_delay)
    except (ValueError, TypeError):
        # Handle non-numeric delays (immediate, manual)
        return fee_delay == merchant_delay

    if fee_delay == '<3':
        return delay_days < 3
    elif fee_delay == '3-5':
        return 3 <= delay_days <= 5
    elif fee_delay == '>5':
        return delay_days > 5
    elif fee_delay == 'immediate':
        return delay_days == 0 or merchant_delay == 'immediate'
    elif fee_delay == 'manual':
        return merchant_delay == 'manual'
    else:
        # Try exact match
        try:
            return delay_days == int(fee_delay)
        except (ValueError, TypeError):
            return False


def matches_monthly_volume(rule_vol: Optional[str], monthly_vol: float) -> bool:
    """Check if monthly volume matches rule (simplified version)"""
    if rule_vol is None:
        return True
    volume_range = parse_volume_range(rule_vol)
    if volume_range is None:
        return True
    return volume_range[0] <= monthly_vol <= volume_range[1]


def matches_fraud_level(rule_fraud: Optional[str], fraud_pct: float) -> bool:
    """Check if fraud level matches rule (simplified version)"""
    if rule_fraud is None:
        return True
    fraud_range = parse_fraud_range(rule_fraud)
    if fraud_range is None:
        return True
    return fraud_range[0] <= fraud_pct <= fraud_range[1]


# =============================================================================
# FEE CALCULATION
# =============================================================================

def calculate_fee(fixed_amount: float, rate: float, transaction_value: float) -> float:
    """
    Calculate fee according to manual.md:
    fee = fixed_amount + rate * transaction_value / 10000
    """
    return fixed_amount + (rate * transaction_value / 10000)


# =============================================================================
# TRANSACTION FILTERING
# =============================================================================

def get_month_day_range(month: int) -> Tuple[int, int]:
    """
    Get day_of_year range for a given month (non-leap year).

    Args:
        month: Month number (1-12)

    Returns:
        Tuple of (start_day, end_day) for day_of_year
    """
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    start_day = sum(month_days[:month-1]) + 1
    end_day = sum(month_days[:month])
    return (start_day, end_day)


def filter_merchant_transactions(df: pd.DataFrame, merchant: str,
                                  year: int, month: int) -> pd.DataFrame:
    """
    Filter transactions for a specific merchant, year, and month.

    Args:
        df: Payments DataFrame
        merchant: Merchant name
        year: Year
        month: Month (1-12)

    Returns:
        Filtered DataFrame
    """
    start_day, end_day = get_month_day_range(month)
    return df[(df['merchant'] == merchant) &
              (df['year'] == year) &
              (df['day_of_year'] >= start_day) &
              (df['day_of_year'] <= end_day)]


def calculate_monthly_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate monthly volume and fraud metrics.

    Args:
        df: Filtered transactions DataFrame

    Returns:
        Dict with 'volume', 'fraud_volume', 'fraud_rate_pct'
    """
    volume = df['eur_amount'].sum()
    fraud_volume = df[df['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    fraud_rate_pct = (fraud_volume / volume * 100) if volume > 0 else 0

    return {
        'volume': volume,
        'fraud_volume': fraud_volume,
        'fraud_rate_pct': fraud_rate_pct
    }


def add_intracountry_flag(df: pd.DataFrame, acquirer_country: Optional[str] = None) -> pd.DataFrame:
    """
    Add intracountry flag to transactions DataFrame.

    Args:
        df: Transactions DataFrame
        acquirer_country: If provided, use this as acquirer country for all transactions.
                          Otherwise, use 'acquirer_country' column in df.

    Returns:
        DataFrame with 'intracountry' column added
    """
    df = df.copy()
    if acquirer_country is not None:
        df['intracountry'] = df['issuing_country'] == acquirer_country
    else:
        df['intracountry'] = df['issuing_country'] == df['acquirer_country']
    return df


# =============================================================================
# FEE RULE MATCHING
# =============================================================================

def matches_fee_rule(fee: Dict, card_scheme: str, account_type: str, mcc: int,
                     is_credit: bool, aci: str, intracountry: bool,
                     capture_delay: Any, monthly_vol: float, fraud_pct: float) -> bool:
    """
    Check if a fee rule matches all given criteria.

    Args:
        fee: Fee rule dictionary
        card_scheme: Transaction card scheme
        account_type: Merchant account type
        mcc: Merchant category code
        is_credit: Whether transaction is credit
        aci: Authorization Characteristics Indicator
        intracountry: Whether transaction is domestic
        capture_delay: Merchant capture delay
        monthly_vol: Monthly volume in EUR
        fraud_pct: Monthly fraud rate as percentage

    Returns:
        True if the fee rule matches all criteria
    """
    # Card scheme - must match exactly
    if fee['card_scheme'] != card_scheme:
        return False

    # Account type - list field
    if not matches_list_field(fee.get('account_type'), account_type):
        return False

    # MCC - list field
    if not matches_list_field(fee.get('merchant_category_code'), mcc):
        return False

    # is_credit - boolean field
    if not matches_bool_field(fee.get('is_credit'), is_credit):
        return False

    # ACI - list field
    if not matches_list_field(fee.get('aci'), aci):
        return False

    # Intracountry - boolean field
    if not matches_bool_field(fee.get('intracountry'), intracountry):
        return False

    # Capture delay
    if not matches_capture_delay(fee.get('capture_delay'), capture_delay):
        return False

    # Monthly volume
    if not matches_monthly_volume(fee.get('monthly_volume'), monthly_vol):
        return False

    # Monthly fraud level
    if not matches_fraud_level(fee.get('monthly_fraud_level'), fraud_pct):
        return False

    return True


def find_matching_fees(fees: List[Dict], card_scheme: str, account_type: str, mcc: int,
                       is_credit: bool, aci: str, intracountry: bool,
                       capture_delay: Any, monthly_vol: float, fraud_pct: float) -> List[Dict]:
    """
    Find all fee rules matching the given criteria.

    Returns:
        List of matching fee rules
    """
    return [fee for fee in fees if matches_fee_rule(
        fee, card_scheme, account_type, mcc, is_credit, aci,
        intracountry, capture_delay, monthly_vol, fraud_pct
    )]
