import re
import logging
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DELIM = "::"
SKIP_MARKERS = {'MALFORMED_LOG', '""', '"', "'"}

DATETIME_FORMATS = [
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"
]
DATE_REGEX = re.compile(
    r"\b\d{2}[/-]\d{2}[/-]\d{4}(?: \d{2}:\d{2}:\d{2})?"
    r"|\b\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?"
)

TRANSACTION_KEYWORDS = [
    'withdrawal', 'deposit', 'purchase', 'refund',
    'debit', 'cashout', 'transfer', 'top-up'
]
TXN_PATTERN    = re.compile(r'\b(' + '|'.join(TRANSACTION_KEYWORDS) + r')\b',
                            re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"\d+\.\d+") 

LOCATION_STOPWORDS = {"of", "to", "in", "at", "atm", "user", "from"}
LOCATION_REGEX = re.compile(
    rf"""
    [$€£]?                    # optional currency before
    (?P<float>\d+\.\d+)       # mandatory float
    [$€£]?                    # optional currency after
    (?:\s*(?:[-\|\@:\#\.\/]+  # skip separators
      |(?:\b(?:{'|'.join(LOCATION_STOPWORDS)})\:?\b)
    ))*                       # repeat
    \s*
    (?P<location>[A-Za-z]{{3,}})  # capture 3+ letter location
    """,
    re.IGNORECASE | re.VERBOSE
)

DEVICE_STOPWORDS = ("device=", "device:", "dev:")
TIMESTAMP_REGEX  = DATE_REGEX


def parse_timestamp(ts_str: str) -> Optional[str]:
    """Try all known datetime formats, return ISO string or None."""
    for fmt in DATETIME_FORMATS:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    logger.debug(f"Unrecognized timestamp format: {ts_str}")
    return None


def extract_device_after_location(line: str, loc_m: re.Match) -> str:
    """Take everything after the location match and strip to the device model."""
    rest = line[loc_m.end():].lstrip(" -|@:/#>.")
    ts_m = TIMESTAMP_REGEX.match(rest)
    if ts_m:
        rest = rest[ts_m.end():].lstrip(" -|@:/#>.")
    low = rest.lower()
    for sw in DEVICE_STOPWORDS:
        if low.startswith(sw):
            rest = rest[len(sw):].lstrip()
            break
    m = re.match(r"^<([^>]+)>", rest)
    if m:
        return m.group(1)
    
    result = re.sub(r"[ \-|\@:/#\.>]+$", "", rest)
    return result if result else "No data"


def parse_raw_format(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse lines in the exact '::'-delimited raw format:
      timestamp::user::txn_type::amount::location::device
    Returns a dict or None if the split does not yield 6 parts.
    """
    parts = line.split(RAW_DELIM)
    if len(parts) != 6:
        return None
    ts, user, txn, amt, loc, dev = [p.strip() for p in parts]
    if loc == "None" or dev == "None":
        return None
    
    return {
        "timestamp": parse_timestamp(ts),
        "user_id": user.replace("usr:", "").replace("user:", ""),
        "txn_type": txn.lower(),
        "txn_amount": float(amt) if re.match(r"^\d+\.\d+$", amt) else None,
        "location": loc,
        "device": dev
    }

def parse_general_line(line: str) -> Dict[str, Any]:
    """
    Fallback for malformed or alternate formats:
    - Finds first date/timestamp
    - Extracts user, txn_type, amount, location, device
    """
    # Timestamp
    ts = None
    m_ts = DATE_REGEX.search(line)
    if m_ts:
        ts = parse_timestamp(m_ts.group())

    # User
    m_user = re.search(r"(?:usr:|user:)?\s*(user\d+)", line, re.IGNORECASE)
    user_id = m_user.group(1) if m_user else None

    # Transaction type
    m_txn = TXN_PATTERN.search(line)
    txn_type = m_txn.group(1).lower() if m_txn else None

    # Amount
    amt = None
    m_amt = AMOUNT_PATTERN.search(line)
    if m_amt:
        amt = float(m_amt.group())

    # Location
    # location = None
    m_loc = LOCATION_REGEX.search(line)
    if m_loc:
        location = m_loc.group("location")
    if location == "None":
        location = "No data"

    # Device
    device = extract_device_after_location(line, m_loc)# if m_loc else "No data"

    if device == "None":
        device = "No data"

    return {
        "timestamp": ts,
        "user_id": user_id,
        "txn_type": txn_type,
        "txn_amount": amt,
        "location": location,
        "device": device
    }

def parse_record(line: str) -> Dict[str, Any]:
    """
    Clean public API: parse any single raw log line into a structured dict.
    Tries the strict '::' parser first, then falls back to general parsing.
    """
    rec = parse_raw_format(line)
    if rec is None:
        rec = parse_general_line(line)
    return rec


def load_and_parse_logs(path: str) -> pd.DataFrame:
    """
    Read a file of raw logs, skip the first two header/malformed lines,
    parse each line via parse_record(), drop fully empty records,
    and return a DataFrame.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        # Skip the original two metadata lines:
        next(f, None)
        next(f, None)

        for raw_line in f:
            line = raw_line.strip()
            if not line or line in SKIP_MARKERS:
                continue

            rec = parse_record(line)
            if not any([rec.get("timestamp"), rec.get("user_id"), rec.get("txn_type")]):
                continue

            records.append(rec)

    df = pd.DataFrame(records)
    return df


def main():
    df = load_and_parse_logs("synthetic_dirty_transaction_logs.csv")
    df.to_csv("parsed_data.csv", index=False, encoding="utf-8", na_rep="No data")
    logger.info("Parsed data saved to parsed_data.csv")

if __name__ == "__main__":
    main()