def money_or_null(raw: Optional[str]) -> Optional[str]:
    """
    Normalize money strings to a consistent format.
    Examples:
      "1053.75"       -> "$1,053.75"
      "($1,053.75)"   -> "-$1,053.75"
      "$1053.7-"      -> "-$1,053.70"
    Returns None if input cannot be parsed.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NULL":
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    if s.endswith("-"):
        neg = True
        s = s[:-1]
    if s.startswith("-"):
        neg = True
        s = s[1:]

    # strip everything except digits and dot
    import re
    s_clean = re.sub(r"[^\d.]", "", s)
    if not s_clean:
        return None

    try:
        amt = float(s_clean)
    except ValueError:
        return None

    out = f"${amt:,.2f}"
    if neg and amt != 0.0:
        out = f"-{out}"
    return out



from datetime import datetime, timedelta
from typing import Optional

def _date_sane(raw: Optional[str]) -> Optional[str]:
    """
    Return the date string if it looks like a plausible date,
    otherwise return None. Keeps the original format.
    Accepts mm/dd/yyyy, mm/dd/yy, 'Sep 1, 2025', 'September 1, 2025', or ISO '2025-09-01'.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NULL":
        return None

    fmts = ["%m/%d/%Y", "%m/%d/%y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]
    today = datetime.today()
    horizon = today + timedelta(days=370)

    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if datetime(1900,1,1) <= dt <= horizon:
                return s  # keep as-is
        except Exception:
            continue

    # if no parse works, just return None
    return None

def _ensure_negative_money(m: Optional[str]) -> Optional[str]:
    """
    For credits: ensure negative sign, except when value is exactly 0.
    Example:
      "$123.45"  -> "-$123.45"
      "($123.45)" -> "-$123.45"
      "-$123.45" -> "-$123.45"
      "$0.00"    -> "$0.00"   (no negative sign)
      "-$0.00"   -> "$0.00"
    """
    if m is None:
        return None
    s = str(m).strip()
    if not s or s.upper() == "NULL":
        return None

    # Already negative?
    if s.startswith("-") or (s.startswith("(") and s.endswith(")")):
        # Check if it's zero
        try:
            amt = float(s.replace("$","").replace(",","").replace("(","").replace(")","").replace("-","").strip())
        except ValueError:
            return s
        if amt == 0.0:
            return "$0.00"
        return s

    # Not negative yet
    try:
        amt = float(s.replace("$","").replace(",","").strip())
    except ValueError:
        return s

    if amt == 0.0:
        return "$0.00"
    return f"-{s}" if s.startswith("$") else f"-${amt:,.2f}"


import pandas as pd

# read your file (swap to read_csv if it's a CSV)
df = pd.read_excel("predictions.xlsx")   # or: pd.read_csv("predictions.csv")

# one row per file; columns are field names; values are ground_truth
wide = (df[["fn", "field", "ground_truth"]]
          .pivot_table(index="fn", columns="field", values="ground_truth", aggfunc="last")
          .reset_index())

wide.columns.name = None
wide.to_csv("ground_truth_by_file.csv", index=False)   # optional: save
