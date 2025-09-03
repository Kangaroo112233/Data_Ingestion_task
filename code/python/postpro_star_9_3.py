Global: “Use only the first-page account summary or the payment coupon when totals/due amounts appear in multiple places. Ignore transaction tables unless they explicitly say ‘Total Payments & Credits’ or equivalent.”

total_credits:
“Return the total of payments/credits for the statement period as printed in the summary/coupon. If the summary shows separate lines such as ‘Payments’ and ‘Other credits’, add them and return a negative value with the dollar sign (e.g., ‘-$1,053.75’). Do not read line-item transactions.”

billing_recipient_address / vendor_address:
“Return in one line: street [unit/suite/apt/#], city, state, ZIP. Include unit/suite/apt if present. Normalize ‘P.O.’ to ‘PO Box’. Do not include names, taglines, or phone numbers. Prefer the coupon block.”

total_due:
“Return the full outstanding amount (‘Total Amount Due’, ‘Payment Due’) from the summary/coupon. Do not return ‘Minimum Payment Due’, ‘Amount Due Now’, or any partial payment.”

import re
from typing import Optional, Dict, Any

# ---------- address cleaner ----------

_PO_BOX_PAT = re.compile(r'(?i)\bP\.?\s*O\.?\s*Box\b')
_UNIT_PAT    = re.compile(r'(?i)\b(?:Apt|Apartment|Unit|Suite|Ste|#)\s*[\w-]+\b')

def _normalize_address_line(s: str) -> str:
    s = " ".join(s.split())  # collapse whitespace/newlines
    s = _PO_BOX_PAT.sub("PO Box", s)
    # collapse repeated spaces/commas
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip(" ,")

def _strip_leading_name(addr: str, *maybe_names: str) -> str:
    s = addr.strip()
    for name in maybe_names:
        if not name or name == "NULL":
            continue
        n = str(name).strip()
        if not n:
            continue
        # if address starts with the name, drop it
        if s.lower().startswith(n.lower()):
            s = s[len(n):].lstrip(" ,")
    return s

def _clean_address(raw: Optional[str], client_name: Optional[str], vendor_name: Optional[str]) -> Optional[str]:
    if raw is None or str(raw).strip().upper() == "NULL":
        return None
    s = _normalize_address_line(str(raw))

    # strip leading names (but keep unit/suite if the name line is separate)
    s = _strip_leading_name(s, client_name, vendor_name)

    # ensure we didn't lose unit/suite tokens (we only *preserve* them if present)
    # (No-op here; _UNIT_PAT is only used to ensure we don't strip such tokens elsewhere.)

    return s or None

# ---------- credits sign enforcement ----------

def _ensure_negative_money(m: Optional[str]) -> Optional[str]:
    """For credits: if normalized money lacks a '-', make it negative."""
    if m is None:
        return None
    s = str(m).strip()
    if not s or s.upper() == "NULL":
        return None
    # already negative? (handles "-$" and "($...)")
    if s.startswith("-") or (s.startswith("(") and s.endswith(")")):
        return s
    # add '-' before the $
    return s if not s.startswith("$") else f"-{s}"

# ---------- integrate into your post_process_fields ----------

def post_process_fields(result: Dict[str, Any]) -> Dict[str, str]:
    out = {k: ("NULL" if v is None else str(v).strip()) for k, v in (result or {}).items()}

    # guarantee schema (same as before) ...
    for k in ["client_name","account_number","total_charges","total_credits",
              "statement_end_date","due_date","billing_recipient_address",
              "vendor_name","vendor_address","total_due"]:
        out.setdefault(k, "NULL")

    # account last-4 (your existing _acct_last4_three_cases)
    if out["account_number"] != "NULL":
        last4 = _acct_last4_three_cases(out["account_number"])
        out["account_number"] = last4 if last4 is not None else "NULL"

    # normalize money (your _money_or_null) then apply credits sign rule
    for key in ("total_charges","total_credits","total_due"):
        if out[key] != "NULL":
            norm = _money_or_null(out[key])
            out[key] = norm if norm is not None else "NULL"

    # make credits negative if model forgot the sign (handles the trailing-minus/paren case via _money_or_null already)
    if out["total_credits"] not in (None, "NULL"):
        out["total_credits"] = _ensure_negative_money(out["total_credits"]) or "NULL"

    # dates (keep your _date_sane)
    for dkey in ("statement_end_date","due_date"):
        if out[dkey] != "NULL":
            sane = _date_sane(out[dkey])
            out[dkey] = sane if sane is not None else "NULL"

    # addresses
    out["billing_recipient_address"] = (
        _clean_address(out["billing_recipient_address"],
                       client_name=out.get("client_name"),
                       vendor_name=out.get("vendor_name"))
        or "NULL"
    )
    out["vendor_address"] = (
        _clean_address(out["vendor_address"],
                       client_name=out.get("client_name"),
                       vendor_name=out.get("vendor_name"))
        or "NULL"
    )

    # vendor_name normalization (keep your hook)
    if out["vendor_name"] != "NULL":
        out["vendor_name"] = _normalize_vendor_name(out["vendor_name"]) or "NULL"

    return out
