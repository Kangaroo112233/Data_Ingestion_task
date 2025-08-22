'client_name': 'The full legal name of the account holder. For BUSINESS credit card statements, if both an individual name and a business name appear (e.g., “John Doe” and “Doefarm LLC”), return ONLY the business name (e.g., “Doefarm LLC”). Prefer the registered legal entity (Inc., LLC, LLP, Ltd., Co., Corp., PC, PLLC) over any personal/authorized-user names. If no business name is present, return the individual’s full name. Exclude titles, roles, taglines, and any address or account fragments, and return the name exactly as printed.'
'total_charges': 'The statement-ending balance — i.e., the amount labeled “New Balance,” “New Balance Total,” or “Statement Balance” (as of the statement close date). DO NOT return “Total Purchases,” “Purchases,” “New Purchases,” “Total Activity,” “Fees Charged,” “Interest Charged,” or any transaction/category subtotal. If both “Current Balance” and “Statement/New Balance” appear, return the Statement/New Balance only. Prefer the figure shown on the first page summary or payment coupon near “Minimum Payment Due” and “Payment Due Date.” Return the value exactly as printed, including the dollar sign and any credit/negative notation (e.g., “-”, “CR”, or parentheses).'

# ==== POST-PROCESSORS =========================================================
import re, json
from typing import Optional, Dict, Any

# patterns: x...xNNNN  (no spaces/hyphens)  |  'ending in:NNNN'
_PAT_MASKED_LAST4 = re.compile(r'^(?:x|X)+(\d{4})$')
_PAT_ENDING_LAST4 = re.compile(r'(?i)^ending\s*in:\s*(\d{4})$')

def _acct_last4_three_cases(val: Optional[str]) -> Optional[str]:
    """Return last-4 ONLY for the 3 allowed formats; else None."""
    if val is None:
        return None
    s = str(val).strip()
    m = _PAT_MASKED_LAST4.match(s)
    if m:
        return m.group(1)
    m = _PAT_ENDING_LAST4.match(s)
    if m:
        return m.group(1)
    return None

def _normalize_vendor_name(val: Optional[str]) -> Optional[str]:
    """CapitalOne (any case/whitespace) -> 'Capital One'. Others unchanged."""
    if val is None:
        return None
    raw = str(val)
    if re.sub(r'\s+', '', raw).lower() == "capitalone":
        return "Capital One"
    return raw

def post_process_fields(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Field-level post processing applied to the parsed JSON dict
    returned by the model.
    - account_number: keep ONLY last 4 if format matches the 3 cases; else 'NULL'
    - vendor_name: map 'CapitalOne' -> 'Capital One'
    """
    out = {k: (None if v is None else str(v)) for k, v in result_dict.items()}

    if 'account_number' in out:
        last4 = _acct_last4_three_cases(out['account_number'])
        out['account_number'] = last4 if last4 is not None else "NULL"

    if 'vendor_name' in out:
        out['vendor_name'] = _normalize_vendor_name(out['vendor_name'])

    return out
# ==============================================================================
# before
result = response['response']
result = '{' + result.split('{', 1)[-1]
if len(result.rsplit('}', 1)) == 1:
    result += '}'
else:
    result = result.rsplit('}', 1)[0] + '}'
result = json.loads(result)

result = validate_fields(result, field_names)   # <--- this line exists now

#after

result = response['response']
result = '{' + result.split('{', 1)[-1]
if len(result.rsplit('}', 1)) == 1:
    result += '}'
else:
    result = result.rsplit('}', 1)[0] + '}'
result = json.loads(result)

# >>> ADD THIS <<<
result = post_process_fields(result)

result = validate_fields(result, field_names)


