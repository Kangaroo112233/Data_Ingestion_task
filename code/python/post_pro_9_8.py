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
