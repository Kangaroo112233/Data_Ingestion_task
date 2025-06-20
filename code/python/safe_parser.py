import ast

def safe_parse_list(s):
    """
    Try to parse a Python‐list literal in `s`.  If it fails because of a
    missing closing bracket, tack one on and retry.  Otherwise return [].
    """
    if not isinstance(s, str):
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except (SyntaxError, ValueError):
        # maybe the trailing ']' got dropped?
        if s.startswith('[') and not s.rstrip().endswith(']'):
            try:
                out = ast.literal_eval(s + ']')
                return out if isinstance(out, list) else []
            except Exception:
                pass
        # fallback: wrap the entire string as a single‐item list
        return [s]
df['box14Other_parsed'] = df['box14Other'].apply(safe_parse_list)
