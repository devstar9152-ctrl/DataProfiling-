# pbl_generator.py
import pandas as pd
import re

def infer_length_rules(series):
    s = series.dropna().astype(str)
    if s.empty:
        return None
    lengths = s.map(len)
    min_l = int(lengths.min())
    max_l = int(lengths.max())
    typical = int(lengths.mode().iloc[0]) if not lengths.mode().empty else None
    return {'min':min_l, 'max':max_l, 'typical':typical}

def is_numeric_only(series, threshold=0.98):
    s = series.dropna().astype(str)
    if s.empty: return False
    frac = (s.str.fullmatch(r'\d+')).mean()
    return frac >= threshold

def is_alpha_only(series, threshold=0.98):
    s = series.dropna().astype(str)
    if s.empty: return False
    frac = (s.str.fullmatch(r'[A-Za-z ]+')).mean()
    return frac >= threshold

def generate_pbl_for_column(series, col_name=None):
    rules = []
    null_allowed = series.isna().sum() > 0
    if not null_allowed:
        rules.append("Nulls are NOT allowed.")
    else:
        rules.append("Nulls are allowed.")

    if is_numeric_only(series):
        rules.append("Only numeric characters allowed.")
        lengths = infer_length_rules(series)
        if lengths:
            rules.append(f"Character length between {lengths['min']} and {lengths['max']}.")
        rules.append("No special characters or alphabets allowed.")
    elif is_alpha_only(series):
        rules.append("Alphabetic characters allowed only.")
    else:
        s = series.dropna().astype(str)
        sample = s.head(50).tolist()
        if any(re.search(r'@', v) for v in sample):
            rules.append("Email format expected (contains '@').")
        if any(re.fullmatch(r'\+?\d[\d\-\s]{6,}', v) for v in sample):
            rules.append("Phone number like pattern detected.")
        lengths = infer_length_rules(series)
        if lengths:
            rules.append(f"Recommended character length between {lengths['min']} and {lengths['max']}.")
        rules.append("Avoid special characters unless required (/, -, : etc).")

    if series.nunique(dropna=True) == len(series.dropna()):
        rules.append("Values are unique — consider as candidate key.")
    else:
        distinct_ratio = series.nunique(dropna=True) / max(1, len(series))
        if distinct_ratio < 0.02:
            rules.append("Low cardinality — likely categorical; map to lookup table or enumerations.")

    return rules
# pbl_generator.py  (append this to the existing file)

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Any, Optional

def _suggest_regex_from_sample(sample_values: List[str], max_samples=100):
    """
    Very simple heuristic-based regex suggestion:
      - If all digits -> \d+
      - If emails -> simple email-ish regex
      - If date-like -> date-ish
      - else fallback: allow most printable chars and limit length.
    (This is a heuristic — fine tune for your data.)
    """
    sample = [str(x) for x in sample_values[:max_samples] if pd.notna(x)]
    if not sample:
        return None
    all_digits = all(re.fullmatch(r'\d+', s) for s in sample)
    if all_digits:
        return r'^\d+$'
    if all(re.search(r'@', s) for s in sample if s):
        return r'^[\w\.-]+@[\w\.-]+\.\w{2,}$'
    if all(re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}', s) for s in sample if s):
        return r'^\d{2,4}[-/]\d{1,2}[-/]\d{1,2}$'
    # fallback: length-limited printable
    lengths = [len(s) for s in sample if s is not None]
    if lengths:
        min_l = min(lengths)
        max_l = max(lengths)
        return rf'^.{{{min_l},{max_l}}}$'
    return None

def derive_rules_from_reference(target_series: pd.Series,
                                reference_series: pd.Series,
                                *,
                                enum_threshold: float = 0.95,
                                sample_size: int = 500) -> List[str]:
    """
    Compare target column values with reference column values and produce PBL rules.
    - enum_threshold: if unique values in reference cover >= threshold fraction of target non-null, suggest allowed-values list.
    Returns a list of readable PBL rules.
    """
    t = target_series.dropna().astype(str)
    r = reference_series.dropna().astype(str)

    rules = []

    # Null rule: if reference has no nulls but target does -> likely null NOT allowed
    if r.isna().sum() == 0:
        if t.isna().sum() == 0:
            rules.append("Nulls are NOT allowed (reference has no nulls).")
        else:
            rules.append("Nulls are present in this dataset but reference expects none.")

    else:
        rules.append("Nulls allowed (reference permits nulls).")

    # Uniqueness / key check
    if r.nunique(dropna=True) == len(r):
        rules.append("Reference values are unique — consider as reference key.")
    # If target values subset of reference values -> enumerated allowed values
    if len(r) > 0 and len(t) > 0:
        # compute overlap fraction: how many target non-null values exist in reference
        t_vals = set(t.unique())
        r_vals = set(r.unique())
        if len(t_vals) == 0:
            overlap_frac = 0.0
        else:
            overlap_frac = sum(1 for v in t_vals if v in r_vals) / max(1, len(t_vals))

        if overlap_frac >= enum_threshold:
            # prepare top allowed values (if small)
            distinct_ref = sorted(list(r_vals))[:200]
            if len(distinct_ref) <= 50:
                rules.append(f"Allowed values derived from reference (enumeration of {len(distinct_ref)} values).")
                rules.append("Allowed values (sample): " + ", ".join(map(str, distinct_ref[:20])))
            else:
                rules.append(f"Reference provides a large allowed list ({len(distinct_ref)} values). Use lookup mapping.")
            rules.append(f"Fraction of target values present in reference: {overlap_frac:.2f}")
        else:
            # if a non-trivial part of target matches reference, say so
            if overlap_frac > 0:
                rules.append(f"{overlap_frac:.2f} fraction of target values match reference values (partial overlap).")
            else:
                rules.append("No meaningful overlap with reference values; reference may be different domain or mismatch.")

    # Data type / numeric checks
    if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(t):
        tnum = pd.to_numeric(t, errors='coerce').dropna()
        rnum = pd.to_numeric(r, errors='coerce').dropna()
        if not tnum.empty:
            rules.append(f"Numeric range observed in target: min={tnum.min()}, max={tnum.max()}.")
        if not rnum.empty:
            rules.append(f"Reference numeric range: min={rnum.min()}, max={rnum.max()}.")

    # Length rules and regex suggestion from reference sample (preferred) else use target sample
    # Prefer using reference to suggest allowed pattern/length
    ref_sample = list(r.sample(min(len(r), sample_size)).astype(str)) if len(r) else []
    tgt_sample = list(t.sample(min(len(t), sample_size)).astype(str)) if len(t) else []
    pattern = _suggest_regex_from_sample(ref_sample or tgt_sample)
    if pattern:
        rules.append(f"Suggested pattern (regex): `{pattern}`")

    # Cardinality hint
    if len(r) > 0:
        rules.append(f"Reference distinct count: {r.nunique(dropna=True)}")
    if len(t) > 0:
        rules.append(f"Target distinct count: {t.nunique(dropna=True)}")

    # Uniqueness in target
    if t.nunique(dropna=True) == len(t):
        rules.append("Values in this dataset are unique — candidate key.")
    else:
        distinct_ratio = t.nunique(dropna=True) / max(1, len(t))
        if distinct_ratio < 0.02:
            rules.append("Low cardinality in target — likely categorical.")

    # Return rules as readable list
    return rules
