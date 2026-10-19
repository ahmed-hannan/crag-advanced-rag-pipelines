"""
Evaluation utilities: normalization + exact match scoring.
"""
import re
import json
from typing import List







def normalize_text(s):
    if s is None:
        return ""

    s = str(s).lower().strip()

    # remove punctuation
    s = re.sub(r"[^\w\s]", "", s)

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip()



def exact_match(pred, gold, alt):

    p = normalize_text(pred)
    g = normalize_text(gold)

    if p == g or p in g or g in p:
        return True

    for a in alt:
        a = normalize_text(a)
        if p == a or p in a or a in p:
            return True

    return False
