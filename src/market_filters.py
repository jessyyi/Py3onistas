"""Dataset-driven hierarchical filters for the JobScope market explorer."""

from __future__ import annotations

import ast
import re
from collections import Counter

import pandas as pd


STATE_SUBREGIONS = {
    "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
    "Southeast": {"DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN"},
    "Midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
    "South Central": {"AR", "LA", "OK", "TX"},
    "Mountain": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"},
    "Pacific": {"AK", "CA", "HI", "OR", "WA"},
}
REGION_SUBREGIONS = {
    "East": ("Northeast", "Southeast"),
    "Central": ("Midwest", "South Central"),
    "West": ("Mountain", "Pacific"),
}

ROLE_PATTERNS = {
    "Data & AI": r"data|analytics|scientist|machine learning|artificial intelligence|research|statistic|business intelligence|\bai\b|\bml\b",
    "Engineering & Technology": r"engineer|developer|software|information technology|\bit\b|cloud|network|security|architect|computer|quality assurance|\bqa\b",
    "Finance & Accounting": r"financ|account|bank|investment|credit|risk|audit|actuar|tax|treasury|quant|insurance",
    "Business & Operations": r"business|operations|project|product|management|administrat|human resources|\bhr\b|consult|strateg|procurement|supply chain",
    "Marketing & Sales": r"marketing|advertis|sales|brand|public relation|customer|retail|commerce|growth",
    "Healthcare & Life Sciences": r"health|medical|nurs|clinical|pharma|biolog|dental|therap|patient|science",
    "Education & Public Services": r"educat|teach|school|university|government|public|legal|law|social|nonprofit",
    "Design & Creative": r"design|creative|content|media|journal|writer|art|video",
}
INDUSTRY_PATTERNS = {
    "Technology": r"software|technology|internet|computer|information|semiconductor|telecommunications|\bit services\b|\bai\b",
    "Financial Services": r"financial|bank|insurance|investment|capital market|accounting|venture|private equity",
    "Healthcare": r"health|hospital|medical|pharma|biotech|wellness|life science",
    "Manufacturing & Energy": r"manufactur|industrial|automotive|aerospace|energy|oil|gas|utilities|electrical|machinery|chemic",
    "Retail & Consumer": r"retail|consumer|food|beverage|hospitality|restaurants|leisure|travel|apparel",
    "Business Services": r"consult|staffing|recruit|professional service|outsourc|legal|advertising|marketing|real estate",
    "Education & Public": r"educat|school|university|government|public administration|nonprofit|civic",
    "Transport & Construction": r"transport|logistic|shipping|warehous|construction|architecture|building",
    "Media & Entertainment": r"media|entertain|publishing|broadcast|gaming|sports|arts",
}


def labels(value) -> list[str]:
    """Convert list-like CSV/parquet cells to normalized nonempty labels."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if pd.notna(x) and str(x).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return labels(value.tolist())
    if not isinstance(value, str) or not value.strip():
        return []
    s = value.strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (tuple, list)):
                return labels(parsed)
        except (SyntaxError, ValueError):
            pass
    return [s]


def group_name(label: str, kind: str) -> str:
    """Assign each source label to one transparent, heuristic broad group."""
    patterns = ROLE_PATTERNS if kind == "role" else INDUSTRY_PATTERNS
    for group, pattern in patterns.items():
        if re.search(pattern, str(label), flags=re.IGNORECASE):
            return group
    return "Other / Unclassified"


def label_counts(data: pd.DataFrame, column: str) -> pd.Series:
    """Count unique postings carrying each label, most frequent first."""
    if column not in data:
        return pd.Series(dtype="int64")
    counter = Counter()
    for value in data[column]:
        counter.update(set(labels(value)))
    return pd.Series(dict(sorted(counter.items(), key=lambda x: (-x[1], x[0]))), dtype="int64")


def group_counts(data: pd.DataFrame, column: str, kind: str) -> pd.Series:
    """Count postings per broad group without duplicate counts per posting."""
    if column not in data:
        return pd.Series(dtype="int64")
    counter = Counter()
    for cell in data[column]:
        counter.update({group_name(x, kind) for x in labels(cell)})
    return pd.Series(dict(sorted(counter.items(), key=lambda x: (-x[1], x[0]))), dtype="int64")


def filter_labels(data: pd.DataFrame, column: str, *, group: str = "All", specific: str = "All", kind: str) -> pd.DataFrame:
    """Filter to postings containing the group and optionally a specific label."""
    if column not in data or (group == "All" and specific == "All"):
        return data
    def keep(cell):
        """Determine whether the job labels satisfy the selected filter criteria."""
        values = labels(cell)
        if specific != "All":
            return specific in values and (group == "All" or group_name(specific, kind) == group)
        return any(group_name(x, kind) == group for x in values)
    return data.loc[data[column].map(keep)]


def state_subregion(state: str) -> str:
    """Map a state abbreviation to its geographic subregion."""
    state = str(state).upper().strip()
    for subregion, states in STATE_SUBREGIONS.items():
        if state in states:
            return subregion
    return "Unknown / Unmapped"


def state_region(state: str) -> str:
    """Map a state abbreviation to its broader geographic region."""
    subregion = state_subregion(state)
    for region, subs in REGION_SUBREGIONS.items():
        if subregion in subs:
            return region
    return "Unknown / Unmapped"


def filter_location(data: pd.DataFrame, region: str = "All", subregion: str = "All", state: str = "All") -> pd.DataFrame:
    """Filter historical postings by the selected geographic criteria."""
    if "state" not in data:
        return data
    values = data["state"].fillna("").astype(str).str.upper().str.strip()
    mask = pd.Series(True, index=data.index)
    if region != "All":
        mask &= values.map(state_region).eq(region)
    if subregion != "All":
        mask &= values.map(state_subregion).eq(subregion)
    if state != "All":
        mask &= values.eq(state)
    return data.loc[mask]


def sorted_states(data: pd.DataFrame) -> list[str]:
    """Return state options ordered for the geographic filter."""
    if "state" not in data:
        return []
    series = data["state"].dropna().astype(str).str.upper().str.strip()
    return series[series.ne("")].value_counts().index.tolist()
