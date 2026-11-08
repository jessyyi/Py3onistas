"""Load, clean and audit the LinkedIn Job Postings (2023-2024) dataset.

Source: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

The main entry point is ``build_clean_jobs``. It returns one row per U.S.
posting with the shared columns other modules use (see SHARED_INTERFACES.md):
job_id, title, description, location, job_function, industry, plus salary,
company and location detail columns.

Rules used here were checked against the full dataset on Nov 5-8, 2026:
- postings.csv has 123,849 rows and every job_id is unique.
- A posting has 1 to 3 job-function tags, so tags are kept as a list.
- normalized_salary matches our own yearly conversion of USD salaries
  (all rows within 1%), so it is used as the yearly salary.
- Locations come in four U.S. formats ("City, ST", "United States",
  "State, United States", metro names). 76 rows match none of them and
  are dropped.
- 4,718 postings repeat an earlier one exactly (same company, title,
  description and location) and are dropped. Postings with the same text in
  different cities are kept and counted in ``same_text_postings``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# File layout of the Kaggle download (paths relative to the raw data folder)
# ---------------------------------------------------------------------------

RAW_FILES = {
    "postings": "postings.csv",
    "companies": "companies/companies.csv",
    "company_industries": "companies/company_industries.csv",
    "job_skills": "jobs/job_skills.csv",
    "job_industries": "jobs/job_industries.csv",
    "skills_map": "mappings/skills.csv",
    "industries_map": "mappings/industries.csv",
}

# Columns each table must have. check_schema() reports anything missing.
REQUIRED_COLUMNS = {
    "postings": [
        "job_id", "company_id", "company_name", "title", "description",
        "location", "min_salary", "med_salary", "max_salary", "pay_period",
        "currency", "normalized_salary", "remote_allowed",
        "formatted_experience_level", "formatted_work_type", "listed_time",
    ],
    "companies": ["company_id", "name"],
    "company_industries": ["company_id", "industry"],
    "job_skills": ["job_id", "skill_abr"],
    "job_industries": ["job_id", "industry_id"],
    "skills_map": ["skill_abr", "skill_name"],
    "industries_map": ["industry_id", "industry_name"],
}

# Order of columns in the cleaned table.
CLEAN_COLUMNS = [
    "job_id", "title", "description", "location", "job_function", "industry",
    "company_id", "company_name", "state", "location_type", "job_industry",
    "salary_min", "salary_max", "salary_med", "salary_period",
    "salary_currency", "salary_annual", "salary_disclosed",
    "remote_allowed", "experience_level", "work_type", "listed_date",
    "description_length", "same_text_postings",
]

# Columns that hold lists (several labels per posting).
LIST_COLUMNS = ["job_function", "industry", "job_industry"]

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

US_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Metro names such as "San Francisco Bay Area" or "Greater Boston".
# All 202 distinct metro names in the dataset were checked by hand: all U.S.
METRO_PATTERN = r"Metropolitan Area|Metroplex|Metro\b|Bay Area|^Greater |Area$"

KEPT_LOCATION_TYPES = ("city_state", "state_name", "metro_area", "country_only")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read every table listed in RAW_FILES from the Kaggle download folder.

    Each CSV is read once and cached as Parquet next to it, so later runs
    load in seconds. Delete the .parquet files to force a fresh read.
    """
    raw_dir = Path(raw_dir)
    tables = {}
    for name, rel_path in RAW_FILES.items():
        csv_path = raw_dir / rel_path
        parquet_path = csv_path.with_suffix(".parquet")
        if parquet_path.exists():
            tables[name] = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            tables[name] = pd.read_csv(csv_path)
            tables[name].to_parquet(parquet_path, index=False)
        else:
            raise FileNotFoundError(f"Missing raw file: {csv_path}")
    return tables


def check_schema(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Return the required columns that are missing, per table.

    An empty dict means every table has the columns this module expects.
    """
    missing = {}
    for name, columns in REQUIRED_COLUMNS.items():
        absent = [c for c in columns if c not in tables[name].columns]
        if absent:
            missing[name] = absent
    return missing


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def classify_location(location: pd.Series) -> pd.DataFrame:
    """Sort each location string into a type and find its U.S. state code.

    Types:
      city_state    "New York, NY"
      state_name    "California, United States" or "Los Angeles, California, United States"
      metro_area    "San Francisco Bay Area", "Greater Boston"
      country_only  "United States"
      missing       empty or NaN
      other         anything else (foreign places, "NAMER", vague regions)

    Returns a DataFrame with columns ``location_type`` and ``state``
    (two-letter code, or NaN when the place has no single state).
    """
    loc = location.fillna("").astype(str).str.strip()

    code = loc.str.extract(r",\s*([A-Z]{2})$")[0]
    code = code.where(code.isin(US_STATE_CODES))

    ends_with_us = loc.str.endswith("United States")
    before_us = loc.str.replace(r",\s*United States$", "", regex=True)
    last_part = before_us.str.split(",").str[-1].str.strip().str.lower()
    name_code = last_part.where(ends_with_us).map(US_STATE_NAMES)

    is_metro = loc.str.contains(METRO_PATTERN, regex=True)

    location_type = pd.Series("other", index=loc.index)
    location_type[is_metro] = "metro_area"
    location_type[name_code.notna()] = "state_name"
    location_type[code.notna()] = "city_state"
    location_type[loc.str.lower() == "united states"] = "country_only"
    location_type[loc == ""] = "missing"

    state = code.fillna(name_code)
    return pd.DataFrame({"location_type": location_type, "state": state})


# ---------------------------------------------------------------------------
# One-to-many tables -> one list per key
# ---------------------------------------------------------------------------

def labels_to_lists(link: pd.DataFrame, key: str, label: str) -> pd.Series:
    """Collapse a one-to-many table into one sorted, unique label list per key.

    Doing this before any merge keeps one row per posting. A plain merge
    would repeat a posting once for every label it has.
    """
    clean = link[[key, label]].dropna().drop_duplicates()
    return clean.groupby(key)[label].agg(sorted)


def build_job_functions(job_skills: pd.DataFrame, skills_map: pd.DataFrame) -> pd.Series:
    """Job-function names per job_id, e.g. ["Engineering", "Information Technology"].

    These are LinkedIn's 35 broad job-function tags. They are not individual
    technical skills.
    """
    named = job_skills.merge(skills_map, on="skill_abr", how="left")
    return labels_to_lists(named, "job_id", "skill_name")


def build_job_industries(job_industries: pd.DataFrame, industries_map: pd.DataFrame) -> pd.Series:
    """Posting-level industry names per job_id."""
    named = job_industries.merge(industries_map, on="industry_id", how="left")
    return labels_to_lists(named, "job_id", "industry_name")


def build_company_industries(company_industries: pd.DataFrame) -> pd.Series:
    """Company-level industry names per company_id (the proposal's 'industry')."""
    return labels_to_lists(company_industries, "company_id", "industry")


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

def add_salary_columns(postings: pd.DataFrame) -> pd.DataFrame:
    """Add salary columns. Missing salaries stay NaN and are never set to 0.

    salary_annual is the dataset's normalized_salary, kept only for USD
    postings (15 postings use other currencies). It was checked against our
    own conversion (hourly x 2080, weekly x 52, biweekly x 26, monthly x 12).
    """
    out = postings.copy()
    out["salary_min"] = out["min_salary"]
    out["salary_max"] = out["max_salary"]
    out["salary_med"] = out["med_salary"]
    out["salary_period"] = out["pay_period"]
    out["salary_currency"] = out["currency"]
    out["salary_annual"] = out["normalized_salary"].where(out["currency"] == "USD")
    out["salary_disclosed"] = out[["min_salary", "med_salary", "max_salary"]].notna().any(axis=1)
    return out


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def _text_key(postings: pd.DataFrame) -> pd.DataFrame:
    """Normalized company, title and description used to compare postings."""
    return pd.DataFrame({
        "company": postings["company_name"].fillna("").str.lower().str.strip(),
        "title": postings["title"].fillna("").str.lower().str.strip(),
        "description": postings["description"].fillna("").str.strip(),
    }, index=postings.index)


def find_content_duplicates(postings: pd.DataFrame, include_location: bool = True) -> pd.Series:
    """Flag repeat postings: same company, title and description as an earlier row.

    With ``include_location=True`` (the default) the location must match too.
    On the full data, 12,746 postings share text with an earlier one, but only
    4,718 of them are in the same place. The rest are the same job posted in
    several cities, which count as separate openings.

    The first copy is not flagged. Postings with an empty description are
    never flagged, because empty text says nothing about being the same job.
    """
    key = _text_key(postings)
    if include_location:
        key["location"] = postings["location"].fillna("").str.strip()
    has_text = key["description"] != ""
    return key.duplicated(keep="first") & has_text


def count_same_text(postings: pd.DataFrame) -> pd.Series:
    """Number of postings with the same company, title and description.

    1 means the text is unique. A value like 20 usually means one job posted
    in 20 cities. Matching can use this to avoid showing the same job many
    times in a Top-K list.
    """
    key = _text_key(postings)
    size = key.groupby(list(key.columns), sort=False)["company"].transform("size")
    return size.where(key["description"] != "", 1).astype(int)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_clean_jobs(
    tables: dict[str, pd.DataFrame],
    drop_content_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the cleaned, U.S.-only postings table.

    Returns ``(clean_jobs, log)``. ``log`` records row counts at each step
    so the audit can report what was removed and why.
    """
    log = {}
    postings = tables["postings"].copy()
    log["rows_raw"] = len(postings)
    log["duplicate_job_ids"] = int(postings["job_id"].duplicated().sum())
    postings = postings.drop_duplicates(subset="job_id", keep="first")

    # Company name: fill gaps from companies.csv when company_id is known.
    names = tables["companies"].drop_duplicates("company_id").set_index("company_id")["name"]
    postings["company_name"] = postings["company_name"].fillna(postings["company_id"].map(names))

    # Location and U.S. filter.
    location = classify_location(postings["location"])
    postings = postings.join(location)
    log["location_type_counts"] = postings["location_type"].value_counts().to_dict()
    is_us = postings["location_type"].isin(KEPT_LOCATION_TYPES)
    log["rows_dropped_not_us"] = int((~is_us).sum())
    postings = postings[is_us]

    # Duplicates: drop exact copies in the same place, keep multi-city postings.
    log["same_text_any_location"] = int(find_content_duplicates(postings, include_location=False).sum())
    is_dup = find_content_duplicates(postings, include_location=True)
    log["content_duplicates_same_location"] = int(is_dup.sum())
    if drop_content_duplicates:
        postings = postings[~is_dup]
    postings["same_text_postings"] = count_same_text(postings)
    log["rows_in_multi_city_groups"] = int((postings["same_text_postings"] > 1).sum())

    # Labels: collapse to lists first, then attach. Row count must not change.
    n_before = len(postings)
    job_function = build_job_functions(tables["job_skills"], tables["skills_map"])
    job_industry = build_job_industries(tables["job_industries"], tables["industries_map"])
    company_industry = build_company_industries(tables["company_industries"])

    postings["job_function"] = postings["job_id"].map(job_function)
    postings["job_industry"] = postings["job_id"].map(job_industry)
    postings["industry"] = postings["company_id"].map(company_industry)
    assert len(postings) == n_before, "label join changed the number of postings"

    log["label_rows_without_posting"] = int(
        (~tables["job_skills"]["job_id"].isin(tables["postings"]["job_id"])).sum()
    )
    for col in LIST_COLUMNS:
        postings[col] = postings[col].apply(lambda v: v if isinstance(v, list) else [])

    # Salary and other fields.
    postings = add_salary_columns(postings)
    postings["experience_level"] = postings["formatted_experience_level"]
    postings["work_type"] = postings["formatted_work_type"]
    postings["remote_allowed"] = postings["remote_allowed"].eq(1)
    postings["listed_date"] = pd.to_datetime(postings["listed_time"], unit="ms", errors="coerce").dt.date
    postings["description_length"] = postings["description"].fillna("").str.len()

    clean = postings[CLEAN_COLUMNS].reset_index(drop=True)
    log["rows_clean"] = len(clean)
    return clean, log


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_clean_jobs(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summary tables for the Nov 12 audit.

    A posting with several job functions is counted once under each one, and
    each share is computed against all postings (so shares can sum above 1).
    """
    n = len(clean)

    missing = (
        clean.drop(columns=LIST_COLUMNS).isna().mean()
        .rename("missing_share").to_frame()
    )
    for col in LIST_COLUMNS:
        missing.loc[col, "missing_share"] = clean[col].str.len().eq(0).mean()
    missing = missing.sort_values("missing_share", ascending=False).round(4)

    by_function = clean[["job_function", "salary_disclosed"]].explode("job_function")
    by_function["job_function"] = by_function["job_function"].fillna("(none)")
    function_coverage = (
        by_function.groupby("job_function")
        .agg(postings=("salary_disclosed", "size"), salary_disclosed=("salary_disclosed", "sum"))
        .assign(
            share_of_postings=lambda d: (d["postings"] / n).round(4),
            salary_disclosure_rate=lambda d: (d["salary_disclosed"] / d["postings"]).round(4),
        )
        .sort_values("postings", ascending=False)
    )

    lengths = clean["description_length"]
    description = pd.Series({
        "empty": int((lengths == 0).sum()),
        "under_200_chars": int(((lengths > 0) & (lengths < 200)).sum()),
        "median_chars": float(lengths.median()),
    }).to_frame("value")

    salary = pd.Series({
        "postings": n,
        "salary_disclosed": int(clean["salary_disclosed"].sum()),
        "salary_annual_usd": int(clean["salary_annual"].notna().sum()),
        "disclosure_rate": round(clean["salary_disclosed"].mean(), 4),
    }).to_frame("value")

    period = clean["salary_period"].value_counts(dropna=False).rename("postings").to_frame()

    return {
        "missing": missing,
        "function_coverage": function_coverage,
        "description": description,
        "salary": salary,
        "salary_period": period,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_clean_jobs(clean: pd.DataFrame, out_dir: str | Path, sample_size: int = 500, seed: int = 42) -> dict[str, Path]:
    """Write the cleaned table (Parquet), a CSV copy and a small random sample.

    Parquet keeps the list columns as lists. In the CSVs, lists are joined
    with "; ". Only the sample is small enough to share on GitHub.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    as_text = clean.copy()
    for col in LIST_COLUMNS:
        as_text[col] = as_text[col].apply("; ".join)

    paths = {
        "parquet": out_dir / "clean_jobs.parquet",
        "csv": out_dir / "clean_jobs.csv",
        "sample": out_dir / "sample_jobs.csv",
    }
    clean.to_parquet(paths["parquet"], index=False)
    as_text.to_csv(paths["csv"], index=False)
    as_text.sample(min(sample_size, len(as_text)), random_state=seed).to_csv(paths["sample"], index=False)
    return paths


def load_clean_jobs(path: str | Path = "data/processed/clean_jobs.parquet") -> pd.DataFrame:
    """Load the cleaned table. Other modules call this to get ``clean_jobs``."""
    return pd.read_parquet(path)


def run_pipeline(raw_dir: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load, check, clean, audit and save. Prints the time each step takes."""
    timings = {}

    start = time.perf_counter()
    tables = load_raw_tables(raw_dir)
    timings["load"] = time.perf_counter() - start

    missing = check_schema(tables)
    if missing:
        raise ValueError(f"Columns missing from raw tables: {missing}")

    start = time.perf_counter()
    clean, log = build_clean_jobs(tables)
    timings["clean"] = time.perf_counter() - start

    start = time.perf_counter()
    audit = audit_clean_jobs(clean)
    timings["audit"] = time.perf_counter() - start

    start = time.perf_counter()
    paths = save_clean_jobs(clean, out_dir)
    for name, table in audit.items():
        table.to_csv(Path(out_dir) / f"audit_{name}.csv")
    timings["save"] = time.perf_counter() - start

    log["timings_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    log["output_files"] = {k: str(v) for k, v in paths.items()}
    return clean, log, audit
