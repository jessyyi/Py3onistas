"""Dataset-backed, configurable interactive charts for JobScope."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.market_filters import label_counts

CHART_LABELS = {
    "skills": "Top Detected Skills",
    "salary": "Annual Salary Distribution",
    "functions": "Job Function Breakdown",
    "industry": "Industry Breakdown",
    "states": "Geographic Distribution",
    "experience": "Experience Level Distribution",
    "education": "Degree Mentions",
    "timeline": "Historical Posting Trend",
    "salary_by_function": "Median Salary by Job Function",
    "salary_disclosure": "Salary Data Availability",
}
DEFAULT_CHARTS = ["skills", "salary", "functions", "industry", "states", "experience"]


def available_charts(data: pd.DataFrame) -> dict[str, str]:
    """Only offer plots with source fields required for a meaningful result."""
    col = set(data.columns)
    enabled = {
        "skills": "skill_labels" in col,
        "salary": "salary_annual" in col,
        "functions": "job_function" in col,
        "industry": "industry" in col,
        "states": "state" in col,
        "experience": "experience_level" in col,
        "education": "degree" in col,
        "timeline": "listed_date" in col,
        "salary_by_function": {"salary_annual", "job_function"}.issubset(col),
        "salary_disclosure": "salary_annual" in col,
    }
    return {key: name for key, name in CHART_LABELS.items() if enabled[key]}


def valid_salary(data: pd.DataFrame) -> pd.Series:
    """Comparable USD full-time annual salaries (no invented observations)."""
    vals = pd.to_numeric(data.get("salary_annual", pd.Series(index=data.index, dtype=float)), errors="coerce")
    allowed = vals.between(10_000, 1_000_000)
    if "salary_currency" in data.columns:
        allowed &= data["salary_currency"].fillna("USD").astype(str).str.upper().eq("USD")
    if "work_type" in data.columns:
        allowed &= data["work_type"].eq("Full-time")
    return vals[allowed]


def chart_data(data: pd.DataFrame, key: str, top_n: int = 10) -> pd.DataFrame:
    """Aggregate once per chart. Supports tests without launching Streamlit."""
    if key in {"skills", "functions", "industry"}:
        field = {"skills": "skill_labels", "functions": "job_function", "industry": "industry"}[key]
        result = label_counts(data, field).head(top_n)
        return pd.DataFrame({"category": result.index.astype(str), "postings": result.values})
    if key in {"states", "experience", "education"}:
        field = {"states": "state", "experience": "experience_level", "education": "degree"}[key]
        if field not in data:
            return pd.DataFrame(columns=["category", "postings"])
        counts = data[field].fillna("Not specified").astype(str).value_counts().head(top_n)
        return pd.DataFrame({"category": counts.index, "postings": counts.values})
    if key == "salary":
        return pd.DataFrame({"annual_salary_usd": valid_salary(data).to_numpy()})
    if key == "salary_disclosure":
        n = len(valid_salary(data))
        return pd.DataFrame({"category": ["Comparable annual USD", "Unavailable / not comparable"],
                             "postings": [n, len(data)-n]})
    if key == "timeline":
        dates = pd.to_datetime(data["listed_date"], errors="coerce").dropna() if "listed_date" in data else pd.Series(dtype="datetime64[ns]")
        if dates.empty:
            return pd.DataFrame(columns=["month", "postings"])
        counts = dates.dt.to_period("M").astype(str).value_counts().sort_index()
        return pd.DataFrame({"month": counts.index, "postings": counts.values})
    if key == "salary_by_function":
        if "job_function" not in data:
            return pd.DataFrame(columns=["category", "median_salary_usd", "salary_observations"])
        salary = valid_salary(data)
        subset = data.loc[salary.index, ["job_function"]].copy()
        subset["annual_salary_usd"] = salary
        subset = subset.explode("job_function").dropna(subset=["job_function"])
        if subset.empty:
            return pd.DataFrame(columns=["category", "median_salary_usd", "salary_observations"])
        result = subset.groupby("job_function")["annual_salary_usd"].agg(["median", "size"])
        result = result[result["size"] >= 3].sort_values("size", ascending=False).head(top_n)
        return pd.DataFrame({"category": result.index.astype(str), "median_salary_usd": result["median"].values,
                             "salary_observations": result["size"].values})
    raise ValueError(f"Unknown chart: {key}")


def draw_chart(st, data: pd.DataFrame, key: str, accent: str, top_n: int = 10, style: str = "Auto") -> None:
    """Draw the selected chart with Plotly; never replace missing data with a claim."""
    rows = chart_data(data, key, top_n)
    title = CHART_LABELS[key]
    if rows.empty or (key in {"salary", "salary_by_function", "timeline"} and len(rows) == 0):
        st.info(f"No usable records for {title.lower()} in this selection.")
        return
    if key == "salary":
        fig = px.histogram(rows, x="annual_salary_usd", nbins=25, title=title,
                           labels={"annual_salary_usd": "Annual salary (USD)"})
        fig.update_traces(marker_color=accent)
    elif key == "timeline":
        fig = px.line(rows, x="month", y="postings", markers=True, title=title)
        fig.update_traces(line_color=accent)
    elif key == "salary_by_function":
        fig = px.bar(rows, x="median_salary_usd", y="category", orientation="h", title=title,
                     hover_data=["salary_observations"])
        fig.update_traces(marker_color=accent)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    elif key == "salary_disclosure":
        fig = px.pie(rows, names="category", values="postings", hole=0.55, title=title,
                     color_discrete_sequence=[accent, "#C9CFCD"])
    elif style == "Donut":
        fig = px.pie(rows, names="category", values="postings", hole=0.55, title=title)
    else:
        fig = px.bar(rows, x="postings", y="category", orientation="h", title=title)
        fig.update_traces(marker_color=accent)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_layout(height=345, margin=dict(l=5, r=8, t=48, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
