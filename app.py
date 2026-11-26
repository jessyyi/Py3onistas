"""JobScope Streamlit application: historical market and explainable job matching."""

from __future__ import annotations

import ast
import os
import hashlib
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from pypdf import PdfReader

from src.feature_extraction import extract_skills
from src.resume_profile import extract_profile_evidence
from src.resume_studio import build_resume_docx, TEMPLATES
from src.market_filters import (REGION_SUBREGIONS, filter_labels, filter_location, group_counts, group_name, label_counts, labels, sorted_states, state_region, state_subregion)
from src.dashboard_charts import DEFAULT_CHARTS, available_charts, chart_data, draw_chart, valid_salary
from src.matching import TextMatcher
from src.recommendation import rank_job_categories, score_jobs

from src.industry_groups import LOOKUP

from src.app_data import (
    full_market_available,
    load_market_data,
)

ROOT = Path(__file__).resolve().parent
THEMES = {
    "Forest Atelier": ("#32645b", "#f4f6f2", "#203e39"),
    "Midnight Ink": ("#334d70", "#f2f5f9", "#26364b"),
    "Bordeaux & Linen": ("#875365", "#faf5f3", "#563944"),
    "Graphite & Gold": ("#87683c", "#f7f6f1", "#383a3c"),
}


def to_labels(value) -> list[str]:
    """Convert list-like sample CSV values and parquet arrays to labels."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return to_labels(value.tolist())
    if not isinstance(value, str) or not value.strip():
        return []
    if value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return to_labels(parsed)
        except (SyntaxError, ValueError):
            pass
    return [value.strip()]


@st.cache_data(show_spinner="Loading historical postings...")
def load_jobs() -> tuple[pd.DataFrame, str]:
    """Prefer locally prepared data, falling back to the committed sample."""
    # Demo mode avoids extracting skills from the entire historical dataset
    # during the first browser page load. Set JOBSCOPE_DATA_MODE=full to opt in.
    sample = ROOT / "data/sample/sample_jobs.csv"
    full = ROOT / "data/processed/clean_jobs.parquet"
    full_mode = os.getenv("JOBSCOPE_DATA_MODE", "sample").lower() == "full"
    sources = [full, sample] if full_mode else [sample, full]
    for path in sources:
        if not path.exists():
            continue
        jobs = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if not {"job_id", "title", "description"}.issubset(jobs.columns):
            continue
        jobs = jobs.dropna(subset=["job_id"]).drop_duplicates("job_id").reset_index(drop=True)
        for field in ("job_function", "industry", "job_industry", "skill_labels"):
            if field in jobs.columns:
                jobs[field] = jobs[field].map(to_labels)
        if "skill_labels" not in jobs.columns:
            if len(jobs) > 5000:
                # Do not run expensive regex extraction for every full-data JD
                # as part of a Streamlit page load. Precompute this offline.
                jobs["skill_labels"] = [[] for _ in range(len(jobs))]
                st.warning("Skill labels are not precomputed in the full dataset. "
                           "Skill charts are unavailable until preprocessing; "
                           "Hybrid matching falls back to TF-IDF for those rows.")
            else:
                jobs["skill_labels"] = jobs["description"].fillna("").map(extract_skills)
        return jobs, ("Full local cleaned dataset" if path.suffix == ".parquet" else "Demonstration sample")
    raise FileNotFoundError("No suitable job data found in data/processed or data/sample.")


@st.cache_resource(show_spinner="Preparing matching model...")
def get_matcher(_jobs: pd.DataFrame, source: str) -> TextMatcher:
    """Reuse the same fitted DataFrame for all candidate scoring operations."""
    return TextMatcher().fit(_jobs)


def theme_style(name: str) -> tuple[str, str]:
    """Apply the selected visual theme and return its accent and text colors."""
    accent, pale, dark = THEMES[name]
    st.markdown(
        f"""<style>
        .stApp {{background: {pale}; color: {dark};}}
        div[data-testid="stMetric"] {{background: white; border:1px solid #e8e8e8;
            padding: 16px; border-radius: 14px;}}
        div[data-testid="stVerticalBlockBorderWrapper"] {{border-radius: 15px;}}
        .stButton button[kind="primary"] {{background-color: {accent}; border-color: {accent};}}
        .hero {{padding: 15px 0 7px 0;}}
        .muted-note {{font-size: 0.85rem; opacity: 0.8;}}
        [data-testid="stHeader"] {{border-bottom: 1px solid rgba(110,110,110,.10);}}
        [data-testid="stAppViewContainer"] {{background: {pale};}}
        .stApp h1, .stApp h2, .stApp h3 {{letter-spacing: -0.025em;}}
        div[data-testid="stMetric"] {{box-shadow: 0 3px 16px rgba(0,0,0,.025);}}
        [data-testid="stTopNav"] {{font-weight: 550;}}
        </style>""",
        unsafe_allow_html=True,
    )
    return accent, dark


def label_options(jobs: pd.DataFrame, field: str) -> list[str]:
    """Return sorted unique labels for the requested job attribute."""
    if field not in jobs.columns:
        return []
    if field in {"job_function", "industry", "job_industry"}:
        return sorted({v for group in jobs[field] for v in to_labels(group)})
    return sorted(str(v) for v in jobs[field].dropna().unique() if str(v).strip())


def subset(jobs: pd.DataFrame, column: str, choice: str) -> pd.DataFrame:
    """Filter job postings by the selected attribute, preserving all rows for All."""
    if choice == "All" or column not in jobs.columns:
        return jobs
    if column in {"job_function", "industry", "job_industry"}:
        return jobs[jobs[column].map(lambda x: choice in to_labels(x))]
    return jobs[jobs[column].astype(str) == choice]


def counts(jobs: pd.DataFrame, column: str, top: int = 8) -> pd.Series:
    """Count the most frequent labels in a job attribute column."""
    if column not in jobs.columns:
        return pd.Series(dtype=int)
    c = Counter(label for group in jobs[column] for label in to_labels(group))
    return pd.Series(dict(c.most_common(top)), dtype=int)


def card_plot(series: pd.Series, title: str, accent: str) -> None:
    """Render a horizontal bar chart for the supplied category counts."""
    if series.empty:
        st.info(f"No available data for {title.lower()}.")
        return
    chart = px.bar(x=series.values, y=series.index, orientation="h", title=title)
    chart.update_traces(marker_color=accent)
    chart.update_layout(yaxis={"categoryorder": "total ascending"},
                        margin=dict(l=5, r=8, t=48, b=10), height=320)
    st.plotly_chart(chart, use_container_width=True)


def _frequency_options(series: pd.Series) -> list[str]:
    """Frequency-ranked choices with no forced narrow default."""
    return ["All"] + [str(x) for x in series.index.tolist()]


def _valid_salary(data: pd.DataFrame) -> pd.Series:
    """Return plausible disclosed annual USD values only."""
    values = pd.to_numeric(data.get("salary_annual", pd.Series(index=data.index, dtype=float)), errors="coerce")
    mask = values.between(10_000, 1_000_000)
    if "salary_currency" in data:
        mask &= data["salary_currency"].fillna("USD").astype(str).str.upper().eq("USD")
    if "work_type" in data:
        mask &= data["work_type"].eq("Full-time")
    return values[mask]


def _chart_bar(series: pd.Series, title: str, accent: str, *, horizontal: bool = True, height: int = 340) -> None:
    """Render a configurable horizontal or vertical categorical bar chart."""
    if series.empty:
        st.info(f"No available data for {title.lower()}.")
        return
    if horizontal:
        fig = px.bar(x=series.values, y=series.index.astype(str), orientation="h", title=title)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Historical postings", yaxis_title="")
    else:
        fig = px.bar(x=series.index.astype(str), y=series.values, title=title)
        fig.update_layout(xaxis_title="", yaxis_title="Historical postings")
    fig.update_traces(marker_color=accent, hovertemplate="%{y}: %{x}<extra></extra>" if horizontal else None)
    fig.update_layout(margin=dict(l=8, r=8, t=50, b=18), height=height)
    st.plotly_chart(fig, use_container_width=True)


def _ranked_specific_options(data: pd.DataFrame, column: str, group: str, kind: str) -> list[str]:
    """Show the most frequent original labels within a broad group."""
    choices = label_counts(data, column)
    if group != "All":
        choices = choices.loc[[group_name(x, kind) == group for x in choices.index]]
    return ["All"] + choices.index.astype(str).tolist()


def _safe_widget_selection(key: str, options: list[str], default="All") -> None:
    """Clear stale cascading selections before Streamlit constructs a widget."""
    if st.session_state.get(key, default) not in options:
        st.session_state[key] = default


def market_page(jobs: pd.DataFrame, source: str, accent: str) -> None:
    """Compact hierarchical filters and a user-customizable market dashboard."""
    st.title("Market Explorer")

    st.caption(
        "Explore historical LinkedIn job postings "
        "from March 24 to April 20, 2024. "
        "All charts use the selected research dataset."
    )

    role_groups = group_counts(jobs, "job_function", "role")
    has_full_taxonomy = "industry_group" in jobs.columns

    if has_full_taxonomy:
        industry_groups = label_counts(
            jobs, "industry_group"
        )
    else:
        industry_groups = group_counts(
            jobs, "industry", "industry"
        )

    available_states = sorted_states(jobs)
    regions = [
        r for r in REGION_SUBREGIONS
        if any(state_region(x) == r for x in available_states)
    ]

    with st.container(border=True):
        role_col, region_col, industry_col, keyword_col = st.columns([1, 0.85, 1, 1.1])
        role_group = role_col.selectbox("Career Field", _frequency_options(role_groups), key="market_role_group")
        region = region_col.selectbox("Location", ["All"] + regions, key="market_region",
                                     help="Choose East, Central or West. Specific states are optional under More Filters.")
        industry_group = industry_col.selectbox("Industry Sector", _frequency_options(industry_groups), key="market_industry_group")
        keyword = keyword_col.text_input("Title Keyword", placeholder="e.g., Analyst", key="market_keyword")

        with st.expander("More Filters · Role / State / Salary / Requirements"):
            with st.container():
                detail1, detail2 = st.columns(2)
                with detail1:
                    role_opts = _ranked_specific_options(jobs, "job_function", role_group, "role")
                    _safe_widget_selection("market_role_specific", role_opts)
                    role_specific = st.selectbox("Specific Job Function", role_opts, key="market_role_specific")
                with detail2:

                    if has_full_taxonomy:
                        raw_counts = label_counts(jobs, "industry")
                        industry_opts = ["All"] + [
                             name for name in raw_counts.index
                             if industry_group == "All"
                             or LOOKUP.get(name, name) == industry_group
                             ]
                    else:
                        industry_opts = _ranked_specific_options(
                            jobs, "industry", industry_group, "industry"
                            )

                    _safe_widget_selection("market_industry_specific", industry_opts)
                    industry_specific = st.selectbox("Specific Industry", industry_opts, key="market_industry_specific")

            state_options = [s for s in available_states if region == "All" or state_region(s) == region]
            previous_states = st.session_state.get("market_states", [])
            if any(x not in state_options for x in previous_states):
                st.session_state["market_states"] = [x for x in previous_states if x in state_options]
            selected_states = st.multiselect("Specific States (optional)", state_options, key="market_states",
                help="Choose one or more states; leaving this empty includes all states in the selected region.")
            req1, req2, req3 = st.columns(3)
            salary_only = req1.checkbox("Comparable annual USD salary only", key="market_salary_only")
            experience = req2.selectbox("Experience Level", ["All"] + label_options(jobs, "experience_level"),
                                        key="market_experience")
            work_type = req3.selectbox("Work Type", ["All"] + label_options(jobs, "work_type"),
                                      key="market_work_type")
            if "degree" in jobs.columns:
                degree = st.selectbox("Detected Degree Mention", ["All"] + label_options(jobs, "degree"), key="market_degree")
            else:
                degree = "All"

            if has_full_taxonomy:
                st.caption(
                    "Industry groups follow the project's research taxonomy. "
                    "Career-field groupings remain heuristic."
                )
            else:
                st.caption(
                    "Demonstration mode uses heuristic groupings."
                )

    # The corresponding widget values persist even when the Advanced panel is closed.
    filtered = filter_labels(jobs, "job_function", group=role_group,
                             specific=role_specific, kind="role")
    filtered = filter_location(filtered, region=region)
    if selected_states:
        filtered = filtered.loc[filtered["state"].fillna("").astype(str).str.upper().isin(selected_states)]
    if has_full_taxonomy:
        if industry_group != "All":
            filtered = filtered.loc[
                filtered["industry_group"].map(
                    lambda values: industry_group in values
                )
            ]

        if industry_specific != "All":
            filtered = filtered.loc[
                filtered["industry"].map(
                    lambda values: industry_specific in values
                )
            ]

    else:
        filtered = filter_labels(
            filtered,
            "industry",
            group=industry_group,
            specific=industry_specific,
            kind="industry",
        )

    filtered = subset(subset(filtered, "experience_level", experience), "work_type", work_type)
    filtered = subset(filtered, "degree", degree)
    if keyword.strip():
        filtered = filtered[filtered["title"].fillna("").str.contains(keyword.strip(), case=False, regex=False)]
    if salary_only:
        filtered = filtered.loc[valid_salary(filtered).index]

    summary, reset = st.columns([6, 1])
    summary.caption(f"**{len(filtered):,} of {len(jobs):,} historical postings** · {source}"
                    + (f" · Region: {region}" if region != "All" else ""))
    if reset.button("Reset All", key="market_reset", use_container_width=True):
        for key in list(st.session_state):
            if key.startswith("market_"):
                del st.session_state[key]
        st.rerun()
    if source == "Full historical dataset":
        st.info(
            f"This dashboard covers {len(jobs):,} historical "
            "postings from our study dataset. "
            "It is not a representative census of the labor "
            "market or a live vacancy database."
        )
    else:
        st.info(
            f"Historical data only. Data source: {source}. "
            "These are not verified current vacancies."
        )
    if filtered.empty:
        st.warning("No postings match these filters. Try Reset All or broaden your selection.")
        return

    salaries = valid_salary(filtered)
    counts_by_function = label_counts(filtered, "job_function")
    counts_by_skill = label_counts(filtered, "skill_labels")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Historical Postings", f"{len(filtered):,}")
    k2.metric("Median Annual Salary (USD)", f"${salaries.median():,.0f}" if not salaries.empty else "Unknown")
    k3.metric("Comparable Salary Share", f"{len(salaries)/len(filtered):.1%}")
    k4.metric("Top Detected Skill", str(counts_by_skill.index[0]) if not counts_by_skill.empty else "Unknown")

    st.subheader("Explore Your Dashboard")
    chart_options = available_charts(jobs)
    if "market_charts" not in st.session_state:
        st.session_state["market_charts"] = [x for x in DEFAULT_CHARTS if x in chart_options]
    else:
        st.session_state["market_charts"] = [x for x in st.session_state["market_charts"] if x in chart_options]

    toolbar, compact = st.columns([4, 1])
    with toolbar:
        with st.popover("Customize Charts", icon="📊", use_container_width=False):
            st.multiselect("Visible Charts", options=list(chart_options),
                           format_func=lambda x: chart_options[x], key="market_charts",
                           help="Charts display in your chosen order. To change it, remove and re-add a chart.")
            st.selectbox("Charts per Row", options=[1, 2, 3], index=1, key="market_chart_columns")
            st.slider("Top Categories per Chart", 5, 20, value=10, key="market_top_n")
            st.selectbox("Category Chart Style", ["Auto", "Bar", "Donut"], key="market_chart_style",
                         help="Applies to categorical charts; salaries and timelines keep their suitable plot types.")
            if st.button("Restore Default Charts", use_container_width=True):
                st.session_state["market_charts"] = [x for x in DEFAULT_CHARTS if x in chart_options]
                st.session_state["market_chart_columns"] = 2
                st.session_state["market_top_n"] = 10
                st.session_state["market_chart_style"] = "Auto"
                st.rerun()
    with compact:
        st.caption(f"{len(st.session_state['market_charts'])} visible charts")

    chosen = st.session_state["market_charts"]
    if not chosen:
        st.info("Select at least one chart from Customize Charts.")
    else:
        ncols = st.session_state.get("market_chart_columns", 2)
        top_n = st.session_state.get("market_top_n", 10)
        for idx in range(0, len(chosen), ncols):
            cols = st.columns(ncols)
            for col, chart_id in zip(cols, chosen[idx:idx+ncols]):
                with col:
                    with st.container(border=True):
                        draw_chart(st, filtered, chart_id, accent, top_n, st.session_state.get("market_chart_style", "Auto"))
    st.caption("Charts recalculate for the current filters. Multiple labels can overlap. Salary data includes comparable disclosed full-time annual USD records only; charts are descriptive, not forecasts.")

def parse_resume(filename: str, file_bytes: bytes) -> str:
    """Read text-based PDF and DOCX resumes, without storing the original file."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if filename.lower().endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported resume format. Use PDF or DOCX.")


def matching_page(jobs: pd.DataFrame, source: str, accent: str) -> None:
    """Render resume parsing, candidate editing, and explainable job recommendations."""
    st.title("Resume Matching")
    st.caption("Discover relevant historical job postings and inspect the evidence behind each recommendation.")
    col_upload, col_profile = st.columns([1, 1.5])
    with col_upload:
        with st.container(border=True):
            st.subheader("Your Resume")
            uploaded = st.file_uploader("Upload a PDF or DOCX (max 10 MB)", type=["pdf", "docx"])
            pasted = st.text_area("Or paste resume text", height=135)
            resume = pasted.strip()
            if uploaded and uploaded.size > 10 * 1024 * 1024:
                st.error("File is larger than 10 MB. Paste text or upload a smaller resume.")
                uploaded = None
            if uploaded and not resume:
                try:
                    resume = parse_resume(uploaded.name, uploaded.getvalue()).strip()
                except Exception:
                    st.warning("Text extraction failed. Please paste your resume text instead.")
            if uploaded and not resume:
                st.info("No extractable text found. Scanned documents require pasted text.")
    with col_profile:
        with st.container(border=True):
            st.subheader("Candidate Profile")
            if not resume and st.session_state.get("studio_matching_text"):
                resume = st.session_state["studio_matching_text"]
            if resume:
                fingerprint = hashlib.sha256(resume.encode("utf-8")).hexdigest()
                if st.session_state.get("resume_matching_fingerprint") != fingerprint:
                    evidence = extract_profile_evidence(resume)
                    st.session_state["resume_matching_fingerprint"] = fingerprint
                    st.session_state["resume_matching_skills"] = ", ".join(extract_skills(resume))
                    st.session_state["resume_matching_education"] = (
                        "\n".join(evidence["education_lines"]))
                    st.session_state["resume_matching_experience"] = (
                        "\n".join(evidence["experience_lines"]))
                evidence_preview = extract_profile_evidence(resume)
                st.caption("Parsed profile overview (unverified; editable below)")
                st.markdown("**Education:** " + str(evidence_preview["education_summary"]))
                st.markdown("**Experience:** " + str(evidence_preview["experience_summary"]))
                st.caption("Dates, school names, roles and eligibility are not independently validated.")
            skills_input = st.text_area(
                "Confirmed Skills (editable)", key="resume_matching_skills", height=82,
                help="Review detected skills and correct them before matching.")
            education = st.text_area(
                "Education — extracted / editable", key="resume_matching_education", height=85,
                help="Verify degree, institution and dates yourself; this is a conservative text extraction.")
            experience = st.text_area(
                "Experience — extracted / editable", key="resume_matching_experience", height=105,
                help="Confirm roles and dates yourself; no employer or seniority verification is performed.")
    settings_a, settings_b, settings_c = st.columns(3)
    method = settings_a.selectbox("Ranking Method", ["Hybrid", "TF-IDF", "Skill-only"])
    top_k = settings_b.selectbox("Top Jobs", [5, 10, 20])
    target = settings_c.selectbox("Preferred Job Function", ["All"] + label_options(jobs, "job_function"))
    st.caption("Scores are model relevance signals, not hiring probabilities or verified qualification assessments.")
    if not st.button("Find Matching Jobs", type="primary", use_container_width=True):
        st.info("Upload or paste a resume, confirm your skills, and select Find Matching Jobs.")
        return
    skills = list(dict.fromkeys(s.strip() for s in skills_input.split(",") if s.strip()))
    if not skills and not education.strip() and not experience.strip():
        st.warning("Add at least one confirmed skill or some education/experience text.")
        return
    candidate = {"candidate_id": "web_user", "skills": skills,
                 "education": education.strip(), "experience": experience.strip()}
    matcher = get_matcher(jobs, source)
    scored = score_jobs(candidate, jobs, matcher=matcher, skills_column="skill_labels", text_weight=0.5)
    ranking_method = {"Hybrid": "hybrid_score", "TF-IDF": "text_score", "Skill-only": "skill_score"}[method]
    displayed = subset(scored, "job_function", target)
    displayed = displayed.dropna(subset=[ranking_method]).sort_values(
        [ranking_method, "job_id"], ascending=[False, True]).head(top_k)
    st.subheader("Recommended Historical Jobs")
    st.caption(f"Method: {method} · Job pool: {source} · Baseline text weight: 0.5 for Hybrid")
    if displayed.empty:
        st.warning("No ranked postings are available for this selection.")
        return
    for rank, (_, job) in enumerate(displayed.iterrows(), start=1):
        title = str(job.get("title", "Unknown Title"))
        score = float(job[ranking_method])
        with st.expander(f"#{rank}  {title}  ·  Relevance {score:.3f} / 1", expanded=(rank == 1)):
            company = job.get("company_name", "Company unavailable")
            location = job.get("location", "Location unavailable")
            st.caption(f"{company} · {location} · Historical posting")
            t1, t2, t3 = st.columns(3)
            t1.metric("TF-IDF Similarity", f"{job['text_score']:.3f}")
            t2.metric("Skill Coverage", f"{job['skill_score']:.1%}" if pd.notna(job["skill_score"]) else "Unknown")
            t3.metric("Hybrid Relevance", f"{job['hybrid_score']:.3f}")
            matched = to_labels(job["matched_skills"])
            unknown = to_labels(job["unverified_skills"])
            st.write("**Matched detected skills:**", ", ".join(matched) or "None confirmed")
            st.write("**Unverified detected skills:**", ", ".join(unknown) or "None detected")
            reason = "Text overlap with the candidate profile"
            if matched:
                reason += "; confirmed overlap includes " + ", ".join(matched[:5])
            if pd.isna(job["skill_score"]):
                reason += ". Skill evidence is unavailable, so Hybrid falls back to TF-IDF"
            st.write("**Why recommended:** " + reason + ".")
            st.caption("Unverified means not confirmed by this extractor; it does not prove a skill is absent.")
            with st.expander("Original job description"):
                st.write(job.get("description", "Description unavailable"))
    st.subheader("Skills to Explore")
    missing = Counter(v for values in displayed["unverified_skills"] for v in to_labels(values))
    if missing:
        st.dataframe(pd.DataFrame(missing.most_common(10), columns=["Unverified Skill", "Jobs in Top Results"]),
                     hide_index=True, use_container_width=True)
    else:
        st.info("No unverified detected skills in the displayed recommendations.")
    st.subheader("Alternative Career Paths")
    if "job_function" in scored.columns:
        links = scored[["job_id", "job_function"]].explode("job_function").rename(columns={"job_function": "category"})
        links = links.dropna(subset=["category"])
        if not links.empty:
            paths = rank_job_categories(scored[["job_id", "text_score"]], links,
                                         score_column="text_score", m=5)
            st.dataframe(paths.head(8), hide_index=True, use_container_width=True)
        else:
            st.info("Category labels are not available for these jobs.")
    else:
        st.info("Category labels are not available in this dataset.")


def resume_studio_page() -> None:
    """Private in-session resume editor and editable DOCX generator."""
    st.title("Resume Studio")
    st.caption("Build a clean, single-column resume from confirmed information. No account or server-side resume storage is configured.")
    left, right = st.columns([1.1, 1.3])
    with left:
        st.subheader("Import & Organize")
        source = st.file_uploader("Import PDF / DOCX", type=["pdf", "docx"], key="studio_upload")
        pasted = st.text_area("Or paste existing resume text", height=100, key="studio_paste")
        if st.button("Extract from Resume", use_container_width=True):
            raw = pasted.strip()
            if not raw and source:
                if source.size > 10 * 1024 * 1024:
                    st.error("File is larger than 10 MB.")
                    return
                try:
                    raw = parse_resume(source.name, source.getvalue()).strip()
                except Exception as exc:
                    st.error("Could not extract text. Try a text-based PDF/DOCX or paste text.")
                    return
            if not raw:
                st.warning("Please upload a text-based resume or paste text.")
            else:
                evidence = extract_profile_evidence(raw)
                st.session_state["studio_education"] = "\n".join(evidence["education_lines"])
                st.session_state["studio_experience"] = "\n".join(evidence["experience_lines"])
                st.session_state["studio_projects"] = "\n".join(evidence["project_lines"])
                st.session_state["studio_skills"] = ", ".join(extract_skills(raw))
                st.session_state["studio_email"] = evidence["email"]
                st.session_state["studio_matching_text"] = raw
                st.success("Draft extracted. Review every field before exporting.")
        template = st.selectbox("Resume Layout", list(TEMPLATES), key="studio_template")
        st.caption("Technical Classic: compact tech resume · Editorial Modern: refined hierarchy · Research Minimal: academic-friendly layout")
        st.info("Imported formatting, dates and roles may be incomplete. The editor never invents experience or achievements.")
    with right:
        st.subheader("Edit Your Resume")
        c1,c2 = st.columns(2)
        name = c1.text_input("Full Name", key="studio_name")
        email = c2.text_input("Email", key="studio_email")
        location = c1.text_input("Location", key="studio_location")
        links = c2.text_input("LinkedIn / Portfolio", key="studio_links")
        summary = st.text_area("Professional Summary", height=75, key="studio_summary")
        education = st.text_area("Education", height=100, key="studio_education")
        experience = st.text_area("Experience (one entry per line, bullets with '- ')", height=155, key="studio_experience")
        projects = st.text_area("Projects", height=110, key="studio_projects")
        skills = st.text_area("Skills", height=75, key="studio_skills")
        fields = {"name":name, "email":email, "location":location, "links":links,
                  "summary":summary, "education":education, "experience":experience,
                  "projects":projects, "skills":skills}
        export = build_resume_docx(fields, template)
        if st.button("Use Edited Profile for Job Matching", use_container_width=True):
            composed = "\n".join(
                f"{label}\n{value.strip()}" for label, value in [
                    ("Education", education), ("Experience", experience),
                    ("Projects", projects), ("Skills", skills)] if value.strip()
            )
            if not composed.strip():
                st.warning("Add some education, experience, project or skill details first.")
            else:
                st.session_state["studio_matching_text"] = composed
                st.success("Saved for this session. Open Job Matching in the top navigation.")
        st.download_button("Download Editable DOCX", data=export,
                           file_name="jobscope_resume.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           type="primary", use_container_width=True)
        with st.expander("Preview Resume Text", expanded=False):
            st.markdown(f"**{name or 'Your Name'}**")
            st.caption(" · ".join(v for v in (email, location, links) if v))
            for title, value in [("Summary", summary), ("Education", education),
                                 ("Experience", experience), ("Projects", projects), ("Skills", skills)]:
                if value.strip():
                    st.markdown(f"**{title}**")
                    st.text(value)
    st.caption("Resume Studio creates editable documents; it does not verify claims or guarantee automated screening results.")



def main() -> None:
    """Configure the Streamlit application, visual theme, and page navigation."""
    st.set_page_config(
        page_title="JobScope · Career Intelligence",
        page_icon="📊",
        layout="wide",
    )

    heading, theme = st.columns([5, 1])

    heading.markdown(
        "### ◈ JobScope  "
        "<span style='font-size:0.82rem;opacity:.6;"
        "font-weight:400'>CAREER INTELLIGENCE</span>",
        unsafe_allow_html=True,
    )

    selected_theme = theme.selectbox(
        "Design Palette",
        list(THEMES),
        label_visibility="collapsed",
    )

    accent, _ = theme_style(selected_theme)

    def market_view():
        """Load the available market dataset and render the market dashboard."""
        try:
            if full_market_available():
                jobs = load_market_data()
                source = "Full historical dataset"
            else:
                jobs, source = load_jobs()
                st.warning(
                    "Full dataset unavailable. "
                    f"Displaying: {source}."
                )

            market_page(jobs, source, accent)

        except Exception as exc:
            st.error(
                f"Market data could not be loaded: {exc}"
            )

    def matching_view():
        """Load the matching dataset and render the recommendation interface."""
        try:
            jobs, source = load_jobs()
            matching_page(jobs, source, accent)

        except Exception as exc:
            st.error(
                f"Matching data could not be loaded: {exc}"
            )

    nav = st.navigation(
        [
            st.Page(
                market_view,
                title="Market Explorer",
                icon=":material/monitoring:",
                url_path="market",
                default=True,
            ),
            st.Page(
                matching_view,
                title="Job Matching",
                icon=":material/work:",
                url_path="matching",
            ),
            st.Page(
                resume_studio_page,
                title="Resume Studio",
                icon=":material/description:",
                url_path="studio",
            ),
        ],
        position="top",
    )

    nav.run()


if __name__ == "__main__":
    main()
