
"""Analyze NLP skill-detection coverage across job categories."""

from collections import Counter
from pathlib import Path

import pandas as pd

from src.feature_extraction import extract_skills


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/sample/sample_jobs.csv"
OUTPUT_PATH = ROOT / "docs/nlp_error_analysis.md"


def safe_markdown(value, max_length=140):
    """Format a value safely for a Markdown table."""
    text = str(value).replace("|", "/").replace("\n", " ")
    text = " ".join(text.split())
    return text[:max_length]


def main():
    jobs = pd.read_csv(INPUT_PATH)
    jobs["description"] = jobs["description"].fillna("")
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)
    jobs["skill_count"] = jobs["skill_labels"].apply(len)
    jobs["has_skills"] = jobs["skill_count"] > 0

    total = len(jobs)
    detected = int(jobs["has_skills"].sum())
    undetected = total - detected
    coverage = detected / total if total else 0

    print(f"Total jobs: {total}")
    print(f"Jobs with detected skills: {detected}")
    print(f"Jobs without detected skills: {undetected}")
    print(f"Skill detection coverage: {coverage:.1%}")

    # Group by job function as recorded in the cleaned sample
    category_stats = (
        jobs.assign(
            job_function=jobs["job_function"].fillna("Unknown")
        )
        .groupby("job_function", dropna=False)
        .agg(
            total_jobs=("job_id", "size"),
            detected_jobs=("has_skills", "sum"),
            average_skills=("skill_count", "mean"),
        )
        .reset_index()
    )

    category_stats["coverage"] = (
        category_stats["detected_jobs"]
        / category_stats["total_jobs"]
    )

    category_stats = category_stats.sort_values(
        ["total_jobs", "job_function"],
        ascending=[False, True],
    )

    # Count the most frequently detected skill labels
    skill_counts = Counter(
        skill
        for labels in jobs["skill_labels"]
        for skill in labels
    )

    # Inspect samples where no skills were detected
    zero_skill_jobs = jobs.loc[
        ~jobs["has_skills"]
    ].sort_values("job_id").head(10)

    # Inspect jobs with exactly one detected skill
    one_skill_jobs = jobs.loc[
        jobs["skill_count"] == 1
    ].sort_values("job_id").head(10)

    lines = [
        "# JobScope — NLP Skill Detection Error Analysis",
        "",
        "**Dataset:** 500-job cleaned sample",
        "**Method:** Dictionary-based skill extraction",
        "**Evaluation period:** November 2026",
        "",
        "## 1. Overall Detection Coverage",
        "",
        f"- Total jobs: {total}",
        f"- Jobs with detected skills: {detected}",
        f"- Jobs without detected skills: {undetected}",
        f"- Skill detection coverage: {coverage:.1%}",
        "",
        "Detection coverage is not recall or extraction accuracy.",
        "Missing labels do not establish missing job requirements.",
        "",
        "## 2. Coverage by Job Function",
        "",
        "| Job Function | Jobs | Detected | Coverage | Avg. Skills |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in category_stats.iterrows():
        lines.append(
            f"| {safe_markdown(row['job_function'])} "
            f"| {int(row['total_jobs'])} "
            f"| {int(row['detected_jobs'])} "
            f"| {row['coverage']:.1%} "
            f"| {row['average_skills']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Most Frequently Detected Skills",
        "",
        "| Skill | Job Count |",
        "|---|---:|",
    ])

    for skill, count in skill_counts.most_common(20):
        lines.append(
            f"| {safe_markdown(skill)} | {count} |"
        )

    lines.extend([
        "",
        "## 4. Zero-Skill Detection Examples",
        "",
        "| Job ID | Title | Description Excerpt |",
        "|---|---|---|",
    ])

    for _, row in zero_skill_jobs.iterrows():
        lines.append(
            f"| {row['job_id']} "
            f"| {safe_markdown(row['title'])} "
            f"| {safe_markdown(row['description'])} |"
        )

    lines.extend([
        "",
        "## 5. Single-Skill Detection Examples",
        "",
        "| Job ID | Title | Detected Skill |",
        "|---|---|---|",
    ])

    for _, row in one_skill_jobs.iterrows():
        lines.append(
            f"| {row['job_id']} "
            f"| {safe_markdown(row['title'])} "
            f"| {safe_markdown(row['skill_labels'][0])} |"
        )

    lines.extend([
        "",
        "## 6. Known Recommendation Failure Mode",
        "",
        "Skill coverage can equal 1.0 when only one skill "
        "was detected and the candidate possesses it.",
        "",
        "Example observed in the integration demo:",
        "",
        "- Job: Veterans Affairs Specialist",
        "- Text similarity: approximately 0.0085",
        "- Skill coverage: 1.0",
        "- Hybrid score: approximately 0.5042",
        "",
        "This creates a risk of promoting weakly related jobs.",
        "The example does not prove that the entire ranking "
        "system performs poorly.",
        "",
        "## 7. Manual Review and Next Steps",
        "",
        "- [ ] Review zero-skill examples for possible omissions.",
        "- [ ] Review single-skill examples for false positives.",
        "- [ ] Identify occupation-specific vocabulary gaps.",
        "- [ ] Compare text-only and hybrid ranking relevance.",
        "- [ ] Validate ranking quality using independent labels.",
        "",
        "## 8. Limitations",
        "",
        "The analysis reports automated detection statistics, "
        "not manually verified precision or recall.",
        "Job-function values are grouped as stored in the sample.",
        "The sample may not represent the full U.S. job market.",
        "",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
