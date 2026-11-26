"""Create reproducible JobScope evaluation figures from existing CSV outputs.

Run from the repository root:
    python -m scripts.analyze_evaluation_results

Required inputs (relative to repository root):
    data/processed/evaluation_results/model_comparison_by_profile.csv
    data/processed/evaluation_results/provisional_consensus.csv
    data/processed/evaluation_results/ranked_judged_jobs.csv

This script does not modify recommendation algorithms or annotations.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/evaluation_results"
FIGURES = ROOT / "figures/evaluation"
FIGURES.mkdir(parents=True, exist_ok=True)
PROFILES = ["candidate_ds", "candidate_finance", "candidate_health"]
PROFILE_LABELS = ["Data Science", "Finance", "Healthcare"]
METHODS = ["text", "skill", "hybrid"]
METHOD_LABELS = {"text": "TF-IDF", "skill": "Skill-only", "hybrid": "Hybrid (0.5)"}
COLORS = {"text": "#2563EB", "skill": "#D97706", "hybrid": "#059669"}


def save(fig, filename):
    fig.savefig(FIGURES / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES / filename}")


def performance_figure(df):
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(PROFILES))
    width = 0.24
    for j, method in enumerate(METHODS):
        vals = []
        for profile in PROFILES:
            subset = df[(df.candidate_id == profile) & (df.method == method)]
            if len(subset) != 1:
                raise ValueError(f"Expected exactly one row for {profile}/{method}")
            vals.append(float(subset.iloc[0].ndcg_at_5))
        for i, val in enumerate(vals):
            xpos = x[i] + (j - 1) * width
            if np.isnan(val):
                ax.text(xpos, 0.035, "N/A", ha="center", va="bottom", rotation=90,
                        fontsize=10, color="#6B7280")
            else:
                ax.bar(xpos, val, width, color=COLORS[method], label=METHOD_LABELS[method] if i == 0 else None)
                ax.text(xpos, val + 0.022, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, PROFILE_LABELS)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("NDCG@5 (higher is better)")
    ax.set_title("JobScope | Ranking quality by candidate profile", fontsize=15, weight="bold", pad=15)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    ax.legend(handles, [METHOD_LABELS[m] for m in METHODS], loc="upper right", frameon=False, ncol=3)
    fig.text(0.08, 0.01, "15 judged jobs per profile | provisional median labels | skill-only N/A = incomplete ranking",
             color="#4B5563", fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, "model_performance_comparison.png")


def agreement_figure(df):
    cols = ["score_jessy", "score_aileen", "score_damir"]
    if not all(c in df.columns for c in cols):
        raise ValueError("Missing annotator scores")
    pairs = [("Jessy (AI-assisted)", "Aileen", cols[0], cols[1]),
             ("Jessy (AI-assisted)", "Damir", cols[0], cols[2]),
             ("Aileen", "Damir", cols[1], cols[2])]
    agreement, kappa = [], []
    for a, b, a_col, b_col in pairs:
        score_a, score_b = df[a_col].astype(int), df[b_col].astype(int)
        agreement.append(float((score_a == score_b).mean()))
        kappa.append(float(cohen_kappa_score(score_a, score_b, labels=[0,1,2], weights="quadratic")))
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    y = np.arange(len(pairs))
    h = .32
    bars1 = ax.barh(y - h/2, agreement, h, color="#2563EB", label="Exact agreement")
    bars2 = ax.barh(y + h/2, kappa, h, color="#059669", label="Quadratic-weighted kappa")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_width() + .01, b.get_y() + b.get_height()/2, f"{b.get_width():.3f}",
                    ha="left", va="center", fontsize=10)
    ax.set_yticks(y, [f"{a} vs {b}" for a,b,_,_ in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Agreement statistic (0–1)")
    ax.set_title("JobScope | Annotation agreement", fontsize=15, weight="bold", pad=14)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.11, .02, "45 identical candidate-job pairs | Jessy reviewed AI-suggested labels; not independent blind annotation",
             fontsize=9, color="#4B5563")
    fig.tight_layout(rect=[0, .04, 1, 1])
    save(fig, "annotation_agreement.png")


def errors_figure(ranked):
    examples = [
        ("Data Science", "candidate_ds", "Senior Data Analyst", 2, "Relevant job demoted"),
        ("Finance", "candidate_finance", "Database Administrator", 0, "Irrelevant job promoted"),
        ("Healthcare", "candidate_health", "Nurse Practitioner", 0, "Eligibility not verified"),
    ]
    table_rows = []
    csv_rows = []
    for label, candidate, title, human, issue in examples:
        q = ranked[(ranked.candidate_id == candidate) & (ranked.title == title)]
        def rank_for(method):
            row = q[q.method == method]
            return int(row.iloc[0]["rank"]) if len(row) == 1 else None
        text_rank, hybrid_rank = rank_for("text"), rank_for("hybrid")
        if text_rank is None or hybrid_rank is None:
            raise ValueError(f"Missing rank for {candidate}/{title}")
        actual = set(q.relevance_score.astype(int).tolist())
        if actual != {human}:
            raise ValueError(f"Label mismatch for {title}: {actual}")
        table_rows.append([label, title, f"{human}", f"#{text_rank}", f"#{hybrid_rank}", issue])
        csv_rows.append(dict(candidate_id=candidate, title=title, relevance_score=human,
                             text_rank=text_rank, hybrid_rank=hybrid_rank, issue=issue))
    pd.DataFrame(csv_rows).to_csv(DATA / "ranking_error_cases.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axis("off")
    table = ax.table(cellText=table_rows,
                     colLabels=["Profile", "Example job", "Human label", "TF-IDF rank", "Hybrid rank", "Interpretation"],
                     colWidths=[.11, .29, .11, .11, .11, .27], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#F3F7FA")
    ax.set_title("JobScope | Representative ranking errors", fontsize=16, weight="bold", pad=20)
    fig.text(.06, .035, "Provisional median relevance: 0 = irrelevant; 1 = partly relevant; 2 = highly relevant.\n"
             "Interpretations are hypotheses, not verified causes; inspect extracted skills and job requirements before attribution.",
             fontsize=9, color="#4B5563")
    fig.tight_layout(rect=[.02, .08, .98, 1])
    save(fig, "ranking_error_analysis.png")


def main():
    comparison = pd.read_csv(DATA / "model_comparison_by_profile.csv")
    labels = pd.read_csv(DATA / "provisional_consensus.csv")
    ranked = pd.read_csv(DATA / "ranked_judged_jobs.csv")
    if len(labels) != 45:
        raise ValueError(f"Expected 45 judged pairs, got {len(labels)}")
    performance_figure(comparison)
    agreement_figure(labels)
    errors_figure(ranked)
    print("Evaluation figure generation completed.")

if __name__ == "__main__":
    main()
