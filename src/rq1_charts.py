"""Tables and charts for the RQ1 part of the presentation and website.

The story goes in resume order: education (degree, major), then experience
(years by level and by industry), then skills, then which job functions sit
close together, then pay. Each chart has one "table" function that makes the
numbers and one plot function that draws them, so the tests can check the
numbers without drawing anything.

Before calling these, run industry_groups.add_industry_group(jobs) so the
industry column holds the plain groups (CS/IT, Finance, ...).
"""

from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import resume_analysis as ra
from src.industry_groups import STAFFING
from src.market_analysis import _valid_salary

LEVELS = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
SOURCE = "Source: LinkedIn job postings (Kaggle), listed Mar 24 to Apr 20, 2024. U.S. only."
GROUP_NOTE = ("Industry = the hiring company's LinkedIn industry, merged into plain groups "
              "(for example, CS/IT = IT services, software, internet, telecom and hardware companies).")

# colors (same warm palette as our slides)
INK, QUIET, BG = "#2B211C", "#7A6A5E", "#F2EBDD"
LIGHT, MID, GREEN, DARK = "#DDD3C2", "#8FB59A", "#2E5A3A", "#1B3A24"
GOLD, BLUE, GRAY = "#A87A2E", "#6E8FA0", "#8A7F74"
ROW_SHADE = "#EAE1D0"
GREENS = LinearSegmentedColormap.from_list("greens", ["#F7F2E8", "#CFE0D2", "#8FB59A", "#2E5A3A", "#1B3A24"])

# use a clean font if the computer has one, otherwise matplotlib's default
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


# ---------- small helpers ----------

def _explode(jobs, cols):
    """One row per (posting, label) for the list column in cols[0]."""
    return jobs[cols].explode(cols[0]).dropna()


def _titles(fig, height, title, subtitle, note):
    # title on top, gray "how to read" line under it, source note at the bottom
    fig.text(0.04, 1 - 0.3 / height, title, fontsize=21, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 1 - 0.78 / height, subtitle, fontsize=13, color=QUIET, va="top")
    fig.text(0.04, 0.12 / height, note, fontsize=10, color=QUIET)


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=130, facecolor=BG)
    return fig


# ---------- 1. minimum degree by industry ----------

def minimum_degree(jobs, min_named=500):
    """Lowest degree each posting names, as shares per industry group.

    High school and associate are merged into "Below bachelor". Only postings
    that name a degree are counted, and groups with fewer than min_named of
    those are dropped.
    """
    t = ra.degree_requirements(jobs, by="industry", min_n=1, top=None).drop(STAFFING, errors="ignore")
    named = (t["postings"] * t["states_degree"]).round().astype(int)
    out = pd.DataFrame({
        "Below bachelor": t["High school"] + t["Associate"],
        "Bachelor": t["Bachelor"],
        "Master": t["Master"],
        "PhD": t["PhD"],
    })
    out["master_or_higher"] = out["Master"] + out["PhD"]
    out["postings_naming_degree"] = named
    out = out[out["postings_naming_degree"] >= min_named]
    return out.sort_values("master_or_higher", ascending=False)


def plot_stacked(table, cols, colors, title, subtitle, note, right_labels, right_head, path=None):
    """Horizontal 100% bars, one row per group, with a text column on the right."""
    h = 2.6 + 0.42 * len(table)
    fig, ax = plt.subplots(figsize=(12, h), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.3, right=0.8, top=1 - 1.55 / h, bottom=0.95 / h)
    rows = table[::-1]
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for col, color in zip(cols, colors):
        w = rows[col].to_numpy() * 100
        ax.barh(y, w, left=left, color=color, height=0.7, edgecolor=BG, linewidth=1.5, label=col)
        for yy, (x0, ww) in enumerate(zip(left, w)):
            if ww >= 7:  # no label on tiny pieces, they would overlap
                ax.text(x0 + ww / 2, yy, f"{ww:.0f}%", ha="center", va="center", fontsize=11.5,
                        fontweight="bold", color="white" if color in (GREEN, DARK) else INK)
        left += w
    for yy, lab in zip(y, right_labels[::-1]):
        ax.text(102, yy, lab, va="center", fontsize=11.5, color=INK)
    ax.text(102, len(rows) - 0.3, right_head, va="bottom", fontsize=11, color=QUIET, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(rows.index, fontsize=12.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(ncol=len(cols), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=12, handlelength=1.3)
    _titles(fig, h, title, subtitle, note)
    return _save(fig, path)


def plot_minimum_degree(table, path=None):
    labels = [f"{m:.0%}   n = {n:,}" for m, n in zip(table["master_or_higher"], table["postings_naming_degree"])]
    return plot_stacked(
        table, ["Below bachelor", "Bachelor", "Master", "PhD"], [LIGHT, MID, GREEN, DARK],
        "Minimum degree asked in job postings, by industry",
        "Lowest degree each posting names. Each bar = 100% of postings that name a degree. Sorted by master's or higher.",
        SOURCE + "\nIndustries with 500+ postings that name a degree. Staffing and recruiting firms left out, "
        "because they post jobs for other companies.\n" + GROUP_NOTE,
        labels, "Master+   postings", path)


# ---------- 2. majors by industry, 5. skills by industry ----------

def top_items(share, sizes, k, n_groups, min_share=0.0):
    """Top k columns of a share table for the n_groups biggest rows, in long format."""
    sizes = sizes.drop(STAFFING, errors="ignore").sort_values(ascending=False).head(n_groups)
    rows = []
    for group in sizes.index:
        best = share.loc[group].sort_values(ascending=False).head(k)
        best = best[best >= min_share]
        for rank, (item, value) in enumerate(best.items(), start=1):
            rows.append({"industry": group, "rank": rank, "item": item, "share": value,
                         "postings": int(sizes[group])})
    return pd.DataFrame(rows)


def top_majors(jobs, n_industries=15, min_named=300):
    # postings that name no major are left out of the share (that's what major_share does)
    share, sizes = ra.major_share(jobs, by="industry", min_n=min_named, top_groups=None, top_majors=30)
    return top_items(share, sizes, k=3, n_groups=n_industries).rename(columns={"item": "major"})


def top_skills(jobs, job_skills, n_industries=10, min_share=0.03):
    share, sizes = ra.skill_share_by_group(jobs, job_skills, by="industry", min_n=1, top_skills=500)
    # "Manufacturing" is in the skill list but it is really the industry word, not a skill
    share = share.drop(columns=["Manufacturing"], errors="ignore")
    return top_items(share, sizes, k=4, n_groups=n_industries, min_share=min_share).rename(columns={"item": "skill"})


SHORT_MAJORS = {"Marketing / Communications": "Marketing / Comms", "Psychology / Social Sciences": "Psychology / Social Sci.",
                "Life / Physical Sciences": "Life / Phys. Sciences", "Information Systems / IT": "Info Systems / IT",
                "Supply Chain / Logistics": "Supply Chain"}


def plot_top_items(long, item_col, heads, title, subtitle, note, bar_full=0.45, path=None):
    """One row per industry, top items side by side with a short bar under each."""
    groups = long["industry"].drop_duplicates().tolist()
    k = len(heads)
    row_in = 0.72
    h = 2.8 + row_in * len(groups)
    fig = plt.figure(figsize=(14, h), facecolor=BG)
    top_y, row_h = 1 - 1.75 / h, row_in / h
    x0s = np.linspace(0.27, 0.81, k) if k == 4 else np.array([0.30, 0.53, 0.76])
    colw = 0.16 if k == 4 else 0.2
    for i, head in enumerate(heads):
        fig.text(x0s[i], top_y + 0.1 / h, head, fontsize=12, color=QUIET, fontweight="bold")
    for i, group in enumerate(groups):
        y = top_y - (i + 0.6) * row_h
        if i % 2 == 0:  # shade every other row so it is easier to follow across
            fig.patches.append(plt.Rectangle((0.03, y - row_h * 0.5), 0.95, row_h, transform=fig.transFigure,
                                             facecolor=ROW_SHADE, edgecolor="none", zorder=0))
        part = long[long["industry"] == group]
        fig.text(0.04, y + row_h * 0.1, group, fontsize=13, color=INK, va="center", fontweight="bold")
        fig.text(0.04, y - row_h * 0.22, f"n = {int(part['postings'].iloc[0]):,}", fontsize=10.5, color=QUIET, va="center")
        for j, (_, r) in enumerate(part.iterrows()):
            fig.patches.append(plt.Rectangle((x0s[j], y - row_h * 0.3), colw * min(r["share"] / bar_full, 1), row_h * 0.16,
                                             transform=fig.transFigure, facecolor=GREEN if j == 0 else MID, edgecolor="none"))
            name = SHORT_MAJORS.get(r[item_col], r[item_col])
            fig.text(x0s[j], y + row_h * 0.12, f"{name}  {r['share']:.0%}", fontsize=12, va="center", color=INK,
                     fontweight="bold" if j == 0 else "normal")
    _titles(fig, h, title, subtitle, note)
    return _save(fig, path)


def plot_majors(long, path=None):
    return plot_top_items(
        long, "major", ["Most named major", "2nd", "3rd"],
        "Top 3 majors named in job postings, by industry",
        "Share of the industry's postings (that name a major) naming each major. A posting can name several majors.",
        SOURCE + "\nThe 15 industries with the most postings that name a major. Staffing and recruiting firms left out.\n"
        + GROUP_NOTE, bar_full=0.7, path=path)


def plot_skills(long, path=None):
    return plot_top_items(
        long, "skill", ["Most asked skill", "2nd", "3rd", "4th"],
        "Most asked skills in job postings, by industry",
        "Share of the industry's postings that mention each skill. Bar length = share.",
        SOURCE + "\nThe 10 industries with the most postings. Staffing and recruiting firms left out. Skills from the "
        "project's fixed skill list. Skills named in under 3% of postings not shown.\n" + GROUP_NOTE,
        bar_full=0.45, path=path)


# ---------- 3. years asked by LinkedIn level ----------

YEAR_BINS = ["1 year", "2 to 3 years", "4 to 5 years", "6+ years"]


def years_by_level(jobs, levels=LEVELS):
    """Mix of minimum years asked per LinkedIn level (only postings that state years)."""
    df = jobs[["experience_level", "min_years"]].dropna()
    df = df[df["experience_level"].isin(levels)]
    # same cut points as text_signals.years_group: <=1, <=3, <=5, more
    df["years_bin"] = pd.cut(df["min_years"], [-np.inf, 1, 3, 5, np.inf], labels=YEAR_BINS)
    mix = pd.crosstab(df["experience_level"], df["years_bin"], normalize="index")
    mix = mix.reindex(index=[lvl for lvl in levels if lvl in mix.index], columns=YEAR_BINS, fill_value=0)
    g = df.groupby("experience_level")["min_years"]
    mix["share_3plus"] = g.apply(lambda s: (s >= 3).mean())
    mix["median_years"] = g.median()
    mix["postings_naming_years"] = g.size()
    return mix


def plot_years_by_level(table, path=None):
    labels = [f"{s:.0%}   {m:.0f} yrs   n = {int(n):,}"
              for s, m, n in zip(table["share_3plus"], table["median_years"], table["postings_naming_years"])]
    return plot_stacked(
        table, YEAR_BINS, ["#CFE0D2", MID, GREEN, DARK],
        "Years of experience asked, by LinkedIn level",
        "Minimum years each posting names. Each bar = 100% of postings at that level that name a number of years.",
        SOURCE + "\nOnly postings that state a number of years. \"3-5 years\" counts as 3. The 3+ yrs column includes "
        "postings asking exactly 3, which sit in the 2 to 3 bar.",
        labels, "3+ yrs   median   postings", path)


# ---------- 4. years by industry and level ----------

def years_grid(jobs, levels=("Entry level", "Associate", "Mid-Senior level", "Director"),
               n_industries=10, min_n=50, leave_out=("Retail",)):
    """Median minimum years per industry group and level.

    Cells with fewer than min_n postings become NaN. I leave Retail out because
    its LinkedIn levels don't follow years at all (Mid-Senior asks fewer years
    than Entry), so the row would not mean anything.
    """
    df = _explode(jobs, ["industry", "experience_level", "min_years"])
    df = df[~df["industry"].isin([STAFFING, *leave_out])]
    top = df["industry"].value_counts().head(n_industries).index
    df = df[df["industry"].isin(top) & df["experience_level"].isin(levels)]
    g = df.groupby(["industry", "experience_level"])["min_years"]
    med = g.median().unstack().reindex(columns=list(levels))
    n = g.size().unstack().reindex(columns=list(levels)).fillna(0).astype(int)
    med = med.where(n >= min_n)
    order = med.sort_values(["Entry level", "Mid-Senior level"]).index
    return med.loc[order], n.loc[order]


def plot_grid(values, fmt, title, subtitle, note, path=None, vmin=0, vmax=10, dark_from=5.5,
              col_labels=None, right=None, right_head=None, width=13):
    """Grid of numbers with a light-to-dark green background. NaN cells say too few."""
    rows, cols = values.shape
    h = 3.6 + 0.5 * rows
    fig, ax = plt.subplots(figsize=(width, h), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.27, right=0.86 if right is not None else 0.97, top=1 - 2.4 / h, bottom=1.15 / h)
    v = values.to_numpy(float)
    ax.imshow(v, cmap=GREENS, vmin=vmin, vmax=vmax, aspect="auto")  # NaN cells stay empty
    for i in range(rows):
        for j in range(cols):
            x = v[i, j]
            if np.isnan(x):
                ax.text(j, i, "too few", ha="center", va="center", fontsize=10, color=QUIET)
            else:
                ax.text(j, i, fmt(x), ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if x >= dark_from else INK)
    if right is not None:
        for i, val in enumerate(right):
            ax.text(cols - 0.3, i, f"{int(val):,}", va="center", fontsize=11, color=QUIET)
        ax.text(cols - 0.3, -0.75, right_head, va="bottom", fontsize=11, color=QUIET, fontweight="bold")
    ax.set_xticks(range(cols))
    ax.set_xticklabels(col_labels or list(values.columns), fontsize=12.5, linespacing=1.4)
    ax.xaxis.tick_top()
    ax.set_yticks(range(rows))
    ax.set_yticklabels(values.index, fontsize=12.5)
    ax.set_xticks(np.arange(-.5, cols), minor=True)
    ax.set_yticks(np.arange(-.5, rows), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _titles(fig, h, title, subtitle, note)
    return _save(fig, path)


def plot_years_grid(med, path=None):
    return plot_grid(
        med, lambda x: f"{x:g} yr" if x == 1 else f"{x:g} yrs",
        "Median years of experience asked, by industry and level",
        "Read across a row to see how the years asked grow with each LinkedIn level in that industry. Darker = more years.",
        SOURCE + "\nThe 10 industries with the most postings that state years. Cells with under 50 postings marked too few. "
        "Internship and Executive left out: under 50 postings in most industries.\nRetail left out: its LinkedIn levels "
        "do not follow years (Mid-Senior level asks a median of 1 year, Entry level 2).\n" + GROUP_NOTE,
        path=path)


# ---------- 6. job functions that share postings ----------

AREAS = {
    "Tech, data and design": ["Information Technology", "Engineering", "Analyst", "Research", "Project Management",
                              "Design", "Art/Creative"],
    "Business and marketing": ["Sales", "Business Development", "Marketing", "Public Relations", "Writing/Editing"],
    "Finance": ["Finance", "Accounting/Auditing"],
    "Teaching": ["Education", "Training"],
}
AREA_COLORS = {"Tech, data and design": GREEN, "Business and marketing": GOLD, "Finance": BLUE, "Teaching": GRAY}


def function_pairs(jobs, min_postings=1000, skip=("Other", "General Business")):
    """Every pair of job functions with how many postings carry both labels.

    overlap = shared postings / postings of the smaller function. So 0.92 for
    Sales + Business Development means 92% of the smaller one's postings also
    carry the other label.
    """
    labels = jobs["job_function"][jobs["job_function"].apply(len) > 0]
    sizes = labels.explode().value_counts()
    keep = set(sizes[sizes >= min_postings].index) - set(skip)
    shared = Counter()
    for tags in labels:
        tags = sorted(set(t for t in tags if t in keep))
        for a, b in combinations(tags, 2):
            shared[(a, b)] += 1
    rows = [(a, b, c, c / min(sizes[a], sizes[b])) for (a, b), c in shared.items()]
    out = pd.DataFrame(rows, columns=["function_a", "function_b", "shared_postings", "overlap"])
    return out.sort_values("overlap", ascending=False).reset_index(drop=True)


def strong_pairs(pairs, min_overlap=0.5, leave_out=(("Management", "Manufacturing"),)):
    """Pairs above min_overlap, grouped by area for the chart.

    Management + Manufacturing is left out on purpose: most Manufacturing
    postings also get the Management tag, so that pair is about tagging, not a
    real move between two kinds of work.
    """
    drop = {frozenset(p) for p in leave_out}
    out = pairs[pairs["overlap"] >= min_overlap].copy()
    keep = [frozenset((a, b)) not in drop for a, b in zip(out["function_a"], out["function_b"])]
    out = out[pd.Series(keep, index=out.index, dtype=bool)]
    area = {f: name for name, fs in AREAS.items() for f in fs}
    out["area"] = out["function_a"].map(area).fillna("Other")
    order = {name: i for i, name in enumerate([*AREAS, "Other"])}
    out["area_order"] = out["area"].map(order)
    out = out.sort_values(["area_order", "overlap"], ascending=[True, False])
    return out.drop(columns="area_order").reset_index(drop=True)


def plot_pairs(strong, note_extra="", path=None):
    short = {"Information Technology": "IT", "Accounting/Auditing": "Accounting"}
    labels = [f"{short.get(a, a)}  +  {short.get(b, b)}" for a, b in zip(strong["function_a"], strong["function_b"])]
    h = 2.9 + 0.42 * len(strong)
    fig, ax = plt.subplots(figsize=(12, h), facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.33, right=0.8, top=1 - 1.75 / h, bottom=0.95 / h)
    y = np.arange(len(strong))[::-1]
    colors = strong["area"].map(AREA_COLORS).fillna(GRAY)
    ax.barh(y, strong["overlap"] * 100, color=colors, height=0.68)
    for yy, s, n in zip(y, strong["overlap"], strong["shared_postings"]):
        ax.text(s * 100 + 1.2, yy, f"{s:.0%}", va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(112, yy, f"{n:,}", va="center", fontsize=11.5, color=INK)
    ax.text(112, len(strong) - 0.35, "shared postings", va="bottom", fontsize=11, color=QUIET, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(strong) - 0.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in AREA_COLORS.items()]
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(-0.4, 1.0), frameon=False, fontsize=12,
              handlelength=1.2)
    _titles(fig, h, "Job functions that often appear on the same posting",
            "% = share of the smaller function's postings that also carry the other function. Pairs of 50%+ shown.",
            SOURCE + " Functions with 1,000+ postings.\n" + note_extra)
    return _save(fig, path)


# ---------- 7. salary by industry and state ----------

def salary_grid(jobs, n_industries=5, n_states=6, min_n=50):
    """Median posted yearly salary for the biggest industry groups in the biggest states.

    Industries are ranked by all their postings, states by postings that show a
    usable full-time yearly salary. Cells under min_n postings become NaN.
    """
    paid = _valid_salary(jobs)  # full-time, yearly USD between 10k and 1M
    states = paid["state"].value_counts().head(n_states)
    sizes = jobs["industry"].explode().value_counts().drop(STAFFING, errors="ignore").head(n_industries)
    df = _explode(paid, ["industry", "state", "salary_annual"])
    df = df[df["industry"].isin(sizes.index) & df["state"].isin(states.index)]
    g = df.groupby(["industry", "state"])["salary_annual"]
    med = g.median().unstack().reindex(index=sizes.index, columns=states.index)
    n = g.size().unstack().reindex(index=sizes.index, columns=states.index).fillna(0).astype(int)
    return med.where(n >= min_n), n, states, sizes


def plot_salary_grid(med, states, sizes, path=None):
    cols = [f"{s}\n{states[s]:,}" for s in med.columns]
    return plot_grid(
        med / 1000, lambda x: f"${x:.0f}k",
        "Median yearly salary posted in job listings, by industry and state",
        f"Rows: the {len(med)} industries with the most postings. Columns: the {len(med.columns)} states with the most "
        "salary postings (number under each).\nDarker = higher salary. This is the salary shown in postings, not the "
        "legal minimum wage.",
        SOURCE + " Full-time postings with a yearly salary of \\$10,000 to \\$1,000,000.\nStaffing and Recruiting left "
        "out. Cells with under 50 postings marked too few.\n" + GROUP_NOTE,
        path=path, vmin=30, vmax=170, dark_from=110, col_labels=cols, right=sizes.to_numpy(),
        right_head="all postings", width=12)
