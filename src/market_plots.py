"""Charts for the market analysis.

Every chart follows the same rules so the slides look like one set:
- the title says what the chart shows (not the conclusion)
- the gray line under the title says how to read it
- only the items that matter get color; everything else is light gray
- few labels, large text, no heavy grid lines

Each function takes a summary table from market_analysis.py, draws one
figure, saves it as PNG when path is given, and returns the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# Use the first of these fonts that is installed (Helvetica Neue on a Mac).
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["Helvetica Neue", "Inter", "Arial", "DejaVu Sans"] if f in _INSTALLED), "sans-serif"
)

BLUE = "#2a78d6"
RED = "#d6453d"
INK = "#1d1d1b"
QUIET = "#6b6a65"
SUBTITLE = "#4a4945"
LIGHT = "#d9d8d2"
FAINT = "#f0efec"
GRAY_DOT = "#9a9891"
BLUES = LinearSegmentedColormap.from_list("blues", ["#e8f0fb", "#9cc3ef", "#2a78d6", "#123f78"])

WIDTH = 10  # inches, same for every chart

# Tile map: (column, row) for each state, roughly matching U.S. geography.
STATE_TILES = {
    "AK": (0, 0), "ME": (10, 0),
    "VT": (9, 1), "NH": (10, 1),
    "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2), "IL": (5, 2),
    "WI": (6, 2), "MI": (7, 2), "NY": (8, 2), "RI": (9, 2), "MA": (10, 2),
    "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3), "IN": (5, 3),
    "OH": (6, 3), "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
    "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4),
    "WV": (6, 4), "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
    "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5),
    "SC": (7, 5), "DC": (8, 5),
    "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
    "HI": (0, 7), "TX": (3, 7), "FL": (8, 7),
}


# ---------------------------------------------------------------------------
# Shared layout
# ---------------------------------------------------------------------------

def _empty(table, what: str) -> bool:
    """Print a note and return True when there is nothing to draw."""
    if table is None or len(table) == 0:
        print(f"Nothing to plot for '{what}': no group met the minimum size. Try a smaller min_n.")
        return True
    return False


def _new_figure(height: float, label_chars: int = 0, right: float = 0.9):
    """Figure with fixed room (in inches) for the header, footer and row labels."""
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    left = min(0.42, 0.04 + 0.0115 * label_chars) if label_chars else 0.06
    fig.subplots_adjust(left=left, right=right, top=1 - 1.25 / height, bottom=0.65 / height)
    return fig, ax


def _header_footer(fig, title: str, subtitle: str, note: str) -> None:
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.3 / height, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.68 / height, subtitle, fontsize=10.5, color=SUBTITLE, va="top")
    fig.text(0.02, 0.15 / height, note, fontsize=8.5, color=QUIET, va="bottom")


def _clean(ax) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, path) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor="white")


def _row_colors(n: int, top: int, bottom: int) -> list[str]:
    """Rows sorted low to high: the last top blue, the first bottom red, rest gray."""
    return [RED if i < bottom else BLUE if i >= n - top else LIGHT for i in range(n)]


def _style_row_labels(ax, colors: list[str]) -> None:
    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(INK if color != LIGHT else QUIET)
        label.set_fontweight("bold" if color != LIGHT else "normal")


# ---------------------------------------------------------------------------
# Lollipops
# ---------------------------------------------------------------------------

def plot_lollipop(values: pd.Series, title: str, subtitle: str, note: str,
                  top: int = 3, bottom: int = 0, average: float | None = None,
                  average_label: str = "all postings", value_format: str = "{:.0%}",
                  counts: pd.Series | None = None, sort: bool = True,
                  colors: list[str] | None = None, label_all: bool = False, path=None):
    """Dots on thin stems. The top highest are blue, the bottom lowest
    red, the rest gray; only colored rows get a value label (unless
    label_all). With sort=False the given order is kept (first row at
    the top). colors overrides the automatic colors row by row."""
    if colors is not None:
        colors = pd.Series(list(colors), index=values.index)
    values = values.dropna()   # a group with no data would leave an empty row
    if _empty(values, title):
        return None
    if colors is not None:
        colors = colors.reindex(values.index)
    if sort:
        values = values.sort_values()
    else:
        values = values.iloc[::-1]
    if counts is not None:
        counts = counts.reindex(values.index)
    n = len(values)
    if colors is None:
        colors = _row_colors(n, top, bottom)
    else:
        colors = list(colors.reindex(values.index))
    y = np.arange(n)
    labels_len = max(len(str(i)) for i in values.index)

    fig, ax = _new_figure(0.36 * n + 2.1, labels_len)
    ax.hlines(y, 0, values.to_numpy(), color=colors, linewidth=2)
    ax.scatter(values.to_numpy(), y, s=85, color=colors, zorder=3)
    if average is not None:
        ax.axvline(average, color=QUIET, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(average, n - 0.35, f"  {average_label} {value_format.format(average)}",
                fontsize=9.5, color=QUIET, va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(values.index, fontsize=10.5)
    _style_row_labels(ax, colors)
    for i, (v, color) in enumerate(zip(values, colors)):
        if color != LIGHT or label_all:
            text = value_format.format(v)
            if counts is not None:
                text += f"  (n = {int(counts.iloc[i]):,})"
            # fixed 9-point gap from the dot; white box keeps dashed lines from crossing the text
            ax.annotate(text, (v, i), xytext=(9, 0), textcoords="offset points", va="center",
                        fontsize=10, zorder=2, fontweight="bold" if color != LIGHT else "normal",
                        color=color if color != LIGHT else QUIET,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))

    ax.set_xlim(0, values.max() * 1.3)
    ax.set_ylim(-0.7, n + 0.4)
    ax.set_xticks([])
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_diverging_lollipop(values: pd.Series, title: str, subtitle: str, note: str,
                            counts: pd.Series | None = None, highlight: int = 3, path=None):
    """Signed values (0.05 = +5%) around a zero line. The highlight largest
    positive values are blue and the highlight most negative are red, with
    labels; the rest are gray."""
    if _empty(values, title):
        return None
    values = values.sort_values()
    n = len(values)
    colors = []
    for i, v in enumerate(values):
        if v > 0 and i >= n - highlight:
            colors.append(BLUE)
        elif v < 0 and i < highlight:
            colors.append(RED)
        else:
            colors.append(LIGHT)
    pct = values.to_numpy() * 100
    y = np.arange(n)
    span = max(abs(pct).max(), 1)
    labels_len = max(len(str(i)) for i in values.index)

    fig, ax = _new_figure(0.36 * n + 2.1, labels_len)
    ax.axvline(0, color=QUIET, linewidth=1)
    ax.hlines(y, 0, pct, color=colors, linewidth=2)
    ax.scatter(pct, y, s=85, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(values.index, fontsize=10.5)
    _style_row_labels(ax, colors)
    for i, (v, color) in enumerate(zip(pct, colors)):
        if color == LIGHT:
            continue
        text = f"{v:+.0f}%"
        if counts is not None:
            text += f"  (n = {int(counts.loc[values.index[i]]):,})"
        right = v >= 0
        ax.annotate(text, (v, i), xytext=(9 if right else -9, 0), textcoords="offset points",
                    va="center", ha="left" if right else "right", fontsize=10,
                    fontweight="bold", color=color)
    ax.set_xlim(-span * 1.7, span * 1.7)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([])
    ax.text(0, n - 0.1, "0", ha="center", fontsize=9, color=QUIET)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# Ranges and pairs
# ---------------------------------------------------------------------------

def plot_ranges(summary: pd.DataFrame, title: str, subtitle: str, note: str,
                top: int = 3, sort: bool = True, all_blue: bool = False, path=None):
    """Salary ranges: line from the 25th to the 75th percentile, dot at the
    median, in thousand USD. The top highest medians are blue (or every
    row with all_blue). With sort=False the given order is kept, first
    row at the top."""
    if _empty(summary, title):
        return None
    summary = summary.sort_values("median") if sort else summary.iloc[::-1]
    k = summary[["p25", "median", "p75"]] / 1000
    n = len(summary)
    colors = [BLUE] * n if all_blue else _row_colors(n, top, 0)
    y = np.arange(n)
    labels_len = max(len(str(i)) for i in summary.index)

    fig, ax = _new_figure(0.36 * n + 2.3, labels_len)
    ax.hlines(y, k["p25"], k["p75"], color=colors, linewidth=5, alpha=0.45)
    ax.scatter(k["median"], y, s=80, color=colors, zorder=3, edgecolor="white", linewidth=1.5)
    for i, (med, color) in enumerate(zip(k["median"], colors)):
        if color != LIGHT:
            ax.annotate(f"${med:,.0f}k  (n = {int(summary['n'].iloc[i]):,})", (k["p75"].iloc[i], i),
                        xytext=(9, 0), textcoords="offset points", va="center", fontsize=10,
                        fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index, fontsize=10.5)
    if not all_blue:
        _style_row_labels(ax, colors)

    low = int(k["p25"].min() // 25 * 25)   # start near the lowest salary, not at $0
    high = int(-(-k["p75"].max() // 25) * 25)
    ax.set_xlim(low, high + (high - low) * 0.35)
    ticks = list(range(low, high + 1, 25))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t}k" for t in ticks], fontsize=9, color=QUIET)
    for t in ticks:
        ax.axvline(t, color=FAINT, linewidth=1, zorder=0)
    ax.set_ylim(-0.7, n - 0.3)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_dumbbell(table: pd.DataFrame, left: str, right: str, left_label: str, right_label: str,
                  title: str, subtitle: str, note: str, path=None):
    """Two medians per row joined by a line (gray = left, blue = right),
    in thousand USD. Rows sorted by the percent gap."""
    if _empty(table, title):
        return None
    k = table[[left, right]] / 1000
    k = k.loc[(k[right] / k[left]).sort_values().index]
    n = len(k)
    y = np.arange(n)
    labels_len = max(len(str(i)) for i in k.index)

    fig, ax = _new_figure(0.42 * n + 2.4, labels_len)
    ax.hlines(y, k[left], k[right], color=LIGHT, linewidth=4)
    ax.scatter(k[left], y, s=85, color=GRAY_DOT, zorder=3, label=left_label)
    ax.scatter(k[right], y, s=85, color=BLUE, zorder=3, label=right_label)
    for i, (a, b) in enumerate(zip(k[left], k[right])):
        gap = b / a - 1
        ax.annotate(f"{gap:+.0%}", (max(a, b), i), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=10, fontweight="bold", color=BLUE if gap > 0 else RED)
    ax.set_yticks(y)
    ax.set_yticklabels(k.index, fontsize=10.5)
    low = int(k.to_numpy().min() // 10 * 10)
    high = int(-(-k.to_numpy().max() // 10) * 10)
    ticks = list(range(low, high + 1, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t}k" for t in ticks], fontsize=9, color=QUIET)
    for t in ticks:
        ax.axvline(t, color=FAINT, linewidth=1, zorder=0)
    ax.set_xlim(low - 2, high + (high - low) * 0.25)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def plot_bubble_matrix(share: pd.DataFrame, title: str, subtitle: str, note: str,
                       highlight: pd.DataFrame | None = None, highlight_color: str = BLUE,
                       highlight_label: str = "highlighted", row_counts: pd.Series | None = None,
                       path=None):
    """Rows by columns of shares (0 to 1) drawn as circles: bigger = higher.

    Cells where highlight is True are colored and labeled with their value;
    the rest are light gray. Empty or missing cells are left blank.
    """
    if _empty(share, title):
        return None
    rows, cols = share.shape
    values = share.to_numpy(dtype=float)
    marks = np.zeros_like(values, dtype=bool) if highlight is None else \
        highlight.reindex(index=share.index, columns=share.columns).fillna(False).to_numpy(dtype=bool)
    row_names = [f"{r}  (n = {int(row_counts[r]):,})" if row_counts is not None else str(r) for r in share.index]
    labels_len = max(len(r) for r in row_names)
    left = min(0.42, 0.04 + 0.0105 * labels_len)
    biggest = np.nanmax(values) if np.isfinite(values).any() else 1
    # Cell size in inches: the largest circle fills about one cell.
    cell_in = min((WIDTH * (0.97 - left)) / cols, 0.6)
    scale = (cell_in * 72 * 0.95) ** 2 / max(biggest, 1e-9)   # marker area (points^2) per unit share
    key_h = cell_in * 0.95 + 0.15                              # room for the biggest key circle
    top_in = 1.05 + key_h + 0.3                               # header + key + gap
    height = top_in + cell_in * rows + 1.7

    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.subplots_adjust(left=left, right=0.97, top=1 - top_in / height, bottom=1.6 / height)

    for i in range(rows):
        ax.axhline(i, color=FAINT, linewidth=0.8, zorder=1)
        for j in range(cols):
            v = values[i, j]
            if not np.isfinite(v) or v <= 0:
                continue
            on = marks[i, j]
            ax.scatter(j, i, s=v * scale, color=highlight_color if on else LIGHT,
                       edgecolor="white", linewidth=0.8, zorder=3)
            if on:
                inside = v * scale > 650
                ax.text(j, i if inside else i - 0.38, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", zorder=4,
                        color="white" if inside else highlight_color)

    ax.set_xticks(range(cols))
    ax.set_xticklabels(share.columns, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_names, fontsize=10)
    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(rows - 0.4, -0.7)
    _clean(ax)

    # Key with the same marker scale as the chart. The key axis is measured in
    # inches, so circles and labels can be placed side by side without overlap.
    key_w = WIDTH * (0.97 - left)
    key = fig.add_axes([left, 1 - (1.05 + key_h) / height, 0.97 - left, key_h / height])
    sizes = [round(biggest * f, 2) for f in (0.15, 0.5, 1.0)]
    items = [(s, LIGHT, f"{s:.0%}") for s in sizes]
    if highlight is not None:
        items.append((sizes[1], highlight_color, highlight_label))
    x = 0.0
    for s, color, text in items:
        radius = np.sqrt(s * scale) / 2 / 72          # marker diameter in points -> radius in inches
        key.scatter([x + radius], [0], s=s * scale, color=color, edgecolor="white", clip_on=False)
        key.text(x + 2 * radius + 0.08, 0, text, va="center", fontsize=9, color=QUIET)
        x += 2 * radius + 0.08 + 0.09 * len(text) + 0.35
    key.set_xlim(0, key_w)
    key.set_ylim(-key_h / 2, key_h / 2)
    key.axis("off")
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_stacked_mix(table: pd.DataFrame, title: str, subtitle: str, note: str, path=None,
                     first_gray: bool = True):
    """100% stacked bars, first row at the top. Columns are ordered categories
    from light to dark blue. With first_gray the first column (such as
    "None stated") is light gray."""
    if _empty(table, title):
        return None
    table = table.iloc[::-1]
    k = table.shape[1]
    if first_gray:
        shades = [FAINT] + [BLUES(x) for x in np.linspace(0.3, 1.0, k - 1)]
    else:
        shades = [BLUES(x) for x in np.linspace(0.2, 1.0, k)]
    labels_len = max(len(str(i)) for i in table.index)
    fig, ax = _new_figure(0.42 * len(table) + 2.6, labels_len, right=0.97)
    left = np.zeros(len(table))
    for position, (color, column) in enumerate(zip(shades, table.columns)):
        widths = table[column].to_numpy() * 100
        ax.barh(table.index.astype(str), widths, left=left, color=color, height=0.62,
                edgecolor="white", linewidth=1.2, label=column)
        dark = position >= (k + 1) // 2
        for y, (x0, w) in enumerate(zip(left, widths)):
            if w >= 7:
                ax.text(x0 + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if dark else INK)
        left += widths
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=10.5)
    ax.legend(ncol=k, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5,
              handlelength=1.2, columnspacing=1.2)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# Waffle, map, time
# ---------------------------------------------------------------------------

def plot_waffle_compare(rates: dict[str, float], counts: dict[str, int], title: str,
                        subtitle: str, note: str, unit: str = "states a salary", path=None):
    """One 10 x 10 grid per group: each square is 1 posting out of 100, blue
    squares are the ones that unit."""
    if _empty(rates, title):
        return None
    k = len(rates)
    fig, axes = plt.subplots(1, k, figsize=(WIDTH, 6.6))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.62, bottom=0.12, wspace=0.3)
    for ax, (label, rate) in zip(axes, rates.items()):
        filled = int(round(rate * 100))
        for i in range(100):
            row, col = divmod(i, 10)
            ax.add_patch(plt.Rectangle((col, 9 - row), 0.86, 0.86, edgecolor="none",
                                       facecolor=BLUE if i < filled else LIGHT))
        ax.set_xlim(-0.2, 10)
        ax.set_ylim(-0.2, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, 1.2, f"{rate:.0%}", transform=ax.transAxes, fontsize=34, fontweight="bold",
                color=BLUE, va="bottom")
        ax.text(0, 1.06, f"{label}  (n = {counts[label]:,})", transform=ax.transAxes,
                fontsize=11, color=INK, va="bottom")
    fig.text(0.02, 0.07, f"Each square = 1 posting out of 100. Blue = the posting {unit}.",
             fontsize=9.5, color=QUIET)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_state_tiles(values: pd.Series, title: str, subtitle: str, note: str,
                     value_format: str = "{:.0%}", highlight: set[str] | None = None,
                     highlight_label: str | None = None, path=None):
    """U.S. tile map: one square per state, darker blue = higher value. States
    without a value are light gray. highlight draws a black outline around
    some states."""
    if _empty(values, title):
        return None
    fig, ax = plt.subplots(figsize=(WIDTH, 7.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=1 - 1.25 / 7.4, bottom=0.6 / 7.4)
    finite = values.dropna()
    low, high = finite.min(), finite.max()
    span = high - low if high > low else 1

    for state, (col, row) in STATE_TILES.items():
        value = values.get(state, np.nan)
        has_value = pd.notna(value)
        level = (value - low) / span if has_value else 0
        outlined = highlight is not None and state in highlight
        ax.add_patch(plt.Rectangle((col, -row), 0.9, 0.9,
                                   facecolor=BLUES(0.08 + 0.92 * level) if has_value else FAINT,
                                   edgecolor=INK if outlined else "white",
                                   linewidth=2.4 if outlined else 1))
        dark = has_value and level > 0.5
        ax.text(col + 0.45, -row + 0.6, state, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white" if dark else INK)
        if has_value:
            ax.text(col + 0.45, -row + 0.28, value_format.format(value), ha="center",
                    va="center", fontsize=8.5, color="white" if dark else INK)

    # Keys go in the empty band between Alaska (row 0) and Vermont (row 1, col 9).
    ax.text(1.3, -0.25, "lighter = lower, darker = higher", fontsize=9, color=QUIET, va="center")
    for x, (label, level) in enumerate([(value_format.format(low), 0.08), (value_format.format(high), 1.0)]):
        ax.add_patch(plt.Rectangle((1.3 + x * 1.4, -0.85), 0.32, 0.32, facecolor=BLUES(level), edgecolor="none"))
        ax.text(1.72 + x * 1.4, -0.69, label, va="center", fontsize=9, color=QUIET)
    if highlight_label:
        ax.add_patch(plt.Rectangle((4.3, -0.85), 0.32, 0.32, facecolor="white", edgecolor=INK, linewidth=2.2))
        ax.text(4.72, -0.69, highlight_label, va="center", fontsize=9, color=QUIET)

    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-7.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_over_time(counts: pd.Series, title: str, subtitle: str, note: str, path=None):
    """Postings per period as a filled line."""
    if _empty(counts, title):
        return None
    x = counts.index.to_timestamp() if hasattr(counts.index, "to_timestamp") else counts.index
    fig, ax = _new_figure(4.2)
    fig.subplots_adjust(left=0.08)
    ax.fill_between(x, counts.to_numpy(), color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, counts.to_numpy(), color=BLUE, linewidth=2)
    peak = counts.idxmax()
    peak_x = peak.to_timestamp() if hasattr(peak, "to_timestamp") else peak
    ax.annotate(f"busiest: {counts.max():,}", (peak_x, counts.max()), xytext=(8, 4),
                textcoords="offset points", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.tick_params(labelsize=9, colors=QUIET)
    ax.grid(axis="y", color=FAINT, linewidth=1)
    ax.set_axisbelow(True)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig


def plot_paired_dots(table: pd.DataFrame, columns: list[str], labels: list[str], title: str,
                     subtitle: str, note: str, value_format: str = "{:.1%}",
                     counts: pd.Series | None = None, path=None):
    """Two values per row as two dots on one line (blue and dark blue).

    Rows keep the given order (first row at the top). Used for the share of
    postings that count internships and the share that ask for research.
    """
    if _empty(table, title):
        return None
    t = table[columns].iloc[::-1]
    names = [f"{r}  (n = {int(counts[r]):,})" if counts is not None else str(r) for r in t.index]
    fig, ax = _new_figure(0.42 * len(t) + 2.4, max(len(n) for n in names))
    y = np.arange(len(t))
    colors = [BLUE, "#123f78"]
    for i in y:
        ax.hlines(i, 0, t.iloc[i].max(), color=FAINT, linewidth=2, zorder=1)
    for col, label, color in zip(columns, labels, colors):
        ax.scatter(t[col], y, s=80, color=color, zorder=3, label=label, edgecolor="white", linewidth=1)
    for i in y:
        a, b = t.iloc[i][columns[0]], t.iloc[i][columns[1]]
        ax.annotate(value_format.format(a), (a, i), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=colors[0])
        ax.annotate(value_format.format(b), (b, i), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=colors[1])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xticks([])
    ax.set_xlim(-0.002, t.to_numpy().max() * 1.15)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5)
    _clean(ax)
    _header_footer(fig, title, subtitle, note)
    _save(fig, path)
    return fig
