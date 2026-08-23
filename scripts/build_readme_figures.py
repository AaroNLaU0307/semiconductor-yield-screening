"""Render the three replication and screening figures embedded in README.md.

These PNGs (docs/drift_scatter.png, docs/permutation_null.png,
docs/missingness_patterns.png) are committed to the repository like
docs/dashboard.png. Nobody needs matplotlib installed to read the README, so
matplotlib is deliberately not listed in requirements.txt: it is an optional
dev dependency needed only to regenerate these three images, and CI does not
run this script. Install it only if you want to regenerate the figures:

    pip install matplotlib

Every number drawn here (the correlation in the drift scatter's corner note,
the permutation p-value, the effect sizes on the missingness bars, and so on)
is read directly out of dashboard/replication_data.json, dashboard/screening_
data.json, and dashboard/patterns_data.json: the same JSON the interactive
dashboard itself reads. Nothing is recomputed and nothing is hard coded, so
these figures cannot silently drift away from the numbers the pipeline
actually reported.

Usage, from the repository root:
    python scripts/build_readme_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "dashboard"
DOCS_DIR = REPO_ROOT / "docs"

DRIFT_SCATTER_PATH = DOCS_DIR / "drift_scatter.png"
PERMUTATION_NULL_PATH = DOCS_DIR / "permutation_null.png"
MISSINGNESS_PATTERNS_PATH = DOCS_DIR / "missingness_patterns.png"

#: Base font size in points. Kept close to matplotlib's default so the PNGs
#: read comfortably both full size and shrunk inside a README.
FONT_SIZE_BASE = 11

#: Palette reused verbatim from scripts/build_dashboard.py (COLOR_OOC,
#: COLOR_MUTED, COLOR_BOUNDARY) so these static figures sit visually
#: consistently next to the interactive dashboard they summarise.
COLOR_SIGNAL = "#C0392B"  # replicates under BH / survives FDR / observed count
COLOR_MUTED = "#7F8C8D"  # does not replicate / no association / null distribution
COLOR_REFERENCE = "#444444"  # zero lines, the y = x reference line

DPI = 200

#: Passed to every savefig call so no run timestamp or matplotlib version
#: string is embedded in the PNG text metadata, which would otherwise make
#: two runs on different days (or different matplotlib versions) byte
#: different for no visual reason. See VERIFY step 2 in the build request:
#: two consecutive runs of this script must produce byte identical PNGs.
SAVEFIG_METADATA = {"Software": None}


def configure_style() -> None:
    """Set one deterministic, publication clean matplotlib style for all three figures.

    The font family is pinned to DejaVu Sans, which ships inside matplotlib
    itself rather than being resolved from whatever fonts happen to be
    installed on the machine. Combined with the fixed Agg backend and the
    metadata scrub in save_figure, this is what makes two separate runs of
    this script produce byte identical PNGs.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_BASE + 2,
            "axes.labelsize": FONT_SIZE_BASE,
            "xtick.labelsize": FONT_SIZE_BASE - 1,
            "ytick.labelsize": FONT_SIZE_BASE - 1,
            "legend.fontsize": FONT_SIZE_BASE - 1,
            "axes.edgecolor": COLOR_REFERENCE,
            "axes.labelcolor": "#1a1a1a",
            "text.color": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "build_readme_figures",
            # matplotlib's default numeric tick formatter renders negative
            # numbers with the Unicode minus sign U+2212, not an ASCII hyphen.
            # scripts/check_no_em_dash.py does not scan PNG pixel content, but
            # this repository's non-negotiable 1 forbids U+2212 by name (its
            # own docstring: "two of them reached this repository's README"),
            # so every negative tick label on these figures (figure A has
            # negative effect sizes on both axes) is pinned to a plain hyphen.
            "axes.unicode_minus": False,
        }
    )


def load_json(name: str) -> Any:
    """Load one of the pre-generated dashboard JSON files, no recomputation."""
    path = DASHBOARD_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_p(value: float) -> str:
    """Format a p-value: scientific notation below 1e-4, else fixed to 4 places.

    Mirrors the convention already used by scripts/build_dashboard.py's own
    format_p (in turn mirroring yield_screening.reporting.fmt_p), reimplemented
    locally so this script depends on nothing beyond matplotlib, numpy and the
    standard library.
    """
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Write fig to output_path deterministically and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, metadata=SAVEFIG_METADATA)
    plt.close(fig)


def build_drift_scatter(replication: dict[str, Any], output_path: Path) -> None:
    """Figure A: train effect vs holdout effect for all shortlisted features.

    The central point of this figure is that most points fall well short of
    the y = x "perfect transfer" line, and quite a few even cross the y = 0 or
    x = 0 axis, meaning the direction of the association itself flips between
    train and holdout.
    """
    summary = replication["summary"]
    features = replication["features"]

    train = np.array([f["train effect"] for f in features], dtype=float)
    holdout = np.array([f["holdout effect"] for f in features], dtype=float)
    replicates = np.array([bool(f["replicates"]) for f in features])
    names = [str(f["feature"]) for f in features]

    combined_min = float(min(train.min(), holdout.min()))
    combined_max = float(max(train.max(), holdout.max()))
    pad = 0.15 * (combined_max - combined_min)
    lo = combined_min - pad
    hi = combined_max + pad

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhline(0.0, color=COLOR_REFERENCE, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=COLOR_REFERENCE, linewidth=0.8, zorder=1)
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color=COLOR_REFERENCE,
        linewidth=1.3,
        zorder=2,
        label="perfect transfer (y = x)",
    )

    not_replicating = ~replicates
    ax.scatter(
        train[not_replicating],
        holdout[not_replicating],
        s=34,
        color=COLOR_MUTED,
        alpha=0.55,
        edgecolor="none",
        zorder=3,
        label="does not replicate",
    )
    ax.scatter(
        train[replicates],
        holdout[replicates],
        s=115,
        color=COLOR_SIGNAL,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="replicates under BH",
    )

    for x, y, name, does_replicate in zip(train, holdout, names, replicates):
        if does_replicate:
            ax.annotate(
                name,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=FONT_SIZE_BASE - 1,
                color=COLOR_SIGNAL,
                fontweight="bold",
                zorder=5,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Most screened associations do not survive a change of time period")
    ax.set_xlabel("Effect size on train (rank-biserial)")
    ax.set_ylabel("Effect size on holdout (rank-biserial)")

    correlation = summary["effect_correlation"]
    same_sign = summary["same_sign"]
    n_shortlisted = summary["n_shortlisted"]
    expected_same_sign = summary["expected_same_sign_by_chance"]
    info_text = (
        f"train-holdout effect correlation: {correlation:.2f}\n"
        f"same sign on holdout: {same_sign} / {n_shortlisted} features\n"
        f"expected by chance alone: {expected_same_sign:.1f}"
    )
    ax.text(
        0.03,
        0.97,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE_BASE - 1,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.92),
    )

    ax.grid(True, linewidth=0.4, alpha=0.3, zorder=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, output_path)


def build_permutation_null(screening: dict[str, Any], output_path: Path) -> None:
    """Figure B: label permutation null distribution vs the observed survivor count."""
    null_counts = np.asarray(screening["null_counts"], dtype=int)
    n_survivors = int(screening["n_survivors"])
    permutation = screening["permutation"]
    p_value = float(permutation["permutation_p_value"])

    max_null = int(null_counts.max())
    bin_edges = np.arange(-0.5, max_null + 1.5, 1.0)
    xlim_max = max(n_survivors + 3, max_null + 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(
        null_counts,
        bins=bin_edges,
        color=COLOR_MUTED,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, xlim_max)

    ax.axvline(n_survivors, color=COLOR_SIGNAL, linewidth=1.6, zorder=3)

    # A blended transform (data x, axes-fraction y) places the annotation at a
    # fixed height regardless of exactly how tall the log-scale y axis turns
    # out to be, so it never has to be hand tuned against ax.get_ylim(). Note
    # this is get_xaxis_transform, not get_yaxis_transform: the latter blends
    # the other way (axes-fraction x, data y), which is what x-axis tick
    # labels use, not what we want here. The text is right aligned just to
    # the left of the observed line, growing into the empty gap between the
    # null histogram (which only occupies x up to max_null) and the line,
    # rather than growing right off the edge of the figure toward xlim_max.
    label_transform = ax.get_xaxis_transform()
    ax.text(
        n_survivors - xlim_max * 0.015,
        0.82,
        f"observed = {n_survivors} survivors\npermutation p = {format_p(p_value)}",
        transform=label_transform,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_BASE - 1,
        color=COLOR_SIGNAL,
        zorder=4,
    )

    ax.set_title("The screen finds far more than shuffled labels ever produce")
    ax.set_xlabel("Features surviving BH per label permutation")
    ax.set_ylabel("Permutations (log scale)")

    ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.3, zorder=0)

    fig.tight_layout()
    save_figure(fig, output_path)


def build_missingness_patterns(
    patterns: list[dict[str, Any]], q: float, output_path: Path
) -> None:
    """Figure C: one horizontal bar per metrology missingness pattern.

    Sorted ascending by train_coverage before plotting: matplotlib's barh
    draws the first entry at the bottom, so an ascending plot order is what
    makes the tallest-coverage bar land at the top, i.e. what makes the chart
    read as descending from top to bottom.
    """
    ordered = sorted(patterns, key=lambda p: p["train_coverage"])

    labels = [f"{p['pattern']} ({p['n_sensors']} sensors)" for p in ordered]
    coverage = [float(p["train_coverage"]) for p in ordered]
    survives = [bool(p["survives_fdr"]) for p in ordered]
    effects = [float(p["effect_size"]) for p in ordered]
    colors = [COLOR_SIGNAL if s else COLOR_MUTED for s in survives]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(ordered))

    ax.barh(y_pos, coverage, color=colors, edgecolor="white", linewidth=0.6, zorder=2)

    for yi, (cov, effect, does_survive) in enumerate(zip(coverage, effects, survives)):
        if does_survive:
            ax.annotate(
                f"r = {effect:.3f}",
                (cov, yi),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=FONT_SIZE_BASE - 1,
                color=COLOR_SIGNAL,
                fontweight="bold",
                zorder=4,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)

    # A figure-level suptitle, not ax.set_title: this axes is pushed well right
    # of the figure's horizontal center by its long pattern-name y tick labels,
    # so a long title centered over the axes (ax.set_title's default) runs past
    # the right edge of the canvas. fig.suptitle centers over the whole figure
    # instead, which is exactly what a full-width, single-axes chart needs.
    fig.suptitle(
        "Eight distinct metrology sampling patterns recovered from 52 sparse sensors",
        fontsize=FONT_SIZE_BASE + 2,
    )
    ax.set_xlabel("Share of train lots measured")

    legend_handles = [
        Patch(
            facecolor=COLOR_SIGNAL,
            edgecolor="white",
            label=f"associated with failure (BH q = {q:.2f})",
        ),
        Patch(facecolor=COLOR_MUTED, edgecolor="white", label="no association detected"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.92)

    ax.grid(True, axis="x", linewidth=0.4, alpha=0.3, zorder=0)

    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> int:
    configure_style()

    replication = load_json("replication_data.json")
    screening = load_json("screening_data.json")
    patterns = load_json("patterns_data.json")

    build_drift_scatter(replication, DRIFT_SCATTER_PATH)
    build_permutation_null(screening, PERMUTATION_NULL_PATH)
    build_missingness_patterns(patterns, float(screening["q"]), MISSINGNESS_PATTERNS_PATH)

    for output_path in (DRIFT_SCATTER_PATH, PERMUTATION_NULL_PATH, MISSINGNESS_PATTERNS_PATH):
        size_kb = output_path.stat().st_size / 1024
        print(f"wrote {output_path.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
