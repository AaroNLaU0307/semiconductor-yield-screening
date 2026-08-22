"""Shared helpers for emitting generated markdown reports.

Reports under `reports/` are generated, never hand edited. Anything a reader might
want to check should be produced by code that ran, so that the report cannot drift
away from the numbers it describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def md_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub flavoured markdown table.

    Formatting is decided per column from the column dtype, not per value. Iterating
    rows would collapse a mixed frame to one dtype and print integer counts as
    "26.0000", which is both ugly and quietly misleading about what was measured.
    """
    formatted: dict[str, list[str]] = {}
    for name in frame.columns:
        column = frame[name]
        is_float = pd.api.types.is_float_dtype(column)
        cells = []
        for value in column:
            if pd.isna(value):
                cells.append("n/a")
            elif is_float and not isinstance(value, bool):
                cells.append(floatfmt.format(value))
            elif isinstance(value, (float, np.floating)) and not isinstance(value, bool):
                cells.append(floatfmt.format(value))
            else:
                cells.append(str(value))
        formatted[str(name)] = cells

    header = "| " + " | ".join(formatted) + " |"
    rule = "| " + " | ".join("---" for _ in formatted) + " |"
    lines = [header, rule]
    for i in range(len(frame)):
        lines.append("| " + " | ".join(formatted[c][i] for c in formatted) + " |")
    return "\n".join(lines)


def kv_table(pairs: list[tuple[str, object]]) -> str:
    """Render a list of (property, value) pairs as a two column markdown table."""
    return md_table(pd.DataFrame([{"property": k, "value": v} for k, v in pairs]))


def fmt_p(value: float) -> str:
    """Format a p-value without pretending to precision it does not have."""
    if pd.isna(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def histogram_lines(counts: np.ndarray, max_bars: int = 12, width: int = 40) -> str:
    """A small text histogram, so the null distribution is visible in the markdown."""
    counts = np.asarray(counts)
    lo, hi = int(counts.min()), int(counts.max())
    if hi == lo:
        return f"all {counts.size} permutations produced exactly {lo} survivors"

    edges = np.unique(np.linspace(lo, hi + 1, min(max_bars, hi - lo + 2)).astype(int))
    lines = ["| survivors | permutations | |", "| --- | --- | --- |"]
    total = counts.size
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        in_bin = int(np.sum((counts >= left) & (counts < right)))
        label = str(left) if right == left + 1 else f"{left} to {right - 1}"
        bar = "#" * int(round(width * in_bin / total))
        lines.append(f"| {label} | {in_bin} | {bar} |")
    return "\n".join(lines)
