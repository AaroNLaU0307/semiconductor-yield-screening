"""Build the self contained HTML dashboard for the yield screening study.

This script implements the dashboard deliverable of PLAN.md phase 5. It reads four
pre-generated JSON files under `dashboard/` (`timeline_data.json`, `spc_data.json`,
`screening_data.json`, `model_data.json`) and writes `dashboard/index.html`. Nothing
is recomputed here: every number shown is read directly from those files.

Self contained requirement. A reviewer must be able to open `dashboard/index.html`
directly from disk, with no server and no internet connection, and have every chart
render correctly. That means the plotly.js library itself has to be embedded in the
file rather than loaded from a CDN. This script does that by calling
`plotly.io.to_html` with `include_plotlyjs=True` on the first chart, which inlines the
full plotly.js source once, and `include_plotlyjs=False` on every later chart, so the
library is not duplicated. The resulting file is therefore large (plotly.js alone is
several megabytes) but requires nothing external at view time.

Determinism. Running this script twice must produce byte identical output: no
timestamps, no "generated on" text, and no ordering that depends on dict or set
iteration order beyond what the source JSON already fixes. Every chart that Plotly
would otherwise tag with a random element id is given an explicit, fixed id.

Usage, from the repository root:
    python scripts/build_dashboard.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "dashboard"
OUTPUT_PATH = DASHBOARD_DIR / "index.html"

# Every dropdown entry in the SPC panel contributes this many precomputed traces
# (train, holdout, out of control overlay, centre line, UCL line, LCL line), all but
# one of which stay hidden until the reader selects that sensor.
TRACES_PER_SENSOR = 6

PLOT_CONFIG: dict[str, Any] = {"displaylogo": False, "responsive": True}

TIMELINE_HEIGHT = 460
SPC_HEIGHT = 580
PR_HEIGHT = 480

COLOR_TRAIN = "#2E6E9E"
COLOR_HOLDOUT = "#D08A2B"
COLOR_OOC = "#C0392B"
COLOR_LIMIT = "#A93226"
COLOR_CENTRE = "#2C2C2C"
COLOR_MUTED = "#7F8C8D"
COLOR_BOUNDARY = "#444444"

# Caption text is reproduced verbatim, character for character, as specified.
CAPTION_TIMELINE = "Weekly failure rate, with the chronological split marked. Conclude: the failure rate is not stable, falling from about 22 percent in July to under 3 percent in September, which is why the split is chronological and why associations learned before the boundary need not hold after it."

# The drift clause is counted from spc_data.json by spc_caption(), not asserted.
CAPTION_SPC = "I chart for one shortlisted sensor. Control limits are computed on the train partition and applied unchanged to holdout. Red points break at least one applied Nelson rule. Conclude: out of control points are common on both sides of the boundary and {drift}, but they do not coincide with failures, see reports/spc.md."

CAPTION_TABLE = "The features that cleared Benjamini-Hochberg at q = 0.10, ranked by effect size. Conclude: several sensors are associated with failure during the train period, and the permutation null rules out chance as the explanation. These are associations, not causes, and nothing here shows that changing a sensor would change yield."

CAPTION_PR = "Precision-recall on the holdout for the declared primary model. The dashed line is no skill, at the holdout failure rate. Conclude: the curve sits essentially on the no skill line, so the model does not usefully rank held out lots, and the marked operating point catches very few failures."

HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semiconductor yield detractor screening</title>
<style>
:root {
  --fg: #1a1a1a;
  --muted: #5a5a5a;
  --border: #dcdcdc;
  --accent: #2e6e9e;
  --bg: #ffffff;
  --bg-alt: #f6f7f9;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--fg);
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  padding: 32px 16px 64px;
}
.page {
  max-width: 1100px;
  margin: 0 auto;
}
h1 {
  font-size: 1.9rem;
  margin: 0 0 8px;
}
h2 {
  font-size: 1.3rem;
  margin-top: 44px;
  margin-bottom: 12px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 14px;
  margin: 20px 0 8px;
  padding: 18px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.stat-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent);
}
.stat-label {
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 2px;
}
.panel { margin-bottom: 8px; }
.caption {
  margin: 10px 0 8px;
  padding: 10px 16px;
  border-left: 4px solid var(--accent);
  background: var(--bg-alt);
  color: var(--muted);
  font-size: 0.92rem;
}
.table-wrap {
  max-height: 480px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
thead th {
  position: sticky;
  top: 0;
  background: #eef1f5;
  text-align: left;
  padding: 8px 10px;
  border-bottom: 2px solid var(--border);
}
tbody td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody tr:nth-child(even) { background: #fafbfc; }
footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.85rem;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--bg-alt);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 0.9em;
}
</style>
</head>
<body>
<div class="page">
"""

HTML_FOOT = """
<footer>
<p>Generated by <code>scripts/build_dashboard.py</code>. Full detail is in <code>reports/</code>.</p>
</footer>
</div>
</body>
</html>
"""


def load_json(name: str) -> dict[str, Any]:
    """Load one of the four pre-generated dashboard JSON files, no recomputation."""
    path = DASHBOARD_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_p(value: float) -> str:
    """Format a p-value: scientific notation below 1e-4, else fixed to 4 places.

    Mirrors the convention already used by `yield_screening.reporting.fmt_p`
    elsewhere in this project. Reimplemented locally, rather than imported, so this
    script depends on nothing beyond plotly and the standard library.
    """
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt4(value: float) -> str:
    """Fixed 4 decimal place formatting for effect sizes, AUC and PR-AUC values."""
    return f"{value:.4f}"


def find_primary_metric(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the declared primary model's metric row.

    The primary model is identified the same way `scripts/run_pipeline.py` identifies
    it: by name prefix, not by picking whichever model scored best. DECISIONS.md D9
    names HistGradientBoosting as primary before any holdout record is read.
    """
    for metric in metrics:
        if str(metric["name"]).startswith("HistGradient"):
            return metric
    raise ValueError("model_data.json has no metric whose name starts with 'HistGradient'")


def precision_axis_max(precision: list[float], recall: list[float], no_skill: float) -> float:
    """A readable upper bound for the precision axis.

    scikit-learn's `precision_recall_curve` appends a final (recall=0, precision=1)
    point by convention. That point is not a real operating position, and using it to
    scale the axis would compress the entire informative part of the curve, which
    here tops out under 0.08, into an unreadable sliver at the bottom. The bound is
    instead set from the largest precision reached anywhere recall is actually
    positive, with headroom above both that value and the no-skill line.
    """
    achieved = [p for p, r in zip(precision, recall) if r > 0]
    reference = max(achieved + [no_skill]) if achieved else no_skill
    padded = reference * 1.8
    step = 0.05
    ceiling = math.ceil(padded / step) * step
    return min(1.0, max(ceiling, 0.1))


def render_figure(fig: go.Figure, div_id: str, include_js: bool, height: int) -> str:
    """Render one figure to an embeddable div, with a fixed id for determinism."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_js,
        config=PLOT_CONFIG,
        div_id=div_id,
        default_width="100%",
        default_height=height,
    )


# ---------------------------------------------------------------------------
# Panel 1: weekly failure rate over time
# ---------------------------------------------------------------------------


def build_timeline_figure(timeline: dict[str, Any]) -> go.Figure:
    week_start = timeline["week_start"]
    fail_rate = timeline["fail_rate"]
    records = timeline["records"]
    boundary_iso = str(timeline["boundary"]).replace(" ", "T")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=week_start,
            y=fail_rate,
            name="Weekly failure rate",
            marker_color=COLOR_TRAIN,
            customdata=records,
            hovertemplate=(
                "Week of %{x|%Y-%m-%d}<br>Failure rate: %{y:.1%}"
                "<br>Records this week: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=week_start,
            y=records,
            name="Weekly records",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLOR_MUTED, width=1.5, dash="dot"),
            marker=dict(size=5, color=COLOR_MUTED),
            hovertemplate="Week of %{x|%Y-%m-%d}<br>Records: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=TIMELINE_HEIGHT,
        margin=dict(t=70, r=70, b=50, l=70),
        xaxis=dict(title="Week starting", type="date"),
        yaxis=dict(title="Weekly failure rate", tickformat=".0%", rangemode="tozero"),
        yaxis2=dict(
            title="Weekly records",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        legend=dict(orientation="h", x=1.0, y=1.04, xanchor="right", yanchor="bottom"),
    )
    fig.add_vline(
        x=boundary_iso,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_BOUNDARY,
        annotation_text="train / holdout boundary",
        annotation_position="top",
        annotation_font_size=12,
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 2: SPC individuals chart, one sensor at a time via a dropdown
# ---------------------------------------------------------------------------


def build_spc_figure(spc: dict[str, Any]) -> go.Figure:
    sensors: list[str] = spc["sensors"]
    charts: dict[str, Any] = spc["charts"]

    first_chart = charts[sensors[0]]
    n_train = len(first_chart["train_values"])
    n_holdout = len(first_chart["holdout_values"])
    n_total = n_train + n_holdout

    fig = go.Figure()
    buttons = []

    for i, sensor in enumerate(sensors):
        chart = charts[sensor]
        limits = chart["limits"]
        train_values = chart["train_values"]
        holdout_values = chart["holdout_values"]
        train_ooc = chart["train_ooc"]
        holdout_ooc = chart["holdout_ooc"]

        if len(train_values) != n_train or len(holdout_values) != n_holdout:
            raise ValueError(f"{sensor} has a different train/holdout split length than {sensors[0]}")

        visible = i == 0
        train_x = list(range(n_train))
        holdout_x = list(range(n_train, n_total))

        ooc_x = [idx for idx, flagged in enumerate(train_ooc) if flagged]
        ooc_y = [train_values[idx] for idx in ooc_x]
        holdout_ooc_idx = [idx for idx, flagged in enumerate(holdout_ooc) if flagged]
        ooc_x = ooc_x + [n_train + idx for idx in holdout_ooc_idx]
        ooc_y = ooc_y + [holdout_values[idx] for idx in holdout_ooc_idx]

        fig.add_trace(
            go.Scatter(
                x=train_x,
                y=train_values,
                mode="lines+markers",
                name="Train",
                marker=dict(size=3, color=COLOR_TRAIN),
                line=dict(width=1, color=COLOR_TRAIN),
                visible=visible,
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=holdout_x,
                y=holdout_values,
                mode="lines+markers",
                name="Holdout",
                marker=dict(size=3, color=COLOR_HOLDOUT),
                line=dict(width=1, color=COLOR_HOLDOUT),
                visible=visible,
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ooc_x,
                y=ooc_y,
                mode="markers",
                name="Out of control",
                marker=dict(size=7, color=COLOR_OOC, line=dict(width=1, color="#7A1F1F")),
                visible=visible,
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, n_total - 1],
                y=[limits["centre"], limits["centre"]],
                mode="lines",
                name="Centre",
                line=dict(color=COLOR_CENTRE, width=1.5),
                visible=visible,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, n_total - 1],
                y=[limits["ucl"], limits["ucl"]],
                mode="lines",
                name="UCL",
                line=dict(color=COLOR_LIMIT, width=1.5, dash="dash"),
                visible=visible,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, n_total - 1],
                y=[limits["lcl"], limits["lcl"]],
                mode="lines",
                name="LCL",
                line=dict(color=COLOR_LIMIT, width=1.5, dash="dash"),
                visible=visible,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        mask = [False] * (len(sensors) * TRACES_PER_SENSOR)
        for offset in range(TRACES_PER_SENSOR):
            mask[i * TRACES_PER_SENSOR + offset] = True
        buttons.append(
            dict(
                label=sensor,
                method="update",
                args=[{"visible": mask}, {"title.text": f"I chart, {sensor}"}],
            )
        )

    fig.update_layout(
        title=dict(text=f"I chart, {sensors[0]}"),
        height=SPC_HEIGHT,
        margin=dict(t=130, r=110, b=50, l=70),
        xaxis=dict(title="Record index (chronological, train then holdout)"),
        yaxis=dict(title="Sensor value"),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                direction="down",
                showactive=True,
                active=0,
                x=0.0,
                xanchor="left",
                y=1.06,
                yanchor="bottom",
            )
        ],
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
    )
    fig.add_vline(
        x=n_train,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_BOUNDARY,
        annotation_text="train / holdout boundary",
        annotation_position="top",
        annotation_font_size=12,
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 3: ranked detractor table (plain HTML, not a Plotly table)
# ---------------------------------------------------------------------------


def build_table_html(screening: dict[str, Any]) -> str:
    shortlist = screening["shortlist"]
    header = (
        "<thead><tr>"
        '<th scope="col">feature</th>'
        '<th scope="col">status</th>'
        '<th scope="col">effect size</th>'
        '<th scope="col">AUC fail over pass</th>'
        '<th scope="col">raw p</th>'
        '<th scope="col">BH adjusted p</th>'
        "</tr></thead>"
    )
    rows = []
    for row in shortlist:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['feature']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{fmt4(row['effect_size_rank_biserial'])}</td>"
            f"<td>{fmt4(row['auc_fail_over_pass'])}</td>"
            f"<td>{format_p(row['p_raw'])}</td>"
            f"<td>{format_p(row['p_bh_adjusted'])}</td>"
            "</tr>"
        )
    body = "<tbody>" + "".join(rows) + "</tbody>"
    return f'<div class="table-wrap"><table>{header}{body}</table></div>'


# ---------------------------------------------------------------------------
# Panel 4: precision-recall on holdout
# ---------------------------------------------------------------------------


def build_pr_figure(model: dict[str, Any]) -> go.Figure:
    precision = model["pr_curve"]["precision"]
    recall = model["pr_curve"]["recall"]
    no_skill = model["no_skill"]
    operating_point = model["operating_point"]
    threshold = model["operating_threshold"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-recall (holdout)",
            line=dict(color=COLOR_TRAIN, width=2, shape="hv"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[operating_point["recall"]],
            y=[operating_point["precision"]],
            mode="markers",
            name="Operating point",
            marker=dict(size=16, color=COLOR_OOC, symbol="diamond", line=dict(width=1, color="#7A1F1F")),
        )
    )
    y_max = precision_axis_max(precision, recall, no_skill)
    fig.update_layout(
        height=PR_HEIGHT,
        margin=dict(t=60, r=40, b=60, l=70),
        xaxis=dict(title="Recall (holdout)", range=[0, 1]),
        yaxis=dict(title="Precision (holdout)", range=[0, y_max]),
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
    )
    fig.add_hline(
        y=no_skill,
        line_dash="dash",
        line_color=COLOR_BOUNDARY,
        annotation_text="no skill",
        annotation_position="top left",
        annotation_font_size=12,
    )
    fig.add_annotation(
        x=operating_point["recall"],
        y=operating_point["precision"],
        text=f"operating point, threshold = {fmt4(threshold)}",
        showarrow=True,
        arrowhead=2,
        ax=90,
        ay=-50,
        font=dict(size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Summary block and page assembly
# ---------------------------------------------------------------------------


def build_summary_html(screening: dict[str, Any], model: dict[str, Any]) -> str:
    """A short summary strip, every value read from the JSON, none hard coded."""
    n_tested = screening["n_tested"]
    n_survivors = screening["n_survivors"]
    permutation_p = screening["permutation"]["permutation_p_value"]

    primary = find_primary_metric(model["metrics"])
    pr_auc = primary["pr_auc"]
    ci_low, ci_high = primary["pr_auc_ci"]
    no_skill = model["no_skill"]

    stats = [
        (str(n_tested), "features tested"),
        (str(n_survivors), "features surviving FDR control"),
        (format_p(permutation_p), "permutation p value"),
        (
            fmt4(pr_auc),
            f"holdout PR-AUC (95 percent interval {fmt4(ci_low)} to {fmt4(ci_high)})",
        ),
        (fmt4(no_skill), "no skill floor (holdout failure rate)"),
    ]
    cards = "\n".join(
        f'<div class="stat"><div class="stat-value">{html.escape(value)}</div>'
        f'<div class="stat-label">{html.escape(label)}</div></div>'
        for value, label in stats
    )
    return f'<div class="summary">\n{cards}\n</div>'


def spc_caption(spc: dict[str, Any]) -> str:
    """CAPTION_SPC with its drift clause counted from the chart data.

    The holdout out of control rate is higher than the train rate for most charted
    sensors but not necessarily all, so the clause states the count it can support.
    """
    sensors: list[str] = spc["sensors"]
    n_higher = 0
    for sensor in sensors:
        chart = spc["charts"][sensor]
        train_rate = sum(chart["train_ooc"]) / len(chart["train_ooc"])
        holdout_rate = sum(chart["holdout_ooc"]) / len(chart["holdout_ooc"])
        n_higher += holdout_rate > train_rate
    if n_higher == len(sensors):
        drift = "more common after it for every charted sensor, which is process drift"
    elif n_higher * 2 > len(sensors):
        drift = (
            f"more common after it for {n_higher} of the {len(sensors)} charted sensors, "
            "which is process drift"
        )
    else:
        drift = (
            f"more common after it for only {n_higher} of the {len(sensors)} charted "
            "sensors, so they show no consistent drift"
        )
    return CAPTION_SPC.format(drift=drift)


def render_panel(title: str, body: str, caption: str) -> str:
    return (
        '<section class="panel">\n'
        f"<h2>{html.escape(title)}</h2>\n"
        f"{body}\n"
        f'<p class="caption">{html.escape(caption)}</p>\n'
        "</section>"
    )


def build_page(
    timeline: dict[str, Any],
    spc: dict[str, Any],
    screening: dict[str, Any],
    model: dict[str, Any],
) -> str:
    timeline_fig = build_timeline_figure(timeline)
    spc_fig = build_spc_figure(spc)
    pr_fig = build_pr_figure(model)

    timeline_div = render_figure(timeline_fig, "panel-timeline", True, TIMELINE_HEIGHT)
    spc_div = render_figure(spc_fig, "panel-spc", False, SPC_HEIGHT)
    pr_div = render_figure(pr_fig, "panel-pr", False, PR_HEIGHT)

    parts = [
        "<h1>Semiconductor yield detractor screening</h1>",
        build_summary_html(screening, model),
        render_panel("Weekly failure rate over time", timeline_div, CAPTION_TIMELINE),
        render_panel("SPC individuals chart by sensor", spc_div, spc_caption(spc)),
        render_panel("Ranked detractors surviving FDR control", build_table_html(screening), CAPTION_TABLE),
        render_panel("Precision-recall on holdout", pr_div, CAPTION_PR),
    ]
    return HTML_HEAD + "\n".join(parts) + HTML_FOOT


def main() -> int:
    timeline = load_json("timeline_data.json")
    spc = load_json("spc_data.json")
    screening = load_json("screening_data.json")
    model = load_json("model_data.json")

    page = build_page(timeline, spc, screening, model)
    OUTPUT_PATH.write_text(page, encoding="utf-8", newline="\n")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"wrote {OUTPUT_PATH} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
