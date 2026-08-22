"""Render docs/dashboard.png, the dashboard preview embedded in README.md.

The image is built from the **same figure builders** the dashboard itself uses, so it
cannot drift away from what a reader sees when they open `dashboard/index.html`. It is
a rendering of the real panels, not a mock up.

This is the only part of the project that needs `kaleido`, and it is deliberately not
in `requirements.txt`: the generated PNG is committed, so nobody needs to reproduce it
to read the repository, and CI does not run this script. Install it only if you want to
regenerate the image:

    pip install kaleido

Usage, from the repository root:
    python scripts/build_readme_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_dashboard import (  # noqa: E402
    build_pr_figure,
    build_spc_figure,
    build_timeline_figure,
    load_json,
)

OUTPUT = REPO_ROOT / "docs" / "dashboard.png"
PANEL_WIDTH = 1200
SCALE = 2


def render_panel(figure, height: int, title: str) -> "Image.Image":  # noqa: F821
    """Rasterise one plotly figure at retina scale."""
    from PIL import Image
    import io

    # The SPC panel carries a plotly dropdown anchored near the top left. Adding a
    # title there without making room collides with it, so the title sits above the
    # plotting area and any menu is pushed below the title rather than across it.
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        margin={"t": 110, "b": 50, "l": 70, "r": 30},
    )
    menus = figure.layout.updatemenus
    if menus:
        for menu in menus:
            menu.update(y=1.02, yanchor="top", x=0.0, xanchor="left")
    payload = figure.to_image(
        format="png", width=PANEL_WIDTH, height=height, scale=SCALE
    )
    return Image.open(io.BytesIO(payload))


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("pillow is required: pip install pillow", file=sys.stderr)
        return 1

    timeline = load_json("timeline_data.json")
    spc = load_json("spc_data.json")
    model = load_json("model_data.json")

    # Panel 3 of the dashboard is the ranked detractor table, which is plain HTML
    # rather than a plotly figure, so it has no rasterisable form and is absent here.
    # The numbering below follows the dashboard so the two cannot be confused.
    panels = [
        render_panel(
            build_timeline_figure(timeline),
            360,
            "Panel 1. Weekly failure rate, with the chronological split marked",
        ),
        render_panel(
            build_spc_figure(spc),
            420,
            "Panel 2. I chart for a shortlisted sensor, out of control points flagged",
        ),
        render_panel(
            build_pr_figure(model),
            420,
            "Panel 4. Precision-recall on holdout, with the chosen operating point",
        ),
    ]

    gap = 16 * SCALE
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")

    offset = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, offset))
        offset += panel.height + gap

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({canvas.width}x{canvas.height}, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
