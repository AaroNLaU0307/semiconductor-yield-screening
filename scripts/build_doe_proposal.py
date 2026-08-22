"""Emit reports/doe_proposal.md: a designed experiment that was NOT executed.

This is a proposal, not a result. The SECOM data is observational, collected from a
running process with no randomised intervention, so nothing in this repository can
establish that a sensor causes yield loss. The only way to get a causal answer is to
set the factors deliberately and randomise, which is what this document specifies and
what this project did not do.

The run counts below are computed, not asserted. The point of the calculation is that
it is expensive: it tells a reader what the observational shortcut is actually
standing in for.

Usage, from the repository root:
    python scripts/build_doe_proposal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yield_screening.reporting import kv_table, md_table  # noqa: E402

REPORT = REPO_ROOT / "reports" / "doe_proposal.md"
SCREEN_TABLE = REPO_ROOT / "reports" / "screening_table.csv"

N_FACTORS = 3
ALPHA = 0.05
POWER = 0.80
BASELINE_FAIL_RATE = 0.0664


def per_arm_n(p_control: float, p_treatment: float, alpha: float, power: float) -> int:
    """Sample size per arm for a two sided two proportion z test."""
    if p_control == p_treatment:
        return int(np.inf)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled = (p_control + p_treatment) / 2
    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_beta
        * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))
    ) ** 2
    return int(np.ceil(numerator / (p_control - p_treatment) ** 2))


def power_table(baseline: float) -> pd.DataFrame:
    """Runs required to detect a range of effect sizes on the failure rate."""
    rows = []
    for label, treatment in (
        ("halved, 6.6 to 3.3 percent", baseline / 2),
        ("cut by a third, 6.6 to 4.4 percent", baseline * 2 / 3),
        ("cut by a quarter, 6.6 to 5.0 percent", baseline * 0.75),
        ("doubled, 6.6 to 13.3 percent", baseline * 2),
        ("raised by half, 6.6 to 10.0 percent", baseline * 1.5),
    ):
        per_arm = per_arm_n(baseline, treatment, ALPHA, POWER)
        # In a two level factorial each main effect is estimated from half the runs
        # on each side, so the total run count is twice the per arm requirement.
        rows.append(
            {
                "effect to detect": label,
                "treatment fail rate": round(float(treatment), 4),
                "lots per level": per_arm,
                "total lots in the design": per_arm * 2,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if not SCREEN_TABLE.exists():
        raise SystemExit(
            "reports/screening_table.csv is missing, run scripts/run_pipeline.py first"
        )
    screen = pd.read_csv(SCREEN_TABLE)
    survivors = screen[screen["survives_fdr"]].copy()
    sensors = survivors[~survivors["feature"].str.contains("__notmeasured")]
    top = sensors.sort_values("abs_effect_size", ascending=False).head(N_FACTORS)

    factor_table = pd.DataFrame(
        {
            "factor": [f"F{i + 1}" for i in range(len(top))],
            "sensor": top["feature"].to_list(),
            "screen effect size": top["effect_size_rank_biserial"].round(4).to_list(),
            "BH adjusted p": top["p_bh_adjusted"].to_list(),
            "low level": "train 25th percentile",
            "high level": "train 75th percentile",
        }
    )

    runs = power_table(BASELINE_FAIL_RATE)
    headline = runs.iloc[0]

    parts = [
        "# Designed experiment: a proposal that was not executed\n",
        "**Status: proposal only.** Nothing in this document was run. It is included "
        "because the rest of this repository can only measure association, and it is "
        "more useful to state precisely what would settle the question than to imply "
        "that a ranking of associations has already settled it.\n",
        "## 1. Why an experiment is required at all\n",
        "Every result in this repository comes from observational records of a "
        "running process. Sensors move together, the process drifts over the "
        "collection window, and no factor was ever set deliberately. Under those "
        "conditions a strong association is consistent with several explanations "
        "that no amount of further analysis can separate: the sensor matters; the "
        "sensor proxies something else that matters; or both move with a third thing "
        "such as tool age or product mix. Randomised assignment is what breaks that "
        "tie, because it makes the factor independent of everything not measured.\n",
        "## 2. Factors\n",
        "The three shortlisted sensor values with the largest absolute effect size "
        "from the screen. Missingness indicators are excluded, since they describe "
        "metrology routing rather than a settable process parameter.\n",
        md_table(factor_table),
        "\nLevels are set at the train 25th and 75th percentiles rather than at "
        "extremes. Extreme levels would raise the power of the design but would run "
        "the process outside its normal operating window, which risks scrapping "
        "material and answers a question about a process nobody runs.\n",
        "\n## 3. Design\n",
        kv_table(
            [
                ("design", f"2^{N_FACTORS} full factorial"),
                ("factors", N_FACTORS),
                ("levels per factor", 2),
                ("distinct corners", 2**N_FACTORS),
                ("randomisation", "run order randomised within each day"),
                (
                    "blocking",
                    "block by day, since the drift measured in DECISIONS.md D0 is "
                    "large enough to swamp the effect being tested",
                ),
                ("response", "lot pass or fail, binary"),
                ("baseline failure rate", BASELINE_FAIL_RATE),
                ("significance level", ALPHA),
                ("target power", POWER),
            ]
        ),
        "\nA full factorial rather than a fractional design: with only three factors "
        "the full design costs eight corners rather than four, and it buys clean "
        "two factor interactions. Interactions are the plausible mechanism here, "
        "since these sensors sit on correlated process steps, so aliasing them away "
        "to save four corners would be a poor trade.\n",
        "\n## 4. Run count, and why this is the important section\n",
        "The response is binary and the baseline failure rate is low, which makes "
        "this expensive. Sample size per level is computed from a two sided two "
        "proportion test at alpha = "
        f"{ALPHA} and power = {POWER}. In a two level factorial, each main effect is "
        "estimated by comparing half the runs against the other half, so the total "
        "run count is twice the per level requirement.\n",
        md_table(runs),
        f"\n**Detecting a halving of the failure rate needs about "
        f"{int(headline['total lots in the design'])} lots.** That is the honest "
        "headline of this proposal. It is a large number, and it is the real reason "
        "observational screening studies like this one exist: the experiment that "
        "would answer the question properly costs more than most fabs will spend on "
        "a hypothesis that is still just a ranking.\n",
        "\nTwo consequences follow, and both are arguments for using the screen "
        "rather than against it. First, the screen is worth running precisely "
        "because it narrows 590 sensors to a handful, and it is far cheaper to buy "
        f"{int(headline['total lots in the design'])} lots of evidence about "
        f"{N_FACTORS} candidates than to attempt anything comparable across 590. "
        "Second, the run "
        "counts above assume the effect is real and of the stated size; the "
        "holdout result in `reports/model.md` gives no reason to be confident of "
        "that, so a smaller pilot to confirm the effect exists at all should come "
        "before committing to the full design.\n",
        "\n## 5. What this design would and would not establish\n",
        "It would establish whether deliberately setting these three parameters "
        "changes the failure rate, within the operating window tested, on the tool "
        "and product mix used, during the period it runs.\n",
        "\nIt would not establish a mechanism, would not extrapolate outside the "
        "tested window, and would not transfer automatically to another tool or "
        "another product. It also would not rescue any sensor that this repository "
        "shortlisted for the wrong reason: if a shortlisted sensor is a proxy, the "
        "experiment will correctly find that setting it changes nothing, which is a "
        "useful and cheap way to be wrong.\n",
    ]
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)}")
    print(
        f"top {N_FACTORS} factors: {', '.join(top['feature'])}; "
        f"halving the failure rate needs {int(headline['total lots in the design'])} lots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
