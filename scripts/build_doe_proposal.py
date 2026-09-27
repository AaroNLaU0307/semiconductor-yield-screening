"""Emit reports/doe_proposal.md: a designed experiment that was NOT executed.

This is a proposal, not a result. The SECOM data is observational, collected from a
running process with no randomised intervention, so nothing in this repository can
establish that a sensor causes yield loss. The only way to get a causal answer is to
set the factors deliberately and randomise, which is what this document specifies and
what this project did not do.

The factors are the shortlisted sensor values that replicated on the holdout period
(`dashboard/replication_data.json`, written by the post-hoc replication analysis in
`scripts/run_pipeline.py`), not the largest train-period effects: those did not hold
up when the period changed, so an experiment on them would test associations the
project's own holdout already fails to support.

The run counts below are computed, not asserted. The point of the calculation is that
it is expensive: it tells a reader what the observational shortcut is actually
standing in for.

Usage, from the repository root:
    python scripts/build_doe_proposal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yield_screening.config import FDR_Q, INDICATOR_SUFFIX
from yield_screening.reporting import kv_table, md_table  # noqa: E402

REPORT = REPO_ROOT / "reports" / "doe_proposal.md"
SCREEN_TABLE = REPO_ROOT / "reports" / "screening_table.csv"
REPLICATION = REPO_ROOT / "dashboard" / "replication_data.json"

NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")
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


def words(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def select_factors(replication: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(factors, passed_over) from the replication analysis.

    Factors are the shortlisted sensor values that replicated on the holdout under BH,
    ordered by holdout adjusted p. Missingness indicators are excluded, since they
    describe metrology routing rather than a settable process parameter.
    `passed_over` lists the sensor values that rank above every factor by train
    effect size and did not replicate: the ones a train-only ranking would pick first.
    """
    features = pd.DataFrame(replication["features"])
    sensors = features[~features["feature"].str.endswith(INDICATOR_SUFFIX)].copy()
    sensors["abs train effect"] = sensors["train effect"].abs()
    factors = sensors[sensors["replicates"]].sort_values("holdout BH adjusted p", kind="stable")
    if factors.empty:
        return factors, sensors.iloc[0:0]
    passed_over = sensors[
        ~sensors["replicates"]
        & (sensors["abs train effect"] > factors["abs train effect"].max())
    ].sort_values("abs train effect", ascending=False, kind="stable")
    return factors, passed_over


def design_note(n_factors: int) -> str:
    """Why a full factorial, stated for the number of factors actually proposed."""
    if n_factors == 1:
        return (
            "With a single factor the design is a two arm comparison, randomised within "
            "each day.\n"
        )
    if n_factors == 2:
        return (
            "A full factorial: with two factors it has four corners, and a half "
            "fraction would confound the two main effects with each other. The full "
            "design also estimates their interaction.\n"
        )
    half = 2 ** (n_factors - 1)
    return (
        f"A full factorial rather than a fractional design: with only {words(n_factors)} "
        f"factors the full design costs {2**n_factors} corners rather than {half}, and it "
        "buys clean two factor interactions. Interactions are the plausible mechanism "
        "here, since these sensors sit on correlated process steps, so aliasing them "
        f"away to save {half} corners would be a poor trade.\n"
    )


def main() -> int:
    for path, script in ((SCREEN_TABLE, "run_pipeline.py"), (REPLICATION, "run_pipeline.py")):
        if not path.exists():
            raise SystemExit(
                f"{path.relative_to(REPO_ROOT)} is missing, run scripts/{script} first"
            )
    replication = json.loads(REPLICATION.read_text(encoding="utf-8"))
    factors, passed_over = select_factors(replication)
    n_factors = len(factors)
    if n_factors == 0:
        raise SystemExit(
            "no shortlisted sensor value replicated on the holdout, so there is no "
            "candidate to design an experiment around"
        )
    n_shortlisted = int(replication["summary"]["n_shortlisted"])

    factor_table = pd.DataFrame(
        {
            "factor": [f"F{i + 1}" for i in range(n_factors)],
            "sensor": factors["feature"].to_list(),
            "train effect": factors["train effect"].round(4).to_list(),
            "holdout effect": factors["holdout effect"].round(4).to_list(),
            "holdout BH adjusted p": factors["holdout BH adjusted p"].to_list(),
            "low level": "train 25th percentile",
            "high level": "train 75th percentile",
        }
    )
    if len(passed_over):
        listed = ", ".join(
            f"{row['feature']} (train {row['train effect']:+.2f}, holdout "
            f"{row['holdout effect']:+.2f})"
            for _, row in passed_over.iterrows()
        )
        passed_over_text = (
            "\nRanking by train-period effect size alone would put "
            f"{words(len(passed_over))} other sensor values ahead of these: {listed}. "
            "None of them replicated (`reports/replication.md`), so an experiment on "
            "them would test associations that did not hold up when the period changed.\n"
        )
    else:
        passed_over_text = ""

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
        f"The {words(n_factors)} shortlisted sensor values that replicated on the "
        f"holdout period, under BH at q = {FDR_Q:.2f} over the family of {n_shortlisted} "
        "shortlisted features (`reports/replication.md`). Missingness indicators are "
        "excluded, since they describe metrology routing rather than a settable process "
        "parameter. The replication analysis is post-hoc, so this choice is too: the "
        "experiment is the test on fresh data that those features still need.\n",
        md_table(factor_table),
        passed_over_text,
        "\nLevels are set at the train 25th and 75th percentiles rather than at "
        "extremes. Extreme levels would raise the power of the design but would run "
        "the process outside its normal operating window, which risks scrapping "
        "material and answers a question about a process nobody runs.\n",
        "\n## 3. Design\n",
        kv_table(
            [
                (
                    "design",
                    f"2^{n_factors} full factorial" if n_factors > 1 else "two arm comparison",
                ),
                ("factors", n_factors),
                ("levels per factor", 2),
                ("distinct corners", 2**n_factors),
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
        "\n" + design_note(n_factors),
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
        f"{n_factors} candidates than to attempt anything comparable across 590. "
        "Second, the run "
        "counts above assume the effect is real and of the stated size; the "
        "holdout result in `reports/model.md` gives no reason to be confident of "
        "that, so a smaller pilot to confirm the effect exists at all should come "
        "before committing to the full design.\n",
        "\n## 5. What this design would and would not establish\n",
        "It would establish whether deliberately setting "
        + ("this parameter" if n_factors == 1 else f"these {words(n_factors)} parameters")
        + " changes the failure rate, within the operating window tested, on the tool "
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
        f"{n_factors} factors: {', '.join(factors['feature'])}; "
        f"halving the failure rate needs {int(headline['total lots in the design'])} lots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
