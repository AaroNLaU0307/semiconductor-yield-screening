# Semiconductor yield detractor screening

[![ci](https://github.com/AaroNLaU0307/semiconductor-yield-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/AaroNLaU0307/semiconductor-yield-screening/actions/workflows/ci.yml)

Screening candidate yield detractors in a wide, imbalanced, heavily missing
semiconductor process dataset (UCI SECOM), with false discovery control, a permutation
null, SPC control charts, and a cost aware predictive baseline.

## Headline findings

**The predictive model does not work, and that is reported rather than dressed up.**
Holdout PR-AUC is **0.062, with a 95 percent interval of 0.050 to 0.098**, against a
no-skill floor of 0.055. The interval contains the floor. ROC-AUC is 0.539. This is not
a working detector and should not be used as one.

**The screen, however, found something real.** On the training period, **32 of 424
features cleared Benjamini-Hochberg at q = 0.10**, and the permutation null is
decisive: across 2000 label permutations, 91.9 percent produced zero survivors, the
null mean was 0.25 and the null maximum was 15, against an observed 32
(**permutation p = 0.0005**). The associations in the train period are not an artifact
of testing 424 things at once.

**Those two findings are not a contradiction, and the reconciliation is the actual
result of this project: the process drifts.** The failure rate falls from 22.2 percent
in July 2008 to 2.9 percent in September, then rises again to 6.1 percent in October.
Every one of the ten charted sensors sits further outside its train control limits
during the holdout period than during the train period. Associations learned before the
split do not survive it. A study that had used random cross validation would have mixed
those periods together, reported a respectable number, and missed this entirely.

**The SPC flags do not anticipate failures either.** Out of control records fail no more
often than in control records, at every severity threshold under two rule sets, smallest
BH adjusted p = 1.000.

One honest caveat on all of it: the holdout contains **26 failures**. Every holdout
number here carries a wide interval, and this is weak evidence of no effect rather than
evidence of no effect.

### A defect found in this project's own frozen design

The out of control flag frozen at Gate 1 defined a record as out of control if any
applied Nelson rule fired on any charted sensor. Applied to ten sensors with per sensor
flag rates of 3 to 60 percent, that flags **468 of 471** holdout records by construction
and cannot discriminate anything. The declared result is reported as specified rather
than quietly replaced, the flaw is attributed to the declaration, and a full severity
sweep is added as a labelled post-hoc diagnostic. See [reports/spc.md](reports/spc.md).

## Why this repository looks different from other SECOM analyses

SECOM is heavily used and many published analyses report far stronger numbers. Before
comparing, check the protocol. Random cross validation on time ordered records,
oversampling applied before the split, and feature selection performed on the full
dataset each inflate the reported figure without improving the detector. The numbers
here come from a chronological split with every transform and the entire screen fitted
on the train partition alone, so **they are not comparable to numbers obtained under
those protocols, and they are not offered as a competitor to them.**

The differentiator is not the score. It is that the protocol supports the score.

## Method

Records are sorted by timestamp and split chronologically at 70 percent (1096 train,
471 holdout, boundary 2008-09-26). Column drop rules, median imputation and the feature
screen are all fitted on train only.

Sensors turn out to be either near-complete or catastrophically sparse, with **no sensor
having a train missing rate between 2 percent and 20 percent**. That empty band is why
the dense band ceiling is not a tunable parameter: any value inside it produces an
identical partition. Dense sensors (416) keep their values and are median imputed.
Sparse sensors (52) have their values dropped and contribute a binary not-measured
indicator instead, because imputing a sensor observed for one lot in five manufactures a
column that mostly carries the missingness pattern. Those 52 collapse to **8 distinct
missingness patterns**, which look like metrology sampling plans, so 8 indicators are
emitted rather than 52 duplicates.

Each of the resulting 424 features is tested against the label with a two sided
Mann-Whitney U test (tie corrected, because imputation creates large tie blocks),
reported with a rank-biserial effect size, and corrected with Benjamini-Hochberg at
q = 0.10. The whole screen, BH correction included, is then rerun on 2000 label
permutations.

Shortlisted sensors get I-MR charts with limits computed on train, and out of control
flags are cross referenced against actual holdout failures with Fisher's exact test.
One class weighted gradient boosting model provides the predictive baseline, with the
grid point and the decision threshold both selected on train out of fold predictions
before the holdout is scored once.

Every threshold was fixed in [DECISIONS.md](DECISIONS.md) before any modelling code was
written, and frozen at an owner approved gate.

### One result deliberately not promoted

The secondary L2 logistic model scored higher on the holdout (PR-AUC 0.104) than the
declared primary (0.062). It is **not** promoted to the headline. The primary was named
before any holdout record was read; swapping it afterwards is model selection on the
evaluation set. ROC-AUC also ranks the two the other way round, and the intervals
overlap, so the ordering is noise. Details in [reports/model.md](reports/model.md).

## Dashboard

![Dashboard](docs/dashboard.png)

`dashboard/index.html` is a single self contained file. Open it directly from disk: no
server, no install, no internet. Four panels, each captioned with what to conclude:
failure rate over time with the split marked, a selectable I chart with out of control
points flagged, the ranked detractor table, and the precision-recall curve with the
chosen operating point.

## Running it

```
pip install -r requirements.txt
python scripts/run_data_qa.py
python scripts/run_pipeline.py
python scripts/build_dashboard.py
python scripts/build_doe_proposal.py
```

The pipeline is deterministic: the same seed produces byte identical reports, which is
enforced by a test that runs it twice and compares.

## Reports

| report | contents |
| --- | --- |
| [reports/data_quality.md](reports/data_quality.md) | what the raw data actually contains, including the ugly parts |
| [reports/screening.md](reports/screening.md) | the screen, BH-FDR result, and the permutation null |
| [reports/spc.md](reports/spc.md) | control charts, the degenerate declared flag, and the severity sweep |
| [reports/model.md](reports/model.md) | holdout metrics, cost sweep, permutation importance |
| [reports/doe_proposal.md](reports/doe_proposal.md) | the experiment that would settle causation, not executed |

Everything in `reports/` is generated by a script and is never hand edited.

## What this project does not establish

It does not establish that any sensor causes yield loss. All 590 sensors come from
observational records of a running process with no randomised intervention, so every
ranking here is a ranking of association. A shortlisted sensor may matter, may proxy
something that matters, or may simply move with tool age or product mix.

[reports/doe_proposal.md](reports/doe_proposal.md) specifies the designed experiment
that would settle it, and computes its cost: detecting a halving of the failure rate
would need roughly **1346 lots**. That number is the honest reason observational
screening studies exist, and it is also the reason the screen is worth running, since
buying that evidence for 3 candidates is tractable and buying it for 590 is not.

Nor does a weak result here establish that no detractors exist. It establishes that none
is detectable across this time boundary, at this sample size, at this effect size,
through this preprocessing.

## Data

UCI SECOM, 1567 records, 590 sensors, 104 failures (6.64 percent), July to October 2008.

> McCann, M. and Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C54305

The raw files are committed under `data/raw` so the pipeline runs offline, and are
stored byte exact (git end of line normalisation is disabled for that directory) so the
recorded sha256 checksums verify on a fresh clone.

| file | sha256 |
| --- | --- |
| `secom.data` | `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c` |
| `secom_labels.data` | `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03` |
| `secom.names` | `6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59` |

Checksums are verified on every load by `yield_screening/data_io.py`.

## Repository layout

| path | contents |
| --- | --- |
| [DECISIONS.md](DECISIONS.md) | every frozen methodological choice with reasons, plus an adversarial self review |
| [PLAN.md](PLAN.md) | phase plan, module layout, test schedule |
| `yield_screening/` | library code |
| `scripts/` | entry points |
| `reports/` | generated reports |
| `dashboard/` | self contained HTML dashboard and its data |
| `tests/` | test suite |

## Tests

```
pytest -q
```

Four properties are protected, each written in the phase that produced the code it
covers: temporal ordering of the split, fit scope (perturbing a holdout row cannot
change any fitted parameter), screening behaviour on planted signal and on pure noise,
and seed determinism. A fifth pins the screening statistic against `scipy` and
`statsmodels`, and pins the fast matrix path used for the 2000 permutations against the
readable reference path, because a permutation null computed from a subtly different
statistic than the observation looks rigorous and is worthless.

On pure noise across 100 fixed seeds, the screen produced at least one survivor in 4
percent of runs, against a theoretical BH bound of 10 percent.

CI also fails the build if an em dash appears anywhere in the repository.

## Licence

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Aaron Lau Chiong Wen.
