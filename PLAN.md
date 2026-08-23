# PLAN

Execution plan for `semiconductor-yield-screening`. Written once. Referenced, not restated.

Scope, standard, and constraints live in the project brief. `DECISIONS.md` holds every
frozen numeric threshold and is the authority when this file and it disagree.

## Phase status

| phase | output | gate | status |
| --- | --- | --- | --- |
| 1. Load, QA, lock decisions | `reports/data_quality.md`, `DECISIONS.md` | Gate 1, Aaron approves | done, approved |
| 2. Screening | `reports/screening.md` | none | done |
| 3. SPC | `reports/spc.md`, `dashboard/spc_data.json` | none | done |
| 4. Predictive baseline | `reports/model.md` | none | done |
| 5. Dashboard and wrap | `dashboard/index.html`, `README.md`, CI | Gate 2, Aaron reviews findings | done, awaiting review |
| 5b. Replication and drift (post-hoc, unplanned) | `reports/replication.md` | none | done |
| 6. SQL warehouse layer (post-Gate, 2026-08-23) | `sql/`, `yield_screening/warehouse.py` | none | done, no reported number moved |

## Module layout

```
yield_screening/
  data_io.py      load and parse raw SECOM, sort by timestamp, verify checksums
  warehouse.py    build and query the DuckDB warehouse, added after Gate 1
  config.py       every frozen constant from DECISIONS.md, single source of truth
  splitting.py    chronological split
  preprocess.py   train-fitted column filter, median imputation, missingness indicators
  screening.py    Mann-Whitney U, rank-biserial effect size, BH-FDR, permutation null
  spc.py          I-MR control limits, Nelson rules, contingency test
  modeling.py     baselines, HistGradientBoosting, L2 logistic, PR-AUC, cost sweep
  reporting.py    markdown emitters
sql/
  schema.sql              lot and measurement tables, long form
  partition.sql           the chronological boundary as a view, not a slice
  partition_boundary.sql  partition sizes and the timestamps either side
  sensor_stats.sql        per sensor coverage, distinctness, modal share, fingerprint
  drop_rules.sql          the declared drop rules applied in SQL
  missingness_patterns.sql  unsupervised grouping of the missingness matrix
  monthly_failure_rate.sql  failure rate by month
  monthly_sensor_means.sql  per sensor monthly means, for the drift analysis
scripts/
  run_data_qa.py        phase 1
  run_pipeline.py       the one command, phases 2 to 5
  check_no_em_dash.py   CI guard for non-negotiable 1
tests/                  four required tests, written in the phase that produces the code
```

## Command surface

One command reproduces everything after QA:

```
python scripts/run_pipeline.py
```

## Test schedule

Tests are written in the phase that produces the code they cover, not deferred.

| test | covers | written in |
| --- | --- | --- |
| temporal ordering | every holdout timestamp is later than every train timestamp | phase 2 |
| fit scope | perturbing a holdout row cannot change a fitted transform parameter | phase 2 |
| screening recovers planted signal, and holds its FDR bound on pure noise | phase 2 |
| seed determinism | same seed, same run, same reported metrics | phase 4 |

Supporting test, not in the required four but cheap and load bearing: the vectorised
permutation screen must agree with `scipy.stats.mannwhitneyu` to floating point
tolerance on a sample of columns. The permutation null is worthless if the fast path
computes a different statistic from the reference path.

## Cut order if behind

DOE proposal, then the second model (L2 logistic), then the SPC contingency test.
The Dash server version is already cut by owner decision. Anything cut is recorded
as cut for time in `README.md`.
