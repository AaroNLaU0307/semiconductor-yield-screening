# DECISIONS

Every methodological choice is fixed here **before any modelling code is written**, and
frozen at Gate 1. Each decision states the rule, the number, and the reason. Where a
threshold could have been tuned, the evidence that it was not is given.

All evidence cited below comes from `reports/data_quality.md`, produced by
`scripts/run_data_qa.py`. Where a statistic is used to justify a threshold, the
**train-only** version of that statistic is the one that justifies it, so that no
holdout record influences the design.

Constants named here are implemented once in `yield_screening/config.py` and imported
everywhere. If a number appears in code but not in this file, that is a defect.

**Status: FROZEN at Gate 1, approved by the owner (Aaron Lau Chiong Wen).** Every value
below was fixed before any screening, SPC or modelling code was written, and before any
holdout record was read. Per D12 none of it may change on the basis of a result.

---

## D0. Dataset as measured

| property | value |
| --- | --- |
| rows | 1567 |
| sensor columns | 590 |
| fail records | 104 |
| fail rate | 0.0664 |
| time span | 2008-07-19 11:55:00 to 2008-10-17 06:07:00 |
| overall cell missing rate | 0.0454 |

Label encoding: the raw files use -1 for pass and +1 for fail. Everything downstream
uses 1 for fail and 0 for pass, so the minority class is the positive class and
precision, recall and PR-AUC carry their usual meaning.

**Measured non-stationarity.** The failure rate is not stable over the record:

| month | records | fails | fail rate |
| --- | --- | --- | --- |
| 2008-07 | 63 | 14 | 0.2222 |
| 2008-08 | 555 | 51 | 0.0919 |
| 2008-09 | 590 | 17 | 0.0288 |
| 2008-10 | 359 | 22 | 0.0613 |

This single table drives several decisions below. A process whose failure rate moves
by a factor of seven across three months is not exchangeable across time, so any
procedure that treats the records as an unordered bag is measuring the wrong thing.

---

## D1. Chronological split

**Rule.** Sort ascending by timestamp, then take the first 70 percent of rows by
position as train and the remainder as holdout.

**Frozen values.**

| property | value |
| --- | --- |
| train rows | 1096 (78 fails, rate 0.0712) |
| holdout rows | 471 (26 fails, rate 0.0552) |
| last train timestamp | 2008-09-26 02:26:00 |
| first holdout timestamp | 2008-09-26 03:12:00 |
| boundary strict | yes, no timestamp straddles the split |

**Why chronological rather than random.** Random cross validation on time-ordered
process data leaks in two distinct ways, and both are active in this dataset. First,
tool state is autocorrelated: adjacent records share chamber condition, consumable age
and recipe revision, so a randomly held-out record has near neighbours in train and the
model is scored on interpolation rather than prediction. Second, and specific to this
data, the failure rate itself moves (D0), so a random fold silently averages the July
regime into the September regime and reports a number that no deployed monitor could
achieve. A random split here does not produce a slightly optimistic estimate, it
produces an estimate of a different quantity.

**Note on the raw file.** The raw file happens to already be in chronological order,
which `reports/data_quality.md` confirms. The loader sorts regardless. Confirming an
ordering is not the same as relying on it.

**Accepted cost, stated plainly.** The holdout contains **26 failures**. That is a small
positive count, and every holdout metric derived from it carries wide uncertainty. This
is not hidden: PR-AUC is reported with a bootstrap interval (D10) and the interval is
reported in the README headline, not buried. A larger holdout would be more precise
about a quantity that would be less meaningful.

---

## D2. Column drop rules

Fitted on **train rows only**. Applied unchanged to holdout.

Applied in this order:

1. Drop if entirely missing on train.
2. Drop if at most one distinct non-null value on train (constant).
3. Drop if the single most common non-null value covers **at least 99 percent** of
   non-null train observations (near-constant).
4. Drop if train missing rate exceeds **0.95**.
5. Deduplicate: among sensors whose train value vectors are identical including the
   missing pattern, keep the lowest numbered and drop the rest.

**Frozen thresholds.** Near-constant modal-share ceiling 0.99. Missing-rate hard
ceiling 0.95. Dense-band ceiling 0.20, see D3.

**Why a modal-share ceiling rather than a variance floor.** The 590 sensors sit on
wildly different units, with means spanning several orders of magnitude. Any absolute
variance floor would be a unit-dependent decision disguised as a statistical one, and
would drop informative low-magnitude sensors while keeping uninformative high-magnitude
ones. Modal share is scale free and directly expresses the property that actually
disqualifies a column: it barely varies, so it cannot be associated with anything.

**Measured effect, reported honestly.** Rule 5 removes **110 sensors on train, all 110
of which rule 2 had already removed**. The deduplication rule therefore does **zero**
additional work on this dataset. It is retained because it is cheap and correct, not
because it accomplished anything here, and the report will say so rather than implying
the pipeline found and removed redundancy it did not find.

---

## D3. Missing data strategy

**The measurement that determines this decision.** Per-sensor train missing rate is
sharply bimodal, with a completely empty band in the middle:

| train missing rate band | sensors |
| --- | --- |
| [0.00, 0.001) | 252 |
| [0.001, 0.01) | 279 |
| [0.01, 0.02) | 7 |
| **[0.02, 0.20)** | **0** |
| [0.20, 0.30) | 20 |
| [0.30, 0.50) | 8 |
| [0.50, 0.80) | 4 |
| [0.80, 0.95) | 20 |
| [0.95, 1.00] | 0 |

Sensors are either near-complete or catastrophically sparse. Nothing lies between
2 percent and 20 percent missing.

**Consequence: the dense-band ceiling is not a tunable parameter.** Any ceiling placed
anywhere in [0.02, 0.20] produces a byte-identical partition. The declared value of
**0.20** is therefore not a choice that could have been tuned toward a nicer result, and
the emptiness of the band is the proof. This is stated because "we tried several
thresholds and picked one" is the standard and correct objection to a screening study,
and here it is answerable with a measurement rather than an assurance.

**Rule.**

- **Dense band, train missing rate at most 0.20.** Keep the values. Impute missing
  entries with the **train median of that sensor**. Expected count: **416 sensors**.
- **Sparse band, train missing rate above 0.20 and at most 0.95.** Do **not** impute the
  values: imputing a sensor observed for one lot in five manufactures a column that is
  mostly a single constant, and that constant then carries whatever association the
  missingness pattern carries, which is a fabricated finding. Instead drop the values
  and retain a binary indicator, 1 where the sensor was not measured. Expected count:
  **52 sensors**.
- **Deduplicate the indicators.** The 52 sparse sensors collapse to **8 distinct
  missingness patterns**, so 8 indicator columns are emitted, each named for its lowest
  numbered member sensor, with full group membership recorded in `reports/screening.md`.
  Emitting 52 exactly identical columns would pad the FDR family with redundant tests
  and misrepresent 8 findings as 52.
- **No indicator for dense-band sensors.** Measured: only 2 of the 416 dense-band
  sensors exceed 1 percent train missingness, and none exceeds 2 percent. An indicator
  there would be near-constant and would be dropped by D2 rule 3 anyway.

**Why the sparse-band indicators are kept rather than discarded.** In a fab, a sensor
measured for only a minority of lots usually reflects a metrology **sampling plan**, not
a broken tool. That the 52 sparse sensors collapse to exactly 8 patterns is consistent
with 8 such plans. If a sampling plan is associated with the failure rate, that is a
real and actionable finding, and it is also a **confound warning**: it may mean the
failing lots were routed differently, not that the measured quantity matters. Both
readings are reported. The indicator is labelled as a process-routing feature, never as
a yield detractor.

**Imputation scope.** Medians are computed on train rows only and stored. The holdout is
transformed with the stored train medians. This is enforced by the fit-scope test.

---

## D4. Screening test and effect size

**Test.** Two-sided Mann-Whitney U, fail group against pass group, on train rows only,
one test per screened feature.

**Why Mann-Whitney U.** Sensor distributions are not reliably normal, several are heavily
skewed or spiky, and the fail group has only 78 train members. A rank test makes no
distributional assumption, is robust to the outliers this data certainly contains, and
retains power at this group size. The tie-corrected normal approximation is used for the
p-value, which matters because median imputation creates large tie blocks; ignoring the
tie correction would understate the variance and inflate significance.

**Effect size, reported for every feature without exception.** Rank-biserial correlation

    r = 2 * AUC - 1,  equivalently  r = 1 - 2U / (n_fail * n_pass)

`AUC` here is the probability that a randomly chosen fail record exceeds a randomly
chosen pass record on that sensor. `r` runs from -1 to +1, is 0 under no association,
and is scale free, so the 424 features are comparable to each other despite their units.
A p-value without an effect size is not a finding; with 1096 train rows, a negligible
shift can be significant, and the effect size is what says whether an engineer should
care.

**Screen family.** Dense-band sensor values plus deduplicated sparse-band indicators,
tested as **one family**: expected **416 + 8 = 424 tests**. The indicators are in the
same family rather than a separate one because splitting a screen into families after
seeing it is a standard route to an uncorrected discovery.

**Note on the indicators.** Mann-Whitney U on a binary column reduces to a comparison of
proportions. It is valid, and using one test for the whole family keeps the FDR
correction honest, which is worth more than a marginal power gain from a bespoke test.

---

## D5. False discovery control

**Rule.** Benjamini-Hochberg at **q = 0.10**, applied across the full screen family of
424 tests, on the raw two-sided p-values.

**Reported alongside the survivor count**, always together:

- number of features tested
- number surviving
- expected number of false discoveries among survivors, `q * (survivors)`
- the smallest raw p-value that failed to survive, so the margin is visible

**Why q = 0.10 rather than 0.05.** This is a screen whose output is a shortlist for
follow-up engineering investigation, not a confirmatory test. The asymmetry of costs
favours tolerating a false lead over discarding a real one, since a false lead is
resolved cheaply by the SPC step and the DOE proposal. The level is declared here in
advance precisely so it cannot be relaxed later if the result is thin.

**BH assumes independence or positive regression dependence.** Sensors in this dataset
are certainly correlated, and there are large blocks of them. Positive correlation is
the case BH tolerates; the concern would be strong negative dependence, which is not the
expected structure among co-located process sensors. This assumption is stated rather
than assumed away, and the permutation null in D6 is the empirical check that does not
depend on it at all.

---

## D6. Permutation null

**Rule.** Hold the feature matrix fixed. Permute the train label vector. Rerun the
**entire screen**, including BH correction, on each permutation. Record the survivor
count each time.

**Frozen values.**

| parameter | value |
| --- | --- |
| permutations B | 2000 |
| seed | 20080719 |
| permuted object | train labels only |
| statistic recorded | number of BH survivors at q = 0.10 per permutation |

**Reported.** The full null distribution of survivor counts, the observed count, and the
permutation p-value `(1 + #{null >= observed}) / (B + 1)`. Reported **whether flattering
or not**.

**Why this is the highest value artifact in the project.** BH-FDR controls the false
discovery rate under assumptions about the dependence structure among 424 correlated
tests. The permutation null does not assume that structure, it reproduces it: the
correlation among sensors is preserved exactly, because only the labels move. If the
observed survivor count sits inside the null distribution, the screen found nothing, and
no amount of BH arithmetic changes that. This is the one result that cannot be argued
with.

**Implementation constraint.** Ranks do not change when labels are permuted, so ranks
are computed once per column and each permutation is a matrix product. This makes 2000
permutations cheap. Because the fast path computes the statistic differently from a
per-column `scipy.stats.mannwhitneyu` call, a test asserts the two agree to floating
point tolerance. A permutation null computed by a subtly different statistic than the
observed value is worse than no permutation null, because it looks rigorous.

---

## D7. Fallback if few or no features survive

**Rule.** If fewer than 3 features survive BH at q = 0.10, additionally carry forward the
top **k = 20** features ranked by absolute rank-biserial effect size.

Carried-forward features that did not clear FDR are labelled

> effect-size ranked, did not clear FDR

**everywhere they appear**: report tables, dashboard table, SPC chart titles, and model
feature list. The label is a column in the emitted data, not prose added by hand, so it
cannot be dropped by accident.

**k = 20 is frozen now**, before any screening result is seen, which is the only time
this number can be chosen honestly.

**Few or zero survivors is a legitimate reportable outcome.** It is not a failure state
and it does not trigger a change of test, level, or preprocessing. If the screen comes
back empty, the project reports an empty screen, a permutation null consistent with it,
and an SPC and modelling section built on the effect-size-ranked fallback, clearly
labelled as such.

---

## D8. SPC

**Charts.** Individuals and moving range (I-MR), one pair per shortlisted sensor.

**Shortlist for SPC.** BH survivors that are dense-band **sensor values**, capped at the
**top 10 by absolute effect size**. Sparse-band missingness indicators are excluded from
SPC: a binary routing flag has no control limits and charting it would be a category
error. If the FDR shortlist is short, the D7 fallback features are charted under their
label.

**Control limits.** Computed on **train only**, applied unchanged to holdout.

    centre    = mean of train individuals
    MR-bar    = mean of train moving ranges of consecutive records
    UCL, LCL  = centre +/- 2.66 * MR-bar
    MR chart UCL = 3.267 * MR-bar

**Nelson rules applied: 1, 2, 3, 5, 6.**

| rule | condition |
| --- | --- |
| 1 | one point beyond 3 sigma |
| 2 | nine consecutive points on the same side of centre |
| 3 | six consecutive points steadily increasing or decreasing |
| 5 | two of three consecutive points beyond 2 sigma, same side |
| 6 | four of five consecutive points beyond 1 sigma, same side |

**Rules 4, 7 and 8 are excluded, with reason.** Rule 4 (fourteen alternating) and rules 7
and 8 (stratification and mixture) are pattern rules that assume a stable within-subgroup
sigma and roughly continuous readings. This data has median-imputed values creating exact
ties, a 37 minute median sampling cadence with gaps up to 48.6 hours, and no rational
subgroups. Under those conditions those three rules fire on artefacts of the data
preparation rather than on process behaviour. Declaring the applied subset is required;
applying all eight and reporting the union would inflate the OOC rate for free.

**Stated limitation.** I-MR assumes approximately normal and independent individuals.
Neither holds cleanly here. Control limits are therefore treated as a **screening
heuristic for where to look**, not as calibrated false-alarm rates, and the report says
so at the point where the limits are introduced.

**Contingency test.** A 2x2 table on **holdout** records: OOC flagged (any applied rule
fires on any shortlisted sensor for that record) against actual fail. Reported with
**Fisher's exact test**, two-sided, plus the risk ratio and its confidence interval.
Fisher rather than chi-square because with 26 holdout failures the expected cell counts
will be small and the chi-square approximation is not trustworthy there.

---

## D9. Predictive baseline

**Floor first.** The majority-class predictor (predict pass always) is reported before
any model: accuracy about 0.945 on holdout, recall 0, PR-AUC equal to the positive rate.
It is reported to make the point that accuracy is uninformative here, not as a
competitor.

**Model.** `HistGradientBoostingClassifier`, class weighted, fitted on train, on the
screened feature set from D4 to D7.

**Second model if time allows.** L2 penalised logistic regression on standardised
screened features, class weighted. Owner has authorised this as in scope.

**Declared grid.** Small and fixed in advance, no search beyond it, no AutoML, no Optuna:

| parameter | values |
| --- | --- |
| `learning_rate` | 0.05, 0.10 |
| `max_leaf_nodes` | 15, 31 |
| `min_samples_leaf` | 20 |
| `max_iter` | 200 with early stopping |
| `class_weight` | balanced |

Selection among these is by mean PR-AUC over a **forward-chaining time series split with
5 folds on train only**. The holdout takes no part in selection.

**Class weights, not SMOTE.** Declared out of scope by the brief and correct
independently. SMOTE interpolates between minority neighbours in a 400 dimensional
space where the minority class has 78 train members; the synthetic points it creates lie
in regions no wafer occupies, and their density is an artefact of the sampling rather
than of the process. On time-ordered data it is worse still: synthesising a record from
neighbours that straddle the split is leakage in its purest form. Class weighting changes
the loss without inventing observations.

---

## D10. Metrics

| role | metric |
| --- | --- |
| **primary** | PR-AUC (average precision), positive class is fail |
| secondary | recall at a fixed false-alarm budget |
| reported, never headlined | ROC-AUC |
| reported as a floor only | accuracy |

**No-skill line.** Every PR curve is drawn with the no-skill line at the **holdout
positive rate 0.0552**, so a reader can see immediately how much of the curve is skill.

**False-alarm budget, frozen.** Recall at a **holdout false positive rate of at most
0.10**, that is, at most 10 percent of passing lots flagged. This is an operational
budget, not a statistical one: it is the rate at which pulling lots for review stops
being affordable. Frozen now so it cannot be moved to where the model looks best.

**Uncertainty, mandatory.** With 26 holdout failures, a point estimate of PR-AUC is not a
result. A **stratified bootstrap over holdout records, 2000 resamples, seed 20080719**,
gives a percentile interval, and the interval is reported next to the point estimate
every time the point estimate appears, including in the README headline.

**Why PR-AUC rather than ROC-AUC or accuracy.** At a 5.5 percent positive rate, ROC-AUC
is dominated by the true negative mass and stays flattering while precision is
unusable; PR-AUC responds to exactly the quantity the engineer cares about, which is how
many of the flagged lots are genuinely at risk. Accuracy is uninformative by
construction, as the 0.945 floor shows.

---

## D11. Cost aware operating point

**Assumed cost ratio, frozen.** Cost of a missed failure to cost of a false alarm =
**10 to 1** as the base case.

**Sweep, frozen.** Ratio from **1:1 to 100:1**, 25 points log spaced. The sweep is the
deliverable; the single point is an illustration of how to read it. A reader whose real
ratio is 40:1 must be able to find their own threshold on the curve.

**Threshold selection is fitted on train, not on holdout.** The minimising threshold is
chosen from **out-of-fold predicted probabilities on train**, produced by the same
forward-chaining 5 fold time series split used in D9. The chosen threshold is then
applied **unchanged** to the holdout. Choosing the threshold on the holdout and then
reporting holdout performance at that threshold would be fitting on the evaluation set,
and it is a very common way this exact analysis is done wrong.

**Reported.** Expected cost against threshold for the base ratio, the selected threshold,
the resulting holdout confusion matrix, and the full sweep of selected threshold against
cost ratio.

---

## D12. Holdout usage policy

The brief says the holdout is opened once in Phase 4, and also specifies an SPC
contingency test on the holdout in Phase 3. Those two statements conflict. Resolving it
by silence would be dishonest, so it is resolved here, in advance.

**Rule.** The holdout is read in exactly **two** places, both declared before either is
run:

1. **D8**, the SPC contingency table.
2. **D9 to D11**, the model metrics and the cost sweep.

**Binding constraint.** No quantity fixed in this document may be changed after any
holdout read. Specifically: no change to the split, the drop rules, the imputation, the
screening test, q, k, the model grid, the false-alarm budget, the cost ratio or the
sweep range. If a holdout result is disappointing, the disappointing result is what gets
reported.

**What "opened once" actually protects,** and what is preserved here, is the absence of
an iterative loop between holdout performance and design choices. Two pre-declared
read-only evaluations do not create such a loop. A single read followed by a redesign
would. The count of reads is not the property that matters; the direction of information
flow is.

---

## D13. Determinism

**Global seed 20080719**, used for the permutation null, the bootstrap, model fitting and
any subsampling. Set explicitly at every call site rather than relying on global state.
`scripts/run_pipeline.py` run twice with the same seed must emit identical reported
metrics. Enforced by the seed determinism test.

---

## D14. Vocabulary

**"Associated with"**, never "causes", "drives", "responsible for" or "root cause". This
is observational data from a running fab with no randomised intervention. Every ranking
in this repository is a ranking of association, and the reports say so at the point of
each ranking rather than once in a disclaimer.

Weak or null findings are reported in plain manufacturing-analytics terms: "did not clear
the screening threshold", "no association detectable above the noise floor". There is no
falsification branding in this repository.

---

# Adversarial self-review

Performed before Gate 1, per the brief. The question asked was: what would a hostile
reviewer attack, what could leak, and what is underspecified.

### 1. "You chose thresholds after looking at the data."

Partly true and worth being precise about. The drop thresholds were chosen after seeing
the missingness structure. Three things bound the damage. **No labels were involved** in
any threshold choice, so the choice cannot manufacture an association. The justifying
statistics are **train only**, so the holdout did not inform the design. And for the one
threshold that could plausibly have been tuned, the dense-band ceiling, the interval
[0.02, 0.20] is **empty of sensors**, so every value in it yields the identical
partition. Residual exposure: the modal-share ceiling of 0.99 and the hard ceiling of
0.95 were picked as round numbers, not from a gap. Both are recorded here before
screening runs, and the survivor count under alternatives is already published in
`reports/data_quality.md` section 5 so a reviewer can check the sensitivity without
taking my word for it.

### 2. "Median imputation before a rank test biases the screen."

Real and partially mitigated. Imputing the train median inserts a spike at the median,
which shrinks the apparent difference between groups (conservative, biases toward finding
nothing) but also creates large tie blocks, which the tie-corrected variance handles.
The unmitigated part: if missingness is itself associated with the label, imputation in
the dense band hides that. Bounded by measurement, because no dense-band sensor exceeds
2 percent train missingness, so at most 2 percent of any dense-band column is imputed.
The sparse band, where this concern would be serious, is not imputed at all (D3).

### 3. "424 correlated tests, and BH assumes a dependence structure you have not verified."

Correct, and this is exactly why D6 exists. The permutation null preserves the
correlation structure exactly and needs no assumption about it. If BH and the permutation
null disagree, the permutation null is reported as the binding result.

### 4. "The screened features are chosen on train, then a model is fitted on train. The screening step is part of the model and you have not paid for it."

The right objection, and it is answered for the reported number but not for the internal
one. The **holdout** metrics are clean: screening used no holdout record, so holdout
PR-AUC pays for the screen in full. The **train** out-of-fold estimates used for
hyperparameter and threshold selection are optimistic, because screening was done on all
of train rather than inside each fold. This is accepted deliberately: nesting the screen
inside 5 folds would leave roughly 62 failures per fold to screen 424 features, which is
too thin to be stable. The consequence is that the out-of-fold PR-AUC is an inflated
number, so it is used **only for selection among the declared grid** and is never
reported as performance. This limitation is stated in `reports/model.md`, not just here.

### 5. "The holdout has 26 failures. Any number you report is noise."

Substantially fair. Mitigated by mandatory bootstrap intervals reported next to every
point estimate (D10) and by the README stating the interval in its first paragraph. Not
fully solvable: it is a property of the dataset. The honest position is that this project
reports a wide interval accurately rather than a narrow interval obtained by a protocol
that would not survive scrutiny.

### 6. "Non-stationarity means train and holdout are different processes, so a weak holdout result may just be drift."

Correct, and it cuts both ways: drift is also the reason the chronological split is the
only defensible one. Distinguishing "the sensors carry no signal" from "the sensors
carried signal that drifted" is not identifiable from 1567 observational records. This is
recorded as a stated limitation, not resolved, and the monthly failure rate table is put
in the dashboard so a reader sees the drift directly rather than being told about it.

### 7. "The missingness indicators are a leak: you are predicting failure from metrology routing, not from process physics."

Sharp, and the reason the indicators are labelled as process-routing features rather than
yield detractors (D3). If an indicator screens strongly, the report will present it as
evidence about lot routing and explicitly warn that it is a confound, not a detractor.
It is kept rather than dropped because concealing a strong association to protect a
tidier narrative would be the worse error.

### 8. "SPC control limits from I-MR are invalid on non-normal, autocorrelated data."

Accepted, stated at the point of use (D8), and the reason the SPC output is framed as
where to look rather than as a calibrated alarm rate. The contingency test against actual
holdout failures is the check that does not depend on the limits being calibrated: it
asks only whether flagged records fail more often, which is a valid question regardless
of whether the 3 sigma limit means 3 sigma.

### 9. Underspecified before this review, now fixed.

- Threshold selection was not clearly assigned to train; it is now (D11).
- The screen family was not clearly defined as including indicators; it is now (D4).
- The permutation fast path could silently compute a different statistic than the
  reference; a test now pins it (D6, PLAN.md).
- "Opened once" was contradictory in the brief; resolved explicitly (D12).
- Bootstrap intervals were not required anywhere; they are now mandatory (D10).

### 10. What this project will not establish, whatever the numbers say

It will not establish that any sensor causes yield loss. It will not establish that a
shortlisted sensor would improve yield if controlled. It will not establish that the
model would hold up on 2009 data or on another fab. And a null screen would not establish
that no detractors exist, only that none is detectable at this sample size, at this
effect size, through this preprocessing.

---

# Addendum, written after the results were known

**Nothing in this addendum changes a frozen value.** Per D12 no quantity above may move
on the basis of a result, and none has. This section records what the results taught,
including where the frozen design turned out to be wrong, and consolidates the answers
to the objections a reviewer is most likely to raise.

## A1. A defect in D8, found by running it

D8 defined a holdout record as out of control if **any** applied Nelson rule fires on
**any** charted sensor. Executed, that flag marks **468 of 471** holdout records, leaves
3 in control, and cannot discriminate anything. Its risk ratio is not estimable.

The cause is arithmetic and should have been foreseen at Gate 1. The measured per sensor
flag rates run from 3 percent to 60 percent, and a union over ten such sensors reaches
roughly 99 percent by construction. The error was declaring a union without checking
what a union over ten correlated series does.

Handling: the declared result is reported exactly as specified in `reports/spc.md`
rather than replaced, the defect is attributed to the declaration rather than to the
data, and a severity sweep is reported in full as a labelled post-hoc diagnostic. The
sweep reports every threshold under two rule sets, so that no single definition is
selected after seeing which one reads best. It finds no association anywhere, smallest
BH adjusted p = 1.000.

What would have been wrong: quietly redefining the flag to rule 1 only, reporting the
resulting p-value, and not mentioning that the original definition was tried first.

## A2. What the permutation null does and does not establish

**Does.** It establishes that the observed survivor count is not what testing 424
correlated features against a label of this class balance produces by chance. The
correlation structure among features is preserved exactly, since only labels are
permuted, so the result does not depend on the dependence assumptions BH itself needs.
Observed 32 against a null mean of 0.25 and a null maximum of 15 over 2000 permutations,
permutation p = 0.0005.

**Does not.** It says nothing about causation. It says nothing about whether the
associations transfer to a later time period, and in this dataset they demonstrably do
not: the same features underpin a holdout PR-AUC whose interval includes no skill. It
does not validate the preprocessing, since the permutation reuses the same imputed
matrix. And it does not establish that any individual survivor is real; it bounds how
many of the 32 the family as a whole could be, not which ones.

The joint reading of a decisive permutation null and a null holdout result is not a
contradiction. It is evidence that the associations are **time local**, which the
monthly failure rate in D0 and the elevated holdout out of control rates in
`reports/spc.md` both independently support.

## A3. Objections consolidated

**Why chronological rather than random.** See D1. Two independent leaks are active
here: tool state is autocorrelated, so a randomly held out record has near neighbours in
train; and the failure rate itself moves by a factor of seven across the window, so a
random fold averages regimes together. A random split does not give a slightly
optimistic estimate of the right quantity, it gives an estimate of a different quantity.
The holdout result in this project is the direct demonstration: performance that a
random split would have reported does not survive an honest time boundary.

**Why class weights rather than SMOTE.** See D9. SMOTE interpolates between minority
neighbours in a 400 dimensional space where the minority class has 78 train members, so
the synthetic points sit in regions no wafer occupies and their density is an artifact of
the sampling scheme. On time ordered data it is worse, since synthesising a record from
neighbours that straddle the split is leakage in its purest form. Class weighting changes
the loss without inventing observations.

**Why PR-AUC rather than accuracy.** See D10. Predicting pass for every lot scores 94.5
percent accuracy on this holdout while catching zero of 26 failures. Accuracy is
uninformative by construction at a 5.5 percent positive rate. ROC-AUC is better but still
dominated by the true negative mass, staying flattering while precision is unusable, so
it is reported and never headlined.

**Why the shortlist is associational.** See D14. The data is observational, from a
running fab, with no factor ever set deliberately. A shortlisted sensor may matter, may
proxy something that matters, or may move with tool age or product mix. The permutation
importance in `reports/model.md` makes the point concretely: it ranks `sensor_000`
first, while the strongest univariate association, `sensor_059`, does not appear in the
top eight. Correlated features and a weak model reshuffle importance freely, which is
exactly why neither ranking can be read as a causal ordering.

**Why the holdout numbers are so wide.** 26 failures. Stated in D1 as an accepted cost at
the time the split was frozen, and reported as an interval everywhere the point estimate
appears rather than being discovered afterwards.

## A4. What would be done differently next time

Three things, recorded because a portfolio artifact that claims no mistakes is less
credible than one that names them.

1. **Check a composite flag's base rate before freezing it.** The D8 defect was
   predictable from the per sensor rates, which were available at Gate 1.
2. **Consider blocking the screen by time period.** The decisive train screen and the
   null holdout result together suggest the associations are time local. Screening
   within period and testing for stability across periods would have measured that
   directly, rather than leaving it to be inferred from three separate observations.
3. **Pre-declare the drift analysis.** The most valuable finding in this project, that
   the process moved and the associations did not transfer, emerged from comparing three
   reports. It deserved to be a planned measurement rather than a synthesis after the
   fact.

## A5. Storage layer added after Gate 1, on 2026-08-23

**Added after Gate 1 and after the results were known. It changed no reported number,
and that claim is enforced by tests rather than asserted.**

The pipeline now reads process data from a local DuckDB warehouse built from the two
raw files, rather than parsing them directly. Queries live in `sql/`.

**Why.** Fab process data lives in a database. A yield analyst's work starts with a
query against a warehouse, not with a file read, so a pipeline that begins with
`read_csv` models the wrong workflow. Reading from a database is a more faithful model
of how this analysis would actually be done.

**DuckDB rather than SQLite.** The access pattern here is an analytical scan across 590
sensor columns, which is what a column store is built for. It installs with pip and
needs no server, so a reviewer can clone the repository and run one command.

**Long form rather than wide.** Measurements are stored as `(record_id, sensor_id,
value)`, one row per cell, 1567 lots by 590 sensors. That is how sensor data is
warehoused, it makes coverage and missingness natural aggregations rather than contorted
column scans, and adding a sensor becomes an insert rather than a schema migration. The
full grid is stored with NULL for unmeasured cells, so the long form round trips back to
the wide matrix exactly. A production warehouse would often omit unmeasured cells
entirely; that would be more compact and would make the round trip lossy, and lossless
was the priority here.

**Division of labour, fixed.** SQL does extraction, filtering and aggregation. Python
does inference. There is no Mann-Whitney U and no BH correction in SQL, because
expressing them there would be a demonstration rather than an improvement.

**What moved into SQL.** Per sensor missing rate and coverage on train; constant and
near constant detection; duplicate detection by value vector fingerprint; the distinct
missingness patterns among sparse sensors; monthly failure rate and monthly per sensor
means; and the train/holdout boundary, which is now a query over a window function
rather than an array slice.

**What did not move, and why.** The fitted `Preprocessor` keeps its pandas
implementation. It is the fitted transform, it must remain independently testable, and
the pipeline must not require a database to apply a stored median. Keeping both is also
what makes the equality test possible: `tests/test_warehouse.py` asserts that the SQL
drop rules produce the same verdict, the same drop reason, and bit identical statistics
for all 590 sensors, and that data read back from the warehouse is bit identical to the
raw parse. The threshold sensitivity grid in `reports/data_quality.md` also stayed in
pandas, since it is exploratory and was not among the computations this change targeted.

**Provenance is unchanged.** The raw file checksums are verified before the warehouse is
built, and values are copied from the parsed frame rather than reparsed from text, so a
value in the database is bit identical to the value the original loader produced. The
database itself is not committed: it is 42 MB of derived data, every byte reproducible
from `data/raw`, and it is rebuilt on demand.

**Evidence that nothing moved.** After the pipeline was switched to read from the
warehouse, `reports/screening.md`, `reports/spc.md`, `reports/model.md`,
`reports/replication.md`, `reports/screening_table.csv` and `reports/doe_proposal.md`
were byte identical to their previous versions. `reports/data_quality.md` gained a new
section on the missingness pattern discovery and its section numbering shifted; every
pre-existing line of it is unchanged.
