# Designed experiment: a proposal that was not executed

**Status: proposal only.** Nothing in this document was run. It is included because the rest of this repository can only measure association, and it is more useful to state precisely what would settle the question than to imply that a ranking of associations has already settled it.

## 1. Why an experiment is required at all

Every result in this repository comes from observational records of a running process. Sensors move together, the process drifts over the collection window, and no factor was ever set deliberately. Under those conditions a strong association is consistent with several explanations that no amount of further analysis can separate: the sensor matters; the sensor proxies something else that matters; or both move with a third thing such as tool age or product mix. Randomised assignment is what breaks that tie, because it makes the factor independent of everything not measured.

## 2. Factors

The three shortlisted sensor values with the largest absolute effect size from the screen. Missingness indicators are excluded, since they describe metrology routing rather than a settable process parameter.

| factor | sensor | screen effect size | BH adjusted p | low level | high level |
| --- | --- | --- | --- | --- | --- |
| F1 | sensor_059 | 0.5125 | 0.0000 | train 25th percentile | train 75th percentile |
| F2 | sensor_477 | 0.3054 | 0.0011 | train 25th percentile | train 75th percentile |
| F3 | sensor_205 | 0.3031 | 0.0011 | train 25th percentile | train 75th percentile |

Levels are set at the train 25th and 75th percentiles rather than at extremes. Extreme levels would raise the power of the design but would run the process outside its normal operating window, which risks scrapping material and answers a question about a process nobody runs.


## 3. Design

| property | value |
| --- | --- |
| design | 2^3 full factorial |
| factors | 3 |
| levels per factor | 2 |
| distinct corners | 8 |
| randomisation | run order randomised within each day |
| blocking | block by day, since the drift measured in DECISIONS.md D0 is large enough to swamp the effect being tested |
| response | lot pass or fail, binary |
| baseline failure rate | 0.0664 |
| significance level | 0.0500 |
| target power | 0.8000 |

A full factorial rather than a fractional design: with only three factors the full design costs eight corners rather than four, and it buys clean two factor interactions. Interactions are the plausible mechanism here, since these sensors sit on correlated process steps, so aliasing them away to save four corners would be a poor trade.


## 4. Run count, and why this is the important section

The response is binary and the baseline failure rate is low, which makes this expensive. Sample size per level is computed from a two sided two proportion test at alpha = 0.05 and power = 0.8. In a two level factorial, each main effect is estimated by comparing half the runs against the other half, so the total run count is twice the per level requirement.

| effect to detect | treatment fail rate | lots per level | total lots in the design |
| --- | --- | --- | --- |
| halved, 6.6 to 3.3 percent | 0.0332 | 673 | 1346 |
| cut by a third, 6.6 to 4.4 percent | 0.0443 | 1674 | 3348 |
| cut by a quarter, 6.6 to 5.0 percent | 0.0498 | 3117 | 6234 |
| doubled, 6.6 to 13.3 percent | 0.1328 | 319 | 638 |
| raised by half, 6.6 to 10.0 percent | 0.0996 | 1083 | 2166 |

**Detecting a halving of the failure rate needs about 1346 lots.** That is the honest headline of this proposal. It is a large number, and it is the real reason observational screening studies like this one exist: the experiment that would answer the question properly costs more than most fabs will spend on a hypothesis that is still just a ranking.


Two consequences follow, and both are arguments for using the screen rather than against it. First, the screen is worth running precisely because it narrows 590 sensors to a handful, and it is far cheaper to buy 1346 lots of evidence about 3 candidates than to attempt anything comparable across 590. Second, the run counts above assume the effect is real and of the stated size; the holdout result in `reports/model.md` gives no reason to be confident of that, so a smaller pilot to confirm the effect exists at all should come before committing to the full design.


## 5. What this design would and would not establish

It would establish whether deliberately setting these three parameters changes the failure rate, within the operating window tested, on the tool and product mix used, during the period it runs.


It would not establish a mechanism, would not extrapolate outside the tested window, and would not transfer automatically to another tool or another product. It also would not rescue any sensor that this repository shortlisted for the wrong reason: if a shortlisted sensor is a proxy, the experiment will correctly find that setting it changes nothing, which is a useful and cheap way to be wrong.
