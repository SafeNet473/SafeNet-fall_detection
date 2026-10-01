# SisFall features: equations and explicit protocol adaptations

Source: Sucerquia, López, Vargas-Bonilla, *SisFall: A Fall and Movement Dataset*, Sensors 2017, 17, 198, [Table 4, sections 3.4 and 4.1](https://bibliotecadigital.udea.edu.co/server/api/core/bitstreams/b379330d-b76d-42dc-979f-69364eac3dc6/content), [DOI](https://doi.org/10.3390/s17010198).

The paper defines a trailing window ending at k, uses trapezoidal integration, and reports 200 samples for these dynamic features, sliding with full overlap. Its filter is a fourth-order 5 Hz Butterworth. Table 4 specifies:

| Feature | Paper equation |
|---|---|
| C2 | `sqrt(ax[k]^2 + az[k]^2)` |
| C3 | `RMS(max(a_tilde[k]) - min(a_tilde[k]))` |
| C8 | `sqrt(sigma_x^2 + sigma_z^2)`, with `sigma_i = std(a_tilde_i[k])` |
| C9 | `sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)` |
| C13 | `integral sqrt(ax[n]^2 + az[n]^2) dn`, over window indices |

## Exact implemented scores

Let `a_i[j]` be the **saved filtered** acceleration in g, axis i in x/y/z and j=0,...,199; N=200, fs=200 Hz. Define `h[j] = sqrt(a_x[j]^2 + a_z[j]^2)`, `r_i = max_j a_i[j] - min_j a_i[j]`, `mu_i = sum_j a_i[j]/N`, and `s_i^2 = sum_j (a_i[j]-mu_i)^2/(N-1)`.

| Score | Exact implementation | Window | Filtering | Units |
|---|---|---|---|---|
| baseline | `max_j sqrt(ax[j]^2 + ay[j]^2 + az[j]^2)` | 200 samples | saved 5 Hz filtered XYZ | g |
| C2 | `max_j h[j]` | instantaneous equation, maximum over 200 samples | same | g |
| C3 | `sqrt(r_x^2 + r_y^2 + r_z^2)` | all 200 samples, once | same | g |
| C8 | `sqrt(s_x^2 + s_z^2)` | all 200 samples, once | same | g |
| C9 | `sqrt(s_x^2 + s_y^2 + s_z^2)` | all 200 samples, once | same | g |
| C13 | `sum(j=0..198) (h[j]+h[j+1])/2` | 200 samples / 199 trapezoid intervals | same | g·sample |

C3 is the norm of the **three separate axis ranges**, not the range of acceleration magnitude. C8/C9 are norms of per-axis standard deviations, not standard deviation of magnitude. C13 integrates horizontal Euclidean magnitude, not `abs(x)+abs(z)`, and has no division by N. No gravity removal, orientation estimation, high-pass filter or additional smoothing is introduced. “Horizontal” here means fixed sensor x/z axes, not a dynamically estimated world-horizontal plane after a fall.

## Ambiguities and adaptations (not claims of exact paper reproduction)

1. **RMS naming:** Table 4 calls RMS “Root Mean Square,” but its explicit C1 expression is an unnormalized Euclidean norm. We use that norm convention for C3. Literal RMS of the three ranges would be `C3/sqrt(3)`. Both scores and thresholds can be converted by this factor; do not compare their numerical thresholds without stating the convention.
2. **Standard-deviation normalization:** the paper does not specify the denominator. We explicitly choose sample standard deviation (`ddof=1`). Population standard deviation (`ddof=0`) multiplies C8/C9 by `sqrt(199/200)` for every fixed-length window.
3. **C13 integration scale:** the printed variable is sample index n, while physical integration could use seconds. We follow `dn` literally with trapezoids at `dx=1 sample`. Seconds-based area is `C13/200` g·s. There is no implicit normalization. N samples nominally represent a one-second window; trapezoids between sample centers span 199/200=0.995 seconds. We do not add an invented endpoint. Published numerical thresholds should not be treated as interchangeable with ours.
4. **Window placement and aggregation:** our required protocol takes one preselected event window per fall and stride-100 windows for ADL. C3/C8/C9/C13 each yield one statistic over those exact 200 samples, equivalent to evaluating a trailing feature at the saved window's last sample. We do not calculate a maximum over additional sliding feature windows within or around it. C2 is instantaneous, so taking its maximum is an explicit window-level adaptation matching the existing baseline. This differs from evaluating fully overlapping features over a whole trial and selecting a trial maximum; it may change which part of a fall drives the score.
5. **Filtering phase:** the saved preprocessing uses fourth-order SOS design followed by forward/backward `sosfiltfilt` on each full trial, before extraction. The paper's filter description does not establish that this phase/edge treatment is identical. We preserve the existing noncausal implementation, with doubled effective response order, exactly. No refiltering is performed.

Items 1–3 are positive constant rescalings at fixed N: if thresholds are rescaled consistently, rankings, PR areas and predictions are identical (verified). Items 4–5 can change discrimination; they are protocol limitations, not removable unit conventions.

## Evaluation contract

Use unchanged `processed/baseline_v2/X_filtered.npy`, window metadata and subject partitions, including the existing five-file exclusion. Compute in float64 from saved float32 samples. Scores remain aligned to window IDs. The baseline retains its saved float32-derived scores and threshold; recomputation checks exact parity.

For each feature independently, predict fall when score >= threshold. Select the threshold maximizing validation balanced accuracy over all distinct score boundaries, including no-positive prediction; ties select the smallest threshold, consistent with the baseline. Training data is not needed for a hand-defined scalar score. Do not select a sign, feature, or threshold from test outcomes.

Report both trapezoidal PR-AUC (`auc(recall, precision)`) and average precision (`average_precision_score`), explicitly labeled. Error rates are FP / ADL windows per activity and FN / fall events per activity. Counts, denominators, baseline deltas and individual error windows are retained. Window-weighted metrics are not trial-weighted and do not estimate false alarms per hour.
