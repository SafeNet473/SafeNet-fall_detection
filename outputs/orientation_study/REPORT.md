# Orientation robustness: fixed waist data, synthetic coordinate rotations

This is not wrist-worn validation. Synthetic constant rotations change coordinates, not sensor location, wrist articulation, impacts, or within-window time-varying orientation. No features are combined or ML models trained. [Equations, causality and MCU costs](../sisfall/ORIENTATION_METHODS.md).

## 1. Unchanged waist setup

`paper_results_unchanged.csv` is a byte-for-byte copy of the previous paper-feature results. Paper thresholds and original metrics were checked for exact equality. All new features use the same windows and subject split; thresholds maximize unrotated validation balanced accuracy.

### validation

| feature | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal |
| --- | --- | --- | --- | --- | --- | --- |
| C2 | 98.67% | 98.33% | 98.50% | 50.82% | 67.09% | 97.05% |
| C3 | 97.07% | 92.24% | 94.65% | 17.93% | 30.27% | 63.29% |
| C8 | 98.67% | 98.42% | 98.54% | 52.19% | 68.27% | 92.72% |
| C9 | 97.07% | 89.58% | 93.32% | 14.00% | 24.47% | 26.68% |
| C13 | 97.07% | 86.74% | 91.90% | 11.34% | 20.31% | 17.86% |
| mag_mean | 91.47% | 80.22% | 85.85% | 7.48% | 13.83% | 8.47% |
| mag_std | 96.80% | 81.22% | 89.01% | 8.27% | 15.23% | 17.28% |
| mag_rms | 98.67% | 80.37% | 89.52% | 8.08% | 14.93% | 10.82% |
| mag_peak | 93.07% | 90.52% | 91.79% | 14.64% | 25.30% | 58.29% |
| mag_ptp | 97.60% | 81.34% | 89.47% | 8.38% | 15.43% | 50.80% |
| jerk_abs_mean | 97.07% | 80.39% | 88.73% | 7.96% | 14.72% | 5.23% |
| jerk_std | 98.40% | 79.81% | 89.11% | 7.85% | 14.54% | 8.50% |
| jerk_rms | 98.40% | 79.78% | 89.09% | 7.84% | 14.52% | 8.46% |
| jerk_abs_peak | 97.33% | 80.97% | 89.15% | 8.21% | 15.14% | 31.56% |
| pre_variance | 95.20% | 79.53% | 87.37% | 7.52% | 13.94% | 16.39% |
| post_variance | 96.53% | 79.43% | 87.98% | 7.58% | 14.06% | 13.48% |
| gyro_mean | 95.73% | 96.07% | 95.90% | 29.87% | 45.53% | 56.89% |
| gyro_std | 94.93% | 95.79% | 95.36% | 28.25% | 43.55% | 56.70% |
| gyro_rms | 95.20% | 97.34% | 96.27% | 38.47% | 54.80% | 60.24% |
| gyro_peak | 94.40% | 96.39% | 95.39% | 31.36% | 47.07% | 59.05% |
| gyro_ptp | 97.07% | 92.65% | 94.86% | 18.76% | 31.45% | 56.36% |
| ga_C2 | 99.73% | 99.62% | 99.68% | 82.02% | 90.01% | 99.39% |
| ga_C8 | 99.47% | 95.15% | 97.31% | 26.38% | 41.70% | 72.10% |
| ga_C13 | 98.67% | 95.93% | 97.30% | 29.74% | 45.71% | 66.23% |
| ga_parallel_peak | 94.67% | 81.55% | 88.11% | 8.23% | 15.14% | 44.22% |
| ga_parallel_std | 96.53% | 79.31% | 87.92% | 7.54% | 13.99% | 6.75% |
| ga_parallel_ptp | 97.60% | 79.90% | 88.75% | 7.82% | 14.49% | 29.37% |
| baseline | 93.07% | 90.52% | 91.79% | 14.64% | 25.30% | 58.29% |

### test

| feature | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal |
| --- | --- | --- | --- | --- | --- | --- |
| C2 | 95.20% | 95.26% | 95.23% | 25.55% | 40.29% | 92.00% |
| C3 | 91.47% | 90.90% | 91.18% | 14.66% | 25.28% | 71.05% |
| C8 | 95.73% | 98.26% | 97.00% | 48.45% | 64.34% | 95.53% |
| C9 | 86.13% | 86.88% | 86.51% | 10.10% | 18.07% | 44.26% |
| C13 | 93.07% | 85.37% | 89.22% | 9.81% | 17.75% | 25.21% |
| mag_mean | 84.53% | 75.77% | 80.15% | 5.63% | 10.56% | 11.17% |
| mag_std | 92.80% | 81.44% | 87.12% | 7.88% | 14.52% | 18.53% |
| mag_rms | 95.73% | 79.46% | 87.59% | 7.38% | 13.71% | 16.36% |
| mag_peak | 90.40% | 89.53% | 89.96% | 12.87% | 22.52% | 61.96% |
| mag_ptp | 94.67% | 82.01% | 88.34% | 8.26% | 15.19% | 50.15% |
| jerk_abs_mean | 93.60% | 80.70% | 87.15% | 7.66% | 14.16% | 4.90% |
| jerk_std | 96.27% | 80.05% | 88.16% | 7.62% | 14.13% | 8.29% |
| jerk_rms | 96.27% | 80.02% | 88.14% | 7.61% | 14.11% | 8.20% |
| jerk_abs_peak | 95.47% | 82.01% | 88.74% | 8.32% | 15.31% | 45.09% |
| pre_variance | 97.33% | 79.49% | 88.41% | 7.51% | 13.94% | 19.91% |
| post_variance | 93.33% | 79.37% | 86.35% | 7.18% | 13.34% | 13.26% |
| gyro_mean | 89.87% | 95.26% | 92.56% | 24.47% | 38.47% | 70.36% |
| gyro_std | 92.27% | 96.87% | 94.57% | 33.49% | 49.15% | 84.04% |
| gyro_rms | 90.67% | 97.24% | 93.95% | 35.94% | 51.48% | 80.38% |
| gyro_peak | 92.00% | 96.77% | 94.39% | 32.76% | 48.32% | 87.43% |
| gyro_ptp | 94.93% | 93.87% | 94.40% | 20.95% | 34.33% | 85.46% |
| ga_C2 | 96.00% | 99.93% | 97.96% | 95.74% | 95.87% | 98.09% |
| ga_C8 | 93.60% | 97.83% | 95.72% | 42.49% | 58.45% | 92.13% |
| ga_C13 | 92.27% | 98.54% | 95.41% | 52.03% | 66.54% | 92.72% |
| ga_parallel_peak | 93.60% | 81.43% | 87.52% | 7.94% | 14.63% | 44.37% |
| ga_parallel_std | 93.60% | 79.20% | 86.40% | 7.15% | 13.28% | 6.22% |
| ga_parallel_ptp | 96.53% | 80.22% | 88.38% | 7.71% | 14.27% | 34.18% |
| baseline | 90.40% | 89.53% | 89.96% | 12.87% | 22.52% | 61.96% |

## 2. Synthetic orientation stress test

10 independent random SO(3) rotations per window, plus 90-degree y-axis yaw and x-axis tilt controls. All thresholds stay fixed. Replicate ranges describe rotation randomness, not subject-level confidence intervals. PR-AUC is trapezoidal, with AP separately in metrics.csv.

### validation

| feature | original_BA | random_BA_mean | random_BA_range | random_PR_AUC |
| --- | --- | --- | --- | --- |
| C2 | 98.50% | 84.42% | 84.10%–84.75% | 0.4248 |
| C3 | 94.65% | 94.35% | 94.11%–94.54% | 0.5842 |
| C8 | 98.54% | 91.29% | 90.94%–91.64% | 0.2041 |
| C9 | 93.32% | 93.32% | 93.32%–93.32% | 0.2668 |
| C13 | 91.90% | 58.30% | 57.45%–59.04% | 0.0301 |
| mag_mean | 85.85% | 85.85% | 85.85%–85.85% | 0.0847 |
| mag_std | 89.01% | 89.01% | 89.01%–89.01% | 0.1728 |
| mag_rms | 89.52% | 89.52% | 89.52%–89.52% | 0.1082 |
| mag_peak | 91.79% | 91.79% | 91.79%–91.79% | 0.5829 |
| mag_ptp | 89.47% | 89.47% | 89.47%–89.47% | 0.5080 |
| jerk_abs_mean | 88.73% | 88.73% | 88.73%–88.73% | 0.0523 |
| jerk_std | 89.11% | 89.11% | 89.11%–89.11% | 0.0850 |
| jerk_rms | 89.09% | 89.09% | 89.09%–89.09% | 0.0846 |
| jerk_abs_peak | 89.15% | 89.15% | 89.15%–89.15% | 0.3156 |
| pre_variance | 87.37% | 87.37% | 87.37%–87.37% | 0.1639 |
| post_variance | 87.98% | 87.98% | 87.98%–87.98% | 0.1348 |
| gyro_mean | 95.90% | 95.90% | 95.90%–95.90% | 0.5689 |
| gyro_std | 95.36% | 95.36% | 95.36%–95.36% | 0.5670 |
| gyro_rms | 96.27% | 96.27% | 96.27%–96.27% | 0.6024 |
| gyro_peak | 95.39% | 95.39% | 95.39%–95.39% | 0.5905 |
| gyro_ptp | 94.86% | 94.86% | 94.86%–94.86% | 0.5636 |
| ga_C2 | 99.68% | 99.68% | 99.68%–99.68% | 0.9939 |
| ga_C8 | 97.31% | 97.31% | 97.31%–97.31% | 0.7210 |
| ga_C13 | 97.30% | 97.30% | 97.30%–97.30% | 0.6623 |
| ga_parallel_peak | 88.11% | 88.11% | 88.11%–88.11% | 0.4422 |
| ga_parallel_std | 87.92% | 87.92% | 87.92%–87.92% | 0.0675 |
| ga_parallel_ptp | 88.75% | 88.75% | 88.75%–88.75% | 0.2937 |
| baseline | 91.79% | 91.79% | 91.79%–91.79% | 0.5829 |

### test

| feature | original_BA | random_BA_mean | random_BA_range | random_PR_AUC |
| --- | --- | --- | --- | --- |
| C2 | 95.23% | 83.42% | 82.74%–84.28% | 0.4483 |
| C3 | 91.18% | 90.35% | 89.66%–90.82% | 0.6833 |
| C8 | 97.00% | 90.86% | 90.05%–91.79% | 0.2670 |
| C9 | 86.51% | 86.51% | 86.51%–86.51% | 0.4426 |
| C13 | 89.22% | 57.56% | 56.28%–58.40% | 0.0326 |
| mag_mean | 80.15% | 80.15% | 80.15%–80.15% | 0.1117 |
| mag_std | 87.12% | 87.12% | 87.12%–87.12% | 0.1853 |
| mag_rms | 87.59% | 87.59% | 87.59%–87.59% | 0.1636 |
| mag_peak | 89.96% | 89.96% | 89.96%–89.96% | 0.6196 |
| mag_ptp | 88.34% | 88.34% | 88.34%–88.34% | 0.5015 |
| jerk_abs_mean | 87.15% | 87.15% | 87.15%–87.15% | 0.0490 |
| jerk_std | 88.16% | 88.16% | 88.16%–88.16% | 0.0829 |
| jerk_rms | 88.14% | 88.14% | 88.14%–88.14% | 0.0820 |
| jerk_abs_peak | 88.74% | 88.74% | 88.74%–88.74% | 0.4509 |
| pre_variance | 88.41% | 88.41% | 88.41%–88.41% | 0.1991 |
| post_variance | 86.35% | 86.35% | 86.35%–86.35% | 0.1326 |
| gyro_mean | 92.56% | 92.56% | 92.56%–92.56% | 0.7036 |
| gyro_std | 94.57% | 94.57% | 94.57%–94.57% | 0.8404 |
| gyro_rms | 93.95% | 93.95% | 93.95%–93.95% | 0.8038 |
| gyro_peak | 94.39% | 94.39% | 94.39%–94.39% | 0.8743 |
| gyro_ptp | 94.40% | 94.40% | 94.40%–94.40% | 0.8546 |
| ga_C2 | 97.96% | 97.96% | 97.96%–97.96% | 0.9809 |
| ga_C8 | 95.72% | 95.72% | 95.72%–95.72% | 0.9213 |
| ga_C13 | 95.41% | 95.41% | 95.41%–95.41% | 0.9272 |
| ga_parallel_peak | 87.52% | 87.52% | 87.52%–87.52% | 0.4437 |
| ga_parallel_std | 86.40% | 86.40% | 86.40%–86.40% | 0.0622 |
| ga_parallel_ptp | 88.38% | 88.38% | 88.38%–88.38% | 0.3418 |
| baseline | 89.96% | 89.96% | 89.96%–89.96% | 0.6196 |

## 3. Axis-assumption diagnostic

| feature | scenario | sensitivity | specificity | BA |
| --- | --- | --- | --- | --- |
| C2 | original | 95.20% | 95.26% | 95.23% |
| C8 | original | 95.73% | 98.26% | 97.00% |
| C2 | yaw_y_90 | 95.20% | 95.26% | 95.23% |
| C8 | yaw_y_90 | 95.73% | 98.26% | 97.00% |
| C2 | tilt_x_90 | 94.13% | 61.01% | 77.57% |
| C8 | tilt_x_90 | 96.80% | 82.33% | 89.56% |

Y-axis yaw preserves the x/z plane. X-axis tilt mixes vertical and horizontal components. Changes under tilt and arbitrary rotations, with yaw invariance, identify dependence on the original horizontal-plane assumption. C9 is already rotation invariant (trace of the 3-axis covariance); C3 axis-range norm is not generally invariant.

## 4. Requested activity failures

### validation: C2

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 13/1592 (0.8%) | 80.5% | 78.8%–82.2% |
| D04 | 56/1592 (3.5%) | 87.7% | 87.1%–89.0% |
| D06 | 0/1219 (0.0%) | 26.3% | 25.7%–26.8% |
| D18 | 44/573 (7.7%) | 34.2% | 30.0%–36.6% |
| D19 | 17/575 (3.0%) | 27.3% | 25.9%–28.3% |
| F13 | 1/25 (4.0%) | 9.6% | 4.0%–16.0% |
| F14 | 0/25 (0.0%) | 5.2% | 0.0%–12.0% |
| F15 | 0/25 (0.0%) | 6.0% | 0.0%–12.0% |

### validation: C8

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 1/1592 (0.1%) | 76.1% | 75.4%–77.1% |
| D04 | 226/1592 (14.2%) | 89.9% | 89.6%–90.3% |
| D06 | 1/1219 (0.1%) | 21.0% | 20.6%–21.5% |
| D18 | 25/573 (4.4%) | 9.2% | 8.0%–10.5% |
| D19 | 2/575 (0.3%) | 20.1% | 19.5%–20.9% |
| F13 | 4/25 (16.0%) | 10.8% | 8.0%–20.0% |
| F14 | 0/25 (0.0%) | 3.2% | 0.0%–8.0% |
| F15 | 0/25 (0.0%) | 0.8% | 0.0%–4.0% |

### validation: mag_std

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 1559/1592 (97.9%) | 97.9% | 97.9%–97.9% |
| D04 | 1567/1592 (98.4%) | 98.4% | 98.4%–98.4% |
| D06 | 300/1219 (24.6%) | 24.6% | 24.6%–24.6% |
| D18 | 71/573 (12.4%) | 12.4% | 12.4%–12.4% |
| D19 | 145/575 (25.2%) | 25.2% | 25.2%–25.2% |
| F13 | 6/25 (24.0%) | 24.0% | 24.0%–24.0% |
| F14 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F15 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |

### validation: mag_peak

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 345/1592 (21.7%) | 21.7% | 21.7%–21.7% |
| D04 | 1203/1592 (75.6%) | 75.6% | 75.6%–75.6% |
| D06 | 269/1219 (22.1%) | 22.1% | 22.1%–22.1% |
| D18 | 52/573 (9.1%) | 9.1% | 9.1%–9.1% |
| D19 | 153/575 (26.6%) | 26.6% | 26.6%–26.6% |
| F13 | 10/25 (40.0%) | 40.0% | 40.0%–40.0% |
| F14 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F15 | 1/25 (4.0%) | 4.0% | 4.0%–4.0% |

### validation: ga_C2

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 2/1592 (0.1%) | 0.1% | 0.1%–0.1% |
| D04 | 23/1592 (1.4%) | 1.4% | 1.4%–1.4% |
| D06 | 24/1219 (2.0%) | 2.0% | 2.0%–2.0% |
| D18 | 10/573 (1.7%) | 1.7% | 1.7%–1.7% |
| D19 | 0/575 (0.0%) | 0.0% | 0.0%–0.0% |
| F13 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F14 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F15 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |

### validation: ga_C8

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 100/1592 (6.3%) | 6.3% | 6.3%–6.3% |
| D04 | 757/1592 (47.6%) | 47.6% | 47.6%–47.6% |
| D06 | 108/1219 (8.9%) | 8.9% | 8.9%–8.9% |
| D18 | 16/573 (2.8%) | 2.8% | 2.8%–2.8% |
| D19 | 9/575 (1.6%) | 1.6% | 1.6%–1.6% |
| F13 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F14 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |
| F15 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |

### test: C2

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 304/1751 (17.4%) | 80.8% | 79.6%–82.2% |
| D04 | 605/1751 (34.6%) | 89.0% | 87.5%–90.1% |
| D06 | 13/1174 (1.1%) | 25.9% | 24.4%–26.5% |
| D18 | 32/574 (5.6%) | 33.1% | 30.1%–36.1% |
| D19 | 0/574 (0.0%) | 25.9% | 24.4%–27.5% |
| F13 | 1/25 (4.0%) | 9.6% | 4.0%–16.0% |
| F14 | 0/25 (0.0%) | 9.6% | 4.0%–20.0% |
| F15 | 0/25 (0.0%) | 9.6% | 0.0%–20.0% |

### test: C8

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 8/1751 (0.5%) | 71.9% | 70.8%–73.1% |
| D04 | 305/1751 (17.4%) | 91.3% | 90.4%–92.6% |
| D06 | 18/1174 (1.5%) | 18.8% | 17.7%–19.5% |
| D18 | 18/574 (3.1%) | 8.7% | 8.2%–9.6% |
| D19 | 0/574 (0.0%) | 18.5% | 16.9%–19.5% |
| F13 | 1/25 (4.0%) | 9.6% | 4.0%–16.0% |
| F14 | 6/25 (24.0%) | 14.8% | 4.0%–24.0% |
| F15 | 2/25 (8.0%) | 6.8% | 0.0%–16.0% |

### test: mag_std

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 1639/1751 (93.6%) | 93.6% | 93.6%–93.6% |
| D04 | 1742/1751 (99.5%) | 99.5% | 99.5%–99.5% |
| D06 | 289/1174 (24.6%) | 24.6% | 24.6%–24.6% |
| D18 | 66/574 (11.5%) | 11.5% | 11.5%–11.5% |
| D19 | 150/574 (26.1%) | 26.1% | 26.1%–26.1% |
| F13 | 6/25 (24.0%) | 24.0% | 24.0%–24.0% |
| F14 | 3/25 (12.0%) | 12.0% | 12.0%–12.0% |
| F15 | 5/25 (20.0%) | 20.0% | 20.0%–20.0% |

### test: mag_peak

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 427/1751 (24.4%) | 24.4% | 24.4%–24.4% |
| D04 | 1476/1751 (84.3%) | 84.3% | 84.3%–84.3% |
| D06 | 193/1174 (16.4%) | 16.4% | 16.4%–16.4% |
| D18 | 37/574 (6.4%) | 6.4% | 6.4%–6.4% |
| D19 | 151/574 (26.3%) | 26.3% | 26.3%–26.3% |
| F13 | 6/25 (24.0%) | 24.0% | 24.0%–24.0% |
| F14 | 5/25 (20.0%) | 20.0% | 20.0%–20.0% |
| F15 | 6/25 (24.0%) | 24.0% | 24.0%–24.0% |

### test: ga_C2

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 0/1751 (0.0%) | 0.0% | 0.0%–0.0% |
| D04 | 0/1751 (0.0%) | 0.0% | 0.0%–0.0% |
| D06 | 0/1174 (0.0%) | 0.0% | 0.0%–0.0% |
| D18 | 8/574 (1.4%) | 1.4% | 1.4%–1.4% |
| D19 | 0/574 (0.0%) | 0.0% | 0.0%–0.0% |
| F13 | 2/25 (8.0%) | 8.0% | 8.0%–8.0% |
| F14 | 2/25 (8.0%) | 8.0% | 8.0%–8.0% |
| F15 | 0/25 (0.0%) | 0.0% | 0.0%–0.0% |

### test: ga_C8

| activity | original | random_mean | random_range |
| --- | --- | --- | --- |
| D03 | 0/1751 (0.0%) | 0.0% | 0.0%–0.0% |
| D04 | 450/1751 (25.7%) | 25.7% | 25.7%–25.7% |
| D06 | 6/1174 (0.5%) | 0.5% | 0.5%–0.5% |
| D18 | 7/574 (1.2%) | 1.2% | 1.2%–1.2% |
| D19 | 1/574 (0.2%) | 0.2% | 0.2%–0.2% |
| F13 | 2/25 (8.0%) | 8.0% | 8.0%–8.0% |
| F14 | 4/25 (16.0%) | 16.0% | 16.0%–16.0% |
| F15 | 5/25 (20.0%) | 20.0% | 20.0%–20.0% |

## Files and reproduction

`metrics.csv`: every feature, split, orientation and metric including deltas. `rotation_summary.csv`: all metric means/ranges/std. `activity_errors.csv`: all ADL/fall rates, denominators and error-count deltas. `target_activity_summary.csv`: requested codes for every feature. `numerical_invariance.csv`: measured score discrepancies before roundoff canonicalization. `thresholds.json`: locked thresholds. Scores align to `window_ids.npy`.

Run `python outputs/sisfall/orientation_study.py --out outputs/orientation_study_rerun` from the project root. Run `python outputs/sisfall/test_orientation.py` for causality, SO(3), decomposition and invariance tests.

Existing zero-phase filtering and event-centered fall selection remain offline. Only gravity estimation is causal. An MCU implementation needs a separate causal-filter/trigger experiment; runtime estimates do not claim bit-exact reproduction of the offline filter. Gravity-aligned invariance assumes the gravity state is expressed in the same frame as the signal. Rapid wrist rotation and dynamic acceleration can invalidate gravity estimates; these tests do not simulate either.
