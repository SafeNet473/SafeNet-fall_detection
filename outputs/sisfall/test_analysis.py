"""Independent checks of vectorized sweep counts and constrained selection."""
import unittest
import analyze_filtered as a
import numpy as np


class AnalysisTests(unittest.TestCase):
    def test_sweep_against_direct_predictions(self):
        rng = np.random.default_rng(4)
        y = np.r_[np.zeros(70, dtype=int), np.ones(30, dtype=int)]
        scores = rng.integers(0, 20, len(y)).astype(float) / 4
        thresholds = np.r_[-1, np.unique(scores), np.nextafter(scores.max(), np.inf)]
        for row in a.sweep(y, scores, thresholds):
            expected = a.p.metrics(y, scores, row['threshold_g'])
            self.assertEqual([[row['tn'], row['fp']], [row['fn'], row['tp']]], expected['confusion_matrix'])
            for key in ('sensitivity', 'specificity', 'precision', 'f1', 'balanced_accuracy'):
                self.assertAlmostEqual(row[key], expected[key])

    def test_operating_point_ties_and_maximality(self):
        y = np.r_[np.zeros(5, dtype=int), np.ones(20, dtype=int)]
        scores = np.r_[[0, 1, 2, 3, 4], np.repeat(np.arange(10), 2)].astype(float)
        for target in (.95, .90, .85):
            threshold = a.operating_threshold(y, scores, target)
            self.assertGreaterEqual(np.mean(scores[y == 1] >= threshold), target)
            self.assertLess(np.mean(scores[y == 1] >= np.nextafter(threshold, np.inf)), target)


if __name__ == '__main__':
    unittest.main()
