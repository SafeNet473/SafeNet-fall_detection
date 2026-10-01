import unittest
import paper_features as f
import numpy as np


class FeatureTests(unittest.TestCase):
    def test_constant_horizontal_vector(self):
        a = np.tile([3., 7., 4.], (1, 200, 1))
        scores = f.extract_features(a)
        self.assertEqual(scores['C2'][0], 5.)
        self.assertEqual(scores['C13'][0], 5*199)
        for name in ('C3', 'C8', 'C9'):
            self.assertEqual(scores[name][0], 0.)

    def test_axis_ranges_sample_variance_and_trapezoids(self):
        a = np.zeros((1, 200, 3))
        a[0, 0] = [3, 4, 0]
        a[0, -1] = [0, 0, 12]
        s = f.extract_features(a)
        self.assertEqual(s['C2'][0], 12.)
        self.assertEqual(s['C3'][0], 13.)
        # One nonzero sample b among N gives sample variance b^2/N.
        self.assertAlmostEqual(s['C8'][0], np.sqrt(153/200))
        self.assertAlmostEqual(s['C9'][0], np.sqrt(169/200))
        self.assertEqual(s['C13'][0], 7.5)

    def test_horizontal_features_ignore_y_and_std_ignores_offset(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(3, 200, 3))
        original = f.extract_features(a)
        a[:, :, 1] *= 100
        modified = f.extract_features(a)
        for name in ('C2', 'C8', 'C13'):
            np.testing.assert_array_equal(original[name], modified[name])
        shifted = f.extract_features(a + [10, 20, 30])
        for name in ('C3', 'C8', 'C9'):
            np.testing.assert_allclose(modified[name], shifted[name], atol=1e-12)

    def test_invalid_window(self):
        for a in (np.zeros((2, 199, 3)), np.full((1, 200, 3), np.nan)):
            with self.assertRaises(ValueError):
                f.extract_features(a)

    def test_threshold_only_uses_supplied_validation(self):
        scores = np.array([0., 1., 2., 3.])
        y = np.array([0, 0, 1, 1])
        threshold = f.p.fit_threshold(scores, y)
        self.assertEqual(threshold, 2.)
        result = f.evaluate(y, scores, threshold)
        self.assertEqual(result['balanced_accuracy'], 1.)
        self.assertEqual(result['pr_auc_trapezoidal'], 1.)
        self.assertEqual(result['average_precision'], 1.)


if __name__ == '__main__':
    unittest.main()
