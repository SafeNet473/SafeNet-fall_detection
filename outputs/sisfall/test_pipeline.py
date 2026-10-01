import tempfile
import unittest
from pathlib import Path

import pipeline as p
import numpy as np


class PipelineTests(unittest.TestCase):
    def test_ambiguous_subject_audit(self):
        with tempfile.TemporaryDirectory(dir=p.PROJECT / "outputs") as root:
            root = Path(root)
            (root / 'SA15').mkdir()
            (root / 'SE15').mkdir()
            for folder in ('SA15', 'SE15'):
                (root / folder / 'D17_SE15_R01.txt').write_text('1,2,3,4,5,6,7,8,9;')
            audit = []
            rows = p.metadata(root, audit=audit)
            self.assertEqual(len(rows), 1)
            self.assertEqual(audit[0]['action'], 'exclude')
            rows = p.metadata(root, mismatch_policy='folder')
            self.assertEqual({r['subject_id'] for r in rows}, {'SA15', 'SE15'})

    def test_parser_and_conversion(self):
        with tempfile.TemporaryDirectory(dir=p.PROJECT / "outputs") as root:
            root = Path(root)
            file = root / "F05_SA01_R04.txt"
            file.write_text("256,-256,0,1,2,3,4,5,6;\n0,0,512,1,2,3,4,5,6;\n ")
            row = p.metadata(root)[0]
            self.assertEqual((row['subject_id'], row['activity_id'], row['trial_number'], row['label']),
                             ('SA01', 'F05', 4, 1))
            signal, digest = p.load_trial(file)
            np.testing.assert_equal(signal, [[1, -1, 0], [0, 0, 2]])
            self.assertEqual(len(digest), 64)
            file.write_text("1,2,3;")
            with self.assertRaises(ValueError):
                p.load_trial(file)

    def test_windows(self):
        signal = np.zeros((550, 3))
        self.assertEqual(p.window_bounds(signal, 0), [(0, 200, -1), (100, 300, -1),
                                                    (200, 400, -1), (300, 500, -1)])
        for peak, expected in [(0, 0), (250, 150), (549, 350)]:
            signal[:] = 0
            signal[peak, 0] = 10
            self.assertEqual(p.window_bounds(signal, 1), [(expected, expected + 200, peak)])
        with self.assertRaises(ValueError):
            p.window_bounds(signal[:199], 1)
        self.assertEqual(p.window_bounds(signal, 1, event_locator=lambda x, n: (0, n, 4)), [(0, 200, 4)])

    def test_filter_attenuation(self):
        time = np.arange(4000) / 200
        low, high = np.sin(2 * np.pi * time), np.sin(2 * np.pi * 30 * time)
        signal = np.column_stack([low + high, np.ones(len(time)), low])
        result = p.preprocess(signal, "filtered")
        self.assertLess(np.sqrt(np.mean((result[200:-200, 0] - low[200:-200])**2)), .01)
        np.testing.assert_allclose(result[:, 1], 1, atol=1e-12)
        np.testing.assert_equal(p.preprocess(signal, "raw"), signal)

    def test_subject_isolation(self):
        rows = [dict(subject_id=f'SA{i:02}', label=label) for i in range(12) for label in (0, 1)]
        split = p.split_subjects(rows)
        self.assertEqual(split, p.split_subjects(list(reversed(rows))))
        self.assertFalse(set(split['train']) & set(split['test']))
        y = np.array([r['label'] for r in rows])
        groups = np.array([r['subject_id'] for r in rows])
        for method in ('groupkfold', 'loso'):
            seen = []
            for train, test in p.grouped_folds(y, groups, method):
                self.assertFalse(set(groups[train]) & set(groups[test]))
                seen.extend(test)
            self.assertEqual(sorted(seen), list(range(len(y))))
        split['test'].append(split['train'][0])
        with self.assertRaises(ValueError):
            p.validate_split(rows, split)

    def test_threshold_matches_exhaustive_search(self):
        scores = np.array([1, 1, 2, 3, 4, 4, 5.])
        y = np.array([0, 1, 0, 1, 1, 0, 1])
        threshold = p.fit_threshold(scores, y)
        candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        self.assertEqual(p.metrics(y, scores, threshold)['balanced_accuracy'],
                         max(p.metrics(y, scores, t)['balanced_accuracy'] for t in candidates))

    def test_output_protection(self):
        with self.assertRaises(ValueError):
            p.safe_destination(p.PROJECT / 'data' / 'generated', p.PROJECT / 'data')
        with self.assertRaises(ValueError):
            p.safe_destination(p.PROJECT / 'outputs', p.PROJECT / 'data')


if __name__ == '__main__':
    unittest.main()
