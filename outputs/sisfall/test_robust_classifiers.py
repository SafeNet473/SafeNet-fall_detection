import tempfile
import unittest
from pathlib import Path
import robust_classifiers as r
import numpy as np


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(473)
        cls.train=rng.normal(size=(800,21))*np.arange(1,22)+np.arange(21)
        cls.val=rng.normal(size=(200,21))*np.arange(1,22)+np.arange(21)+.2
        cls.y_train=(cls.train[:,0]+cls.train[:,1]/2>1).astype(int)
        cls.y_val=(cls.val[:,0]+cls.val[:,1]/2>1).astype(int)
        cls.models,cls.candidates=r.fit_models(cls.train,cls.y_train,cls.val,cls.y_val)

    def test_feature_contract(self):
        self.assertEqual(len(r.FEATURE_NAMES),21)
        self.assertTrue(set(r.FEATURE_NAMES)<=r.of.INVARIANT)
        self.assertFalse({'C2','C8'} & set(r.FEATURE_NAMES))

    def test_scaler_training_only(self):
        scaler=self.models['logistic_regression']['scaler']
        np.testing.assert_allclose(scaler.mean_,self.train.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_,len(self.train))

    def test_validation_selection_and_threshold(self):
        self.assertEqual(len(self.candidates),10)
        for family,entry in self.models.items():
            candidates=[row for row in self.candidates if row['family']==family]
            selected=next(row for row in candidates if row['candidate']==entry['candidate'])
            self.assertEqual(selected['balanced_accuracy'],max(row['balanced_accuracy'] for row in candidates))
            self.assertEqual(entry['threshold'],r.p.fit_threshold(r.scores_for(entry,self.val),self.y_val))
        self.assertLessEqual(self.models['decision_tree']['model'].get_depth(),4)
        self.assertLessEqual(self.models['decision_tree']['model'].tree_.node_count,31)

    def test_export_parity(self):
        with tempfile.TemporaryDirectory(dir=r.p.PROJECT/'outputs') as directory:
            params=r.export_models(self.models,Path(directory))
            np.testing.assert_array_equal(r.exported_tree_scores(params['decision_tree'],self.val),
                                          r.scores_for(self.models['decision_tree'],self.val))
            lr=params['logistic_regression']
            actual=self.val@np.array(lr['raw_feature_coefficients'])+lr['raw_feature_intercept']
            entry=self.models['logistic_regression']
            expected=entry['model'].decision_function(entry['scaler'].transform(self.val))
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()
