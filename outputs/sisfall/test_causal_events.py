"""Causality, filter-state, boundary, feature and locked-export regression tests."""
import unittest
import causal_events as c
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

class CausalTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.random.default_rng(473).normal(size=(900,3))
        self.raw[:,1]+=1

    def test_sample_filter_and_gravity_match_reference(self):
        f=c.Frontend(self.raw[0]); a=[]; g=[]
        for sample in self.raw:
            x,y=f.push(sample); a.append(x.copy()); g.append(y.copy())
        expected,_=sosfilt(c.SOS,self.raw,axis=0,
                          zi=sosfilt_zi(c.SOS)[:,:,None]*self.raw[0][None,None,:])
        np.testing.assert_allclose(a,expected,rtol=1e-12,atol=1e-12)
        eg=c.of.causal_gravity(self.raw[None],self.raw[:1])[0]
        np.testing.assert_allclose(g,eg,rtol=1e-12,atol=1e-12)

    def test_prefix_invariance(self):
        def run(raw):
            f=c.Frontend(raw[0]); out=[]
            for x in raw:
                a,g=f.push(x); out.append(c.derived(a[None],g[None])[0])
            return np.array(out)
        full=run(self.raw); prefix=run(self.raw[:500])
        np.testing.assert_array_equal(full[:500],prefix)
        for thresholds in c.GRID:
            times=c.triggers(full,*thresholds)
            short=c.triggers(prefix,*thresholds)
            self.assertEqual([t for t in times if t<500],short)
            events=c.ring_windows(full,{'x':times})
            early=c.ring_windows(prefix,{'x':short})
            self.assertEqual([e for e in events if e['decision']<500],early)

    def test_ring_order_boundaries_delay_and_features(self):
        g=c.of.causal_gravity(self.raw[None],self.raw[:1])[0]
        values=c.derived(self.raw,g)
        events=c.ring_windows(values,{'x':[0,75,100,350,700,899]})
        for e in events:
            pre,post=c.DESIGNS[e['design']]
            self.assertEqual(e['decision']-e['trigger'],post)
            self.assertEqual(e['end']-e['start'],pre+post)
            expected=c.feature_row(values[e['start']:e['end']])
            self.assertEqual(e['ga_C2'],expected[15])
            self.assertEqual(e['jerk_abs_mean'],expected[5])
            self.assertEqual(e['ga_parallel_peak'],expected[18])
        self.assertFalse(any(e['trigger'] in (0,899) for e in events))
        a=self.raw[:200].astype(np.float32).astype(float)
        expected=c.of.features(a[None],np.zeros((1,200,3)),g[None,:200])
        f=c.feature_row(values[:200])
        for name,index in [('ga_C2',15),('jerk_abs_mean',5),('ga_parallel_peak',18)]:
            self.assertEqual(f[index],expected[name][0])

    def test_locked_tree_comparison_parity(self):
        x=np.random.default_rng(4).uniform(0,30,(500,21))
        for feature,index,threshold in [('ga_C2',15,.945275604724884),
                                       ('jerk_abs_mean',5,14.055817127227783),
                                       ('ga_parallel_peak',18,3.3697245121002197)]:
            for v in [np.float32(threshold),np.nextafter(np.float32(threshold),np.float32(-np.inf)),
                      np.nextafter(np.float32(threshold),np.float32(np.inf))]:
                row=np.ones(21)*20; row[index]=v; x=np.vstack((x,row))
        np.testing.assert_array_equal(c.predict(x),c.exported_predict(x))

    def test_event_matching_and_duplicates(self):
        trial=dict(trial_index=0,path='x',split='validation',activity_id='F01',label='1',
                   n_samples=1000,proxy_peak=400)
        events=[dict(trial_index=0,trigger=t,decision=t+100,prediction=1) for t in (50,350,500)]
        rows=c.summarize([trial],events,{0:[50,350,500]})
        self.assertEqual(rows[0]['detected'],1)
        self.assertEqual(rows[0]['unmatched_or_duplicate_alarms'],2)
        self.assertEqual(rows[0]['latency_s'],.25)
        self.assertEqual(c.aggregate(rows)['event_precision'],1/3)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CausalTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.dump(c.OUT/'regression_tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),
                                            errors=len(result.errors),passed=result.wasSuccessful(),
                                            replay_code_sha256=c.sha(c.Path(c.__file__))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
