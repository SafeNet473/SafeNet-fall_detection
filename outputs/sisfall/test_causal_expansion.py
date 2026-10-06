"""New-feature unit checks, independent of any test-subject data."""
import sys
sys.dont_write_bytecode=True
import unittest
import causal_feature_expansion as e
import numpy as np

class FeatureTests(unittest.TestCase):
    def test_moments_and_segment_edges(self):
        rng=np.random.default_rng(473);v=np.abs(rng.normal(size=(300,3)));p=rng.normal(size=300)
        expected=e.additions(v,p)
        moments=lambda x:sum(float(a*a) for a in x)/len(x)-(sum(float(a) for a in x)/len(x))**2
        jerk=lambda x:200/99*sum(abs(float(x[i])-float(x[i-1])) for i in range(1,100))
        actual=[moments(v[200:,0]),jerk(v[200:,0])-jerk(v[:100,0]),
                sum(float(a)>.5 for a in v[:,1])/300,np.sqrt(moments(p[200:]))]
        np.testing.assert_allclose(expected,actual,rtol=1e-12,atol=1e-12)
        # Exclude differences into/out of pre and late segments.
        constant=np.ones((300,3));constant[100:200,0]=10000
        self.assertEqual(e.additions(constant,np.ones(300))[1],0.)

    def test_threshold_and_constant_signal(self):
        v=np.ones((300,3));v[:,1]=.5
        np.testing.assert_array_equal(e.additions(v,np.ones(300)),[0,0,0,0])
        v[0,1]=np.nextafter(.5,1.)
        self.assertEqual(e.additions(v,np.ones(300))[2],1/300)

    def test_constant_rotation_robustness(self):
        rng=np.random.default_rng(11);a=rng.normal(size=(300,3));g=rng.normal(size=(300,3))
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        def get(a,g):
            values=e.c.derived(a,g)
            aa=a.astype(np.float32).astype(float)
            u=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-8)
            return e.additions(values,(aa*u).sum(axis=1))
        np.testing.assert_allclose(get(a,g),get(a@q,g@q),rtol=2e-6,atol=2e-6)

    def test_causal_prefix(self):
        rng=np.random.default_rng(9);raw=rng.normal(size=(600,3));raw[:,1]+=1
        def replay(data):
            f=e.c.Frontend(data[0]);a=[];g=[]
            for row in data:
                x,y=f.push(row);a.append(x.copy());g.append(y.copy())
            a=np.array(a);g=np.array(g);aa=a.astype(np.float32).astype(float)
            u=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-8)
            return e.c.derived(a,g),(aa*u).sum(axis=1)
        a,p=replay(raw);b,q=replay(raw[:401])
        np.testing.assert_array_equal(e.additions(a[100:400],p[100:400]),e.additions(b[100:400],q[100:400]))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FeatureTests))
    e.OUT.mkdir(parents=True,exist_ok=True)
    e.dump('feature_tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
