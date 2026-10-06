"""Synthetic feature definition, rotation and causal-prefix checks."""
import sys
sys.dont_write_bytecode=True
import unittest
import causal_feature_diversity as d
import numpy as np

def window(a,g):
    a=a.astype(np.float32).astype(float)
    u=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-8)
    p=(a*u).sum(axis=1);b=a-p[:,None]*u
    return np.column_stack((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.abs(p),b,g))

class DiversityTests(unittest.TestCase):
    def test_independent_equations(self):
        rng=np.random.default_rng(8);a=rng.normal(size=(300,3));g=rng.normal(size=(300,3))+[0,1,0]
        w=window(a,g);actual,invalid=d.new_features(w);self.assertFalse(invalid)
        b=w[:,3:6];ga=np.sqrt(sum(sum((float(x)-float(np.mean(b[:,j])))**2 for x in b[:,j])/299 for j in range(3)))
        v=sum((float(x)-float(w[200:,0].mean()))**2 for x in w[200:,0])/100
        ratio=(sum(float(x*x) for x in w[200:,1])/100)/(sum(float(x*x) for x in w[:100,1])/100+1e-6)
        pre=np.mean(g[:100],axis=0);post=np.mean(g[200:],axis=0)
        angle=np.arccos(np.clip(np.dot(pre,post)/(np.linalg.norm(pre)*np.linalg.norm(post)),-1,1))
        np.testing.assert_allclose(actual,[ga,v,ratio,angle],rtol=1e-12,atol=1e-12)
        # Existing ga_C8 uses ddof1; a population result must be rescaled.
        np.testing.assert_allclose(actual[0],np.sqrt(b.var(axis=0,ddof=0).sum())*np.sqrt(300/299))

    def test_degenerate_and_segments(self):
        w=np.zeros((300,9));w[:,0]=1;w[:,7]=1
        np.testing.assert_array_equal(d.new_features(w)[0],[0,0,0,0])
        w[200:300,1]=2
        self.assertEqual(d.new_features(w)[0][2],4/1e-6)
        w[:100,6:9]=0
        values,invalid=d.new_features(w);self.assertTrue(invalid);self.assertEqual(values[3],0.)
        w[:100,6:9]=[0,1,0];w[200:300,6:9]=[1,0,0]
        self.assertAlmostEqual(d.new_features(w)[0][3],np.pi/2)
        before=d.new_features(w)[0].copy();w[100:200,0]=999;w[100:200,1]=999;w[100:200,6:9]=999
        np.testing.assert_array_equal(d.new_features(w)[0][1:],before[1:])

    def test_rotation(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(300,3));g=rng.normal(size=(300,3))+[0,1,0]
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        np.testing.assert_allclose(d.new_features(window(a,g))[0],d.new_features(window(a@q,g@q))[0],rtol=2e-6,atol=2e-6)

    def test_prefix_and_ring(self):
        rng=np.random.default_rng(2);raw=rng.normal(size=(650,3))+[0,1,0]
        def replay(raw):
            f=d.e.c.Frontend(raw[0]);a=[];g=[]
            for r in raw:
                x,y=f.push(r);a.append(x.copy());g.append(y.copy())
            return window(np.array(a),np.array(g))
        full=replay(raw);prefix=replay(raw[:451])
        np.testing.assert_array_equal(d.new_features(full[150:450])[0],d.new_features(prefix[150:450])[0])
        ring=np.empty((301,9))
        for now,row in enumerate(prefix):ring[now%301]=row
        np.testing.assert_array_equal(ring[np.arange(150,450)%301],full[150:450])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiversityTests))
    d.OUT.mkdir(parents=True,exist_ok=True)
    d.dump('feature_tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
