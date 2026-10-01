import unittest
import orientation_features as f
import numpy as np


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(473)

    def test_rotations_are_proper_orthogonal(self):
        r=f.random_rotations(self.rng,100)
        np.testing.assert_allclose(r@r.transpose(0,2,1),np.broadcast_to(np.eye(3),r.shape),atol=2e-15)
        np.testing.assert_allclose(np.linalg.det(r),1,atol=2e-15)

    def test_causal_gravity_has_no_future_dependency(self):
        a=self.rng.normal(size=(2,200,3))
        initial=self.rng.normal(size=(2,3))
        g=f.causal_gravity(a,initial)
        changed=a.copy()
        changed[:,100:]+=1000
        np.testing.assert_array_equal(g[:,:100],f.causal_gravity(changed,initial)[:,:100])
        expected=initial.copy()
        for i in range(200):
            expected=expected+f.ALPHA*(a[:,i]-expected)
            np.testing.assert_allclose(g[:,i],expected,atol=1e-14)

    def test_rotated_gravity_history_equivariance(self):
        a=self.rng.normal(size=(2,200,3))
        initial=self.rng.normal(size=(2,3))
        r=f.random_rotations(self.rng,2)
        rotated_initial=np.einsum('bij,bj->bi',r,initial)
        np.testing.assert_allclose(f.causal_gravity(f.rotate(a,r),rotated_initial),
                                  f.rotate(f.causal_gravity(a,initial),r),atol=1e-14)

    def test_feature_invariance_and_fixed_plane_failure(self):
        a=self.rng.normal(size=(4,200,3))*[1,5,2]+[0,-1,0]
        gyro=self.rng.normal(size=a.shape)
        g=f.causal_gravity(a,a[:,0])
        r=f.random_rotations(self.rng,4)
        before=f.features(a,gyro,g)
        after=f.features(f.rotate(a,r),f.rotate(gyro,r),f.rotate(g,r))
        for name in f.INVARIANT:
            np.testing.assert_allclose(before[name],after[name],rtol=1e-12,atol=1e-12,err_msg=name)
        for name in ('C2','C8'):
            self.assertGreater(np.max(np.abs(before[name]-after[name])),.01)

    def test_statistics_and_decomposition(self):
        a=np.tile([3.,4.,0.],(1,200,1))
        gravity=np.tile([0.,1.,0.],(1,200,1))
        gyro=np.tile([0.,0.,2.],(1,200,1))
        v=f.features(a,gyro,gravity)
        for name in ('mag_mean','mag_rms','mag_peak'):
            self.assertEqual(v[name][0],5)
        self.assertEqual(v['ga_C2'][0],3)
        self.assertEqual(v['ga_C13'][0],3*199)
        self.assertEqual(v['ga_parallel_peak'][0],4)
        self.assertEqual(v['gyro_peak'][0],2)
        for name in ('mag_std','pre_variance','post_variance','jerk_std','jerk_abs_peak','ga_C8'):
            self.assertEqual(v[name][0],0)

    def test_variance_halves_are_label_independent(self):
        a=np.zeros((1,200,3))
        a[0,:100,0]=np.arange(100)
        a[0,100:,0]=2
        g=np.tile([0.,1.,0.],(1,200,1))
        v=f.features(a,np.zeros_like(a),g)
        self.assertAlmostEqual(v['pre_variance'][0],np.var(np.arange(100),ddof=1))
        self.assertEqual(v['post_variance'][0],0)


if __name__=='__main__':
    unittest.main()
