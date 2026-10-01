"""Individual orientation features; causal gravity from raw acceleration."""
import pipeline as p
import numpy as np
from scipy.signal import lfilter
from paper_features import extract_features

FS = 200
ALPHA = 1 - np.exp(-2*np.pi*.5/FS)
GRAVITY_EPS = 1e-8


def causal_gravity(raw, previous):
    """raw (B,N,3), previous EMA state (B,3), containing no later samples."""
    return lfilter([ALPHA], [1, -(1-ALPHA)], np.asarray(raw, dtype=float), axis=1,
                   zi=(1-ALPHA)*np.asarray(previous, dtype=float)[:, None, :])[0]


def random_rotations(rng, count):
    """Uniform SO(3), via normalized isotropic Gaussian quaternions (w,x,y,z)."""
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    w,x,y,z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=1).reshape(-1,3,3)


def rotate(a, matrices):
    return np.einsum('bij,bnj->bni', matrices, a)


def magnitude_stats(magnitude, prefix):
    return {prefix+'_mean': magnitude.mean(axis=1), prefix+'_std': magnitude.std(axis=1, ddof=1),
            prefix+'_rms': np.sqrt(np.mean(magnitude**2, axis=1)),
            prefix+'_peak': magnitude.max(axis=1), prefix+'_ptp': np.ptp(magnitude, axis=1)}


def features(acc, gyro, gravity):
    acc, gyro, gravity = (np.asarray(v, dtype=np.float64) for v in (acc, gyro, gravity))
    if acc.shape != gyro.shape or acc.shape != gravity.shape or acc.shape[1:] != (200,3):
        raise ValueError('Expected equally shaped (batch,200,3) arrays')
    if not all(np.isfinite(v).all() for v in (acc,gyro,gravity)):
        raise ValueError('Nonfinite signals')
    result = extract_features(acc)
    mag = np.linalg.norm(acc, axis=2)
    result.update(magnitude_stats(mag, 'mag'))
    jerk = np.diff(mag, axis=1)*FS
    result.update(jerk_abs_mean=np.abs(jerk).mean(axis=1), jerk_std=jerk.std(axis=1, ddof=1),
                  jerk_rms=np.sqrt(np.mean(jerk**2, axis=1)), jerk_abs_peak=np.abs(jerk).max(axis=1),
                  pre_variance=mag[:, :100].var(axis=1, ddof=1),
                  post_variance=mag[:, 100:].var(axis=1, ddof=1))
    result.update(magnitude_stats(np.linalg.norm(gyro, axis=2), 'gyro'))
    unit = gravity/np.maximum(np.linalg.norm(gravity, axis=2, keepdims=True), GRAVITY_EPS)
    parallel = np.sum(acc*unit, axis=2)
    residual = acc - parallel[:, :, None]*unit
    perpendicular = np.linalg.norm(residual, axis=2)
    result.update(ga_C2=perpendicular.max(axis=1),
                  ga_C8=np.sqrt(residual.var(axis=1, ddof=1).sum(axis=1)),
                  ga_C13=np.sum((perpendicular[:, :-1]+perpendicular[:, 1:])/2, axis=1),
                  ga_parallel_peak=np.abs(parallel).max(axis=1),
                  ga_parallel_std=parallel.std(axis=1, ddof=1),
                  ga_parallel_ptp=np.ptp(parallel, axis=1))
    result['baseline'] = mag.max(axis=1)
    return result


INVARIANT = {'C9', 'baseline', 'mag_mean','mag_std','mag_rms','mag_peak','mag_ptp',
             'jerk_abs_mean','jerk_std','jerk_rms','jerk_abs_peak','pre_variance','post_variance',
             'gyro_mean','gyro_std','gyro_rms','gyro_peak','gyro_ptp',
             'ga_C2','ga_C8','ga_C13','ga_parallel_peak','ga_parallel_std','ga_parallel_ptp'}
