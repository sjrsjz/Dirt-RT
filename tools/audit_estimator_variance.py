"""Direct estimator-variance trial: CPU algebra, FP16, and overlap checks.

Run python -B tools/audit_estimator_variance.py [--compile].
This does not measure in-game quality or validate the g^-3 observation closure.
"""
from pathlib import Path
import argparse
import json
import math
import re
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = -2.


def propagate(sigmas, weights, correlation):
    s = np.asarray(sigmas, dtype=float)
    w = np.asarray(weights, dtype=float)
    active = np.isfinite(w) & (w > 0)
    s, w = s[active], w[active]
    known = np.isfinite(s) & (s >= 0) & (s <= 65504)
    if not np.any(known):
        return UNKNOWN
    donor = np.dot(w[known], s[known]) / w[known].sum()
    s = np.where(known, s, donor)
    a = w / w.sum()
    v = (1-correlation)*np.dot(a*a, s*s) + correlation*np.dot(a, s)**2
    return math.sqrt(v) if 0 <= v <= 65504**2 else UNKNOWN


def tests():
    rng = np.random.default_rng(906202631)
    max_relative_error = 0.
    for _ in range(10000):
        s = np.exp(rng.uniform(-8, 8, 9))
        w = rng.uniform(0, 1, 9)
        a = w/w.sum()
        p = rng.uniform(0, 1)
        covariance = (1-p)*np.diag(s*s) + p*np.outer(s, s)
        reference = a @ covariance @ a
        actual = propagate(s, w, p)**2
        max_relative_error = max(max_relative_error, abs(actual/reference-1))
    assert max_relative_error < 1e-12
    # Unequal weights and unequal endpoint variances.
    assert abs(propagate([1, math.sqrt(1000)], [.9, .1], 0)**2 - 10.81) < 1e-12
    # Rejected neighborhoods never gain confidence merely by advancing a pass.
    sigma = 3.
    for p in (0, .09184833, .12613998, .13781854, .14262104, .14655028):
        sigma = propagate([sigma, 1000, UNKNOWN], [1, 0, 0], p)
        assert abs(sigma-3) < 1e-12
    # Duplicate history (p=1) cannot create independent samples.
    assert propagate([3, 3, 3, 3], [.25]*4, 1) == 3
    assert propagate([UNKNOWN, UNKNOWN], [.5, .5], 0) == UNKNOWN
    assert propagate([UNKNOWN, 3], [1, 0], 0) == UNKNOWN
    # Unknown donors are imputed before actual-weight propagation.
    assert abs(propagate([UNKNOWN, 3], [.9, .1], 0)**2 - 9*.82) < 1e-12
    assert abs(propagate([UNKNOWN, 3], [.9, .1], 1)-3) < 1e-12
    # Temporal independent-mean blend; no further N_eff division.
    assert abs(propagate([2, 4], [.75, .25], 0)**2 - 3.25) < 1e-12
    for value in (-2., 0., 2**-24, .001, 3., 65504.):
        assert np.isfinite(np.float16(value))
    # Repeated realizations of an explicitly correlated Gaussian estimator.
    mc_errors = []
    s = np.array([1., 2., 4.])
    w = np.array([.6, .3, .1])
    for p in (0., .13, 1.):
        noise = math.sqrt(1-p)*rng.standard_normal((250000, 3))
        noise += math.sqrt(p)*rng.standard_normal((250000, 1))
        empirical = np.var((noise*s)@w, ddof=1)
        predicted = propagate(s, w, p)**2
        error = abs(empirical/predicted-1)
        assert error < .015
        mc_errors.append(error)
    # Exact operator energy on an infinite-support-equivalent periodic plane,
    # for the three regular-grid passes. This audits the trial constants; it
    # is not an adaptive Bures-weight calibration.
    constants = (ROOT/'shaders/lib/lighting/denoiser/internal_constants.glsl').read_text(encoding='utf-8')
    h = np.zeros((65, 65))
    h[32, 32] = 1.
    v = 1.
    plane = []
    kernel = [(0, 0, 1.)] + [(x, y, .44445 if x*y else .66667)
        for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y]
    weights = np.array([t[2] for t in kernel])
    for step in (1, 2, 4):
        p = float(re.search(r'CORRELATION_STEP_'+str(step)+r' = ([0-9.]+)', constants)[1])
        h = sum(w*np.roll(h, (y*step, x*step), (0, 1)) for x, y, w in kernel)/weights.sum()
        v = propagate([math.sqrt(v)]*9, weights, p)**2
        exact = float(np.sum(h*h))
        plane.append(dict(step=step, predicted=v, exact=exact, ratio=v/exact))
    # Contract checks on production call sites, not a second standalone formula.
    spatial = (ROOT/'shaders/lib/lighting/denoiser/atrous_filter.glsl').read_text()
    response = (ROOT/'shaders/lib/lighting/denoiser/temporal_response.glsl').read_text()
    assert 'denoiserResolveEstimatorSigma' in spatial
    assert 'clamp(centerEffectiveSamples' not in spatial
    assert '/ max(effectiveSamples' not in response
    for domain in ('diffuse', 'reflection'):
        source = (ROOT/f'shaders/post/denoiser/{domain}/resolve.glsl').read_text()
        assert 'denoiserScratchLoadEffectiveSamples' not in source
    for pattern in ('small', 'large'):
        source = (ROOT/f'shaders/lib/lighting/denoiser/atrous_{pattern}.glsl').read_text()
        assert 'RejectionConfidenceForStep' not in source
        assert 'AccumulateEffectiveSamples' not in source
    return dict(covariance_cases=10000, max_relative_error=max_relative_error,
                heteroscedastic_variance=10.81, center_only_six_pass_variance=9.,
                repeated_history_variance=9., monte_carlo_relative_errors=mc_errors,
                fully_accepted_grid_overlap=plane,
                limitations='Frozen scalar covariance; adaptive weights and changing Bures metric are not validated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    result = dict(tests=tests())
    if args.compile:
        audit = runpy.run_path(str(ROOT/'tools/audit_uncertainty_ingress.py'))
        result['compilation'] = audit['compile_more']()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
