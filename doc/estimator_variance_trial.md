# Direct estimator-variance trial (2026-09-06)

The active mode 0 now stores the uncertainty of the current filtered estimate.
The FP16 sigma lane, SSBO allocation, pass count, and raw moment format are
unchanged. Former filtered/spatial-Neff scratch lanes are unused. Both history
signatures changed so observation-variance histories are rejected after reload.
No commit or in-game measurement accompanies this trial.

## Contract

- Preparation estimates Bures observation variance with the existing g^-3
  closure. Proposal sigma is sqrt(V_observation / N_raw_center). The independent
  current-frame branch starts at sqrt(V_observation), since it has one sample.
  The pooled Kish count corrects the observation estimate; it is not the count
  of the center proposal.
- Each spatial pass normalizes its actual accepted signal weights a_i, and uses
  V_out = (1-p) sum(a_i^2 V_i) + p (sum(a_i sqrt(V_i)))^2.
  Center-only filtering preserves variance at every level, including at high
  raw Neff. Heterogeneous input variances participate before aggregation.
- Light rejection uses D_B^2 / (V_center + V_neighbor), with the same configured
  sensitivity at every level. No extra Neff division or cumulative confidence
  multiplier is applied.
- Reprojection and surface/virtual history mixtures use p=1 for overlapping
  estimates. Identical duplicate history taps therefore retain their variance.
  Diffuse footprint loss inflates sigma by 1/sqrt(clamped area scale).
- Temporal blending uses squared alpha weights with p=0. The raw Kish count
  remains separate and controls the existing baseline update rate.
- Unknown sigma remains -2. If some donors are known, their weighted sigma
  average supplies missing sigma estimates before actual-weight propagation.
  This fallback assumes local stationarity. If none are known, uncertainty
  remains unknown and light is preserved without variance-driven rejection.

For fixed weights, a common scalar error coordinate, and
Cov(i,j)=p sqrt(V_i V_j) for distinct inputs, the spatial formula follows
exactly from Var(sum(a_i X_i))=sum(a_i a_j Cov(i,j)). It requires no stored Neff.
The production Bures scalar is a local-metric contraction; changing metrics,
adaptive signal-dependent weights, model error in g^-3 closure and correlations
between current and history prevent an exact general variance claim. The
independent endpoint-difference approximation also omits cross-covariance.

## Validation

Run:

    python -B tools/audit_estimator_variance.py --compile

With seed 906202631:

- 10,000 covariance-matrix comparisons: maximum relative error 9.99e-16.
- Weights (.9,.1), variances (1,1000): V_out=10.81 at p=0.
- Six center-only passes: input/output variance remains 9.
- Four duplicate history taps: input/output variance remains 9 at p=1.
- Three 250,000-realization Gaussian checks: relative variance discrepancies
  0.252%, 0.083%, 0.309%. These test the stated scalar covariance model.
- All 43 production GLSL compilation variants passed; no GPU execution.

The inherited overlap coefficients are trial values, not newly calibrated
constants. For three fully accepted regular-grid passes at steps 1,2,4,
predicted/exact operator energy is 1.0000, 0.9417, 0.8908. Thus the old constants
underestimate this baseline's variance by about 11% after the third pass.
New adaptive-policy calibration and in-game validation remain necessary.

Prepared and Filtered debug views now display estimator variance. Their scale
changes deliberately: Prepared proposal variance falls with temporal sample
count; Filtered variance contracts according to actual multikernel weights.
The debug views still refer to the proposal before/after A-Trous, not the final
temporally resolved visible-history variance.
