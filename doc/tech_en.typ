#set text(font: "Microsoft YaHei", lang: "en")
#set page(numbering: "1", header: "Asymmetric Laplace Isomorphic Conic Encoding — Technical Documentation", footer: link(
  "https://github.com/sjrsjz",
  [GitHub: sjrsjz],
))
#set document(
  date: datetime.today(),
  author: (
    "https://github.com/sjrsjz",
    "sjrsjz@gmail.com",
  ),
  title: "Asymmetric Laplace Isomorphic Conic Encoding — Technical Documentation",
  description: [
    Asymmetric Laplace Isomorphic Conic Encoding — Technical Documentation
  ],
  keywords: (
    "Real-time Path Tracing",
    "Spatiotemporal Denoising",
    "Maximum Entropy Principle",
    "Maxwell-Jüttner Distribution",
    "Path Guiding",
    "Information Geometry",
  ),
)
#show raw: it => text(it, font: ("Consolas", "Microsoft YaHei"), size: 10pt, lang: "en")

#v(10em)
#align(center)[#text(size: 18pt)[Asymmetric Laplace Isomorphic Conic Encoding \ Technical Documentation]]

#v(5em)

#figure(caption: [Result after denoising with the ALICE encoder and an SVGF-like denoiser, 5 bounces])[
  #align(center)[
    #image("./assets/image-1.png")
  ]
]

#pagebreak()
#outline()
#pagebreak()

#align(center)[
  #rect(width: 100%, stroke: 0.5pt + luma(200), radius: 4pt, inset: 12pt)[
    #set align(left)
    #text(size: 10pt, fill: luma(80))[
      *Open Source License* \
      *Document body* (text, figures and theoretical exposition): licensed under \
      *Creative Commons Attribution 4.0 International (CC BY 4.0)* \
      (#link("https://creativecommons.org/licenses/by/4.0/")[https://creativecommons.org/licenses/by/4.0/]). \
      Anyone is free to copy, modify, redistribute and use this document commercially, provided the author is attributed; when redistributing or adapting, retain this attribution and license notice and indicate if the original document was modified.\
      *Accompanying code* (GLSL / HLSL shader implementation): remains under the *MIT License*, free of charge for academic research, commercial game engine development and offline renderer development; retain the original author attribution and this notice when distributing or using the core algorithm code.\
      *Attribution*: Author sjrsjz (#link("https://github.com/sjrsjz")[GitHub: sjrsjz] / sjrsjz\@gmail.com), title "Asymmetric Laplace Isomorphic Conic Encoding — Technical Document".
    ]
  ]
]
= Preface

== Background

Diffuse lighting reconstruction is an important problem in computer graphics. Its goal is to reconstruct a lighting vector from a set of lighting samples for use in subsequent rendering. In practical applications, due to noise in the lighting samples, the reconstructed lighting vector is often contaminated by noise, necessitating a denoising step.

Note: Essentially, I found SVGF @schied2017svgf too poor and difficult to tune with its unacceptably blurry cold-start, SH too heavy and prone to negative values and ringing artifacts, SG prohibitively slow, ReSTIR @bitterli2020spatiotemporal not a true denoising pass, and NRD @nvidia2021nrd both beyond my grasp and ill-suited for MC shaders (being excessively heavyweight). This directly forced me to produce the work presented below.

== Purpose

This document aims to introduce a novel lighting encoding scheme along with its corresponding denoising solution and reconstruction algorithm.

= Modeling

== First Principles
We define the lighting state space directly as the augmented cone:

$ cal(C) = {(bold(v), omega) in RR^n times RR_(>=0) | omega >= |bold(v)|} $

An arbitrary lighting encoding is represented as a 2-tuple $L = (bold(v), omega) in cal(C)$, where $bold(v) in RR^n$ is the directional moment and $omega in RR_(>=0)$ is the total incident radiance (sum of per-sample scalar magnitudes). The lighting synthesis operator $T: "List"(cal(C)) -> cal(C)$ maps a set of signal samples to a synthesized lighting state, where $cal(L) in "List"(RR^n)$ is the input list of Monte Carlo samples.

Each Monte Carlo sample is a probe ray: its BRDF modulation is deferred to the shading stage, so the raw signal $bold(x)_i = hat(bold(d))_i dot L_i$ encodes the material-independent incident radiance $L_i$ arriving from direction $hat(bold(d))_i$. This probe-based architecture ensures the ALICE encoding captures the full incident light field regardless of surface material (diffuse, metallic, or translucent).

#text(size: 10pt, fill: luma(100))[
  *Design rationale:* the choice of state space $cal(C)$ is not an arbitrary stipulation, but a natural consequence jointly determined by the signal structure and probability theory (see the "Probabilistic Origin of the Cone Constraint" section for details).
]

This modeling proceeds entirely from algebraic first principles and a signal-processing perspective (excluding additional physical assumptions) to derive the analytic form of the lighting synthesis operator $T$.

The operator satisfies the following algebraic axioms:

+ *Axiom 1 — Associativity & Commutativity (signal order-independence)* \
  The synthesis result is independent of the input order of samples and computational grouping. For any sample list $cal(L)$ and any disjoint partition $cal(L) = union.big_k cal(L)_k$, the operator satisfies:
  $ T(cal(L)) = T({ T(cal(L)_k) }) $

  #text(size: 10pt, fill: luma(100))[
    *Signal perspective:* Monte Carlo samples form an unordered data stream; statistical extracted quantities should not depend on sample arrival order. Mathematically, this axiom provides Lie group symmetry guarantees for the underlying topology of the accumulation operation.
  ]

+ *Axiom 2 — Irradiance Conservation (L1 amplitude additivity)* \
  The total irradiance of the system is strictly conserved before and after synthesis. Irradiance is defined as the L1 norm of the signal. For any sample list $cal(L) = {bold(x)_i}$:
  $ omega(T(cal(L))) = sum_(bold(x)_i in cal(L)) |bold(x)_i| $

  #text(size: 10pt, fill: luma(100))[
    *Signal perspective:* this is L1-norm additivity — the most fundamental amplitude conservation law in signal processing. In the Monte Carlo context, the total irradiance of the synthesized lighting equals the sum of irradiances from all sampled rays.
  ]

+ *Axiom 3 — Directional Moment Fidelity (first-moment additivity)* \
  The combined directional moment equals the vector superposition of the directional moments of each sample. For any sample list $cal(L) = {bold(x)_i}$:
  $ bold(v)(T(cal(L))) = sum_(bold(x)_i in cal(L)) bold(x)_i $

  Equivalent formulation: for any linear probe direction $bold(a) in RR^n$, its linear response to the synthesized lighting equals the superposition of the responses to all individual input samples:
  $ bold(a) dot bold(v)(T(cal(L))) = sum_(bold(x)_i in cal(L)) bold(a) dot bold(x)_i $

  #text(size: 10pt, fill: luma(100))[
    *Signal perspective:* this is the additivity of the signal first moment (mean). The directional moment $bold(v)$ is the vector mean of the signal in $RR^n$ multiplied by the number of samples; its linear additivity is a fundamental structural property of signal space.
  ]

+ *Axiom 4 — Positive Homogeneity* \
  When all inputs are scaled uniformly, the synthesized result is scaled by the same factor. For any non-negative scalar $lambda >= 0$:
  $ T({lambda bold(x)_i}) = lambda T({bold(x)_i}) $

  #text(size: 10pt, fill: luma(100))[
    *Signal perspective:* when the signal is uniformly amplified by factor $lambda$, all statistical extracted quantities scale proportionally — the scale-covariance axiom in signal processing.
  ]

We define the auxiliary mean operator $ T_"avg" (cal(L)) = 1/(|cal(L)|) T(cal(L)) $ to represent the average lighting synthesis result of the input samples.

== Operator Derivation

=== Direct Derivation

From Axiom 2 (Irradiance Conservation) and Axiom 3 (Directional Moment Fidelity), the analytic form of the synthesis operator $T$ is *directly and uniquely determined*:

$ T(cal(L)) = (sum_(bold(x)_i in cal(L)) bold(x)_i, sum_(bold(x)_i in cal(L)) |bold(x)_i|) in cal(C) $

Its corresponding mean operator is:

$ T_"avg"(cal(L)) = (E[bold(x)], E[ |bold(x)| ]) $

=== Cone Closure Proof

We must verify $T(cal(L)) in cal(C)$, i.e., prove $omega >= |bold(v)|$:

$ omega = sum_i |bold(x)_i| >= |sum_i bold(x)_i| = |bold(v)| $

This follows directly from the triangle inequality. $square$

=== Linearity

Introducing the augmented signal representation $tilde(bold(x)) = (bold(x), |bold(x)|) in RR^(n+1)$, the synthesis operator reduces to pure vector addition:

$ T(cal(L)) = sum_(bold(x)_i in cal(L)) tilde(bold(x))_i $

The mean operator is simply the mathematical expectation of the augmented signal:

$ T_"avg"(cal(L)) = E[tilde(bold(x))] $

=== Uniqueness Proof

The uniqueness of the synthesis operator is *independently and completely* guaranteed by Axiom 2 and Axiom 3:

- *Axiom 2* independently and completely locks the scalar component: $omega(T(cal(L))) = sum |bold(x)_i|$ is the unique scalar assignment satisfying Irradiance Conservation.
- *Axiom 3* independently and completely locks the vector component: $bold(v)(T(cal(L))) = sum bold(x)_i$ is the unique vector assignment satisfying Directional Moment Fidelity.
- Axiom 1 (Associativity \& Commutativity) provides Lie group symmetry guarantees for the accumulation operation, ruling out any non-commutative or non-associative synthesis schemes.
- Axiom 4 (Positive Homogeneity) rules out any nonlinear dependence on sample count or magnitude.

The two core axioms (2 and 3) each independently and uniquely determine one component of the output state; no other operator can simultaneously satisfy both axioms. Hence the analytic form of $T$ is unique. $square$

== Probabilistic Origin of the Cone Constraint

This section explains why the cone constraint $omega >= |bold(v)|$ on state space $cal(C)$ is not an arbitrary stipulation but an inevitable consequence of probability theory.

=== Jensen's Inequality Guarantee

For any probability measure $mu$ defined on $RR^n$ (with finite first moment and finite first absolute moment), by Jensen's inequality:

$ E_mu[ |bold(x)| ] >= |E_mu[bold(x)]| $

i.e., $omega >= |bold(v)|$. Therefore, any lighting state parameterized by $(bold(v), omega) = (E[bold(x)], E[ |bold(x)| ])$ *automatically falls within the cone* $cal(C)$.

=== Reinterpretation of Boundary States

The cone boundary $omega = |bold(v)|$ corresponds to Jensen's inequality taking equality, which occurs if and only if the distribution degenerates to a Dirac $delta$ function — i.e., all signal samples point in the same direction. This precisely corresponds to the raw single-sample ray state in the Monte Carlo pipeline before any spatial filtering:

$ "single-sample:" quad (bold(v), omega) = (bold(x)_i, |bold(x)_i|), quad omega = |bold(v)| quad "(cone boundary)" $

After spatial filtering and accumulation of multiple samples, due to the dispersion of the empirical distribution, Jensen's inequality holds strictly:

$ "multi-sample:" quad omega = E[ |bold(x)| ] > |E[bold(x)]| = |bold(v)| quad "(cone interior)" $

=== Physical Meaning of the Jensen Gap

Define the Jensen gap:

$ I = omega - |bold(v)| = E[ |bold(x)| ] - |E[bold(x)]| >= 0 $

$I$ measures the degree to which the signal distribution deviates from the Dirac state (perfectly directional). In the physical correspondence, $I$ is equivalent to the "thermal energy" of the photon gas — $I = 0$ corresponds to absolute zero (purely directional light), $I > 0$ corresponds to finite temperature (an isotropic diffuse scattering component is present).

== Algorithmic Significance

This section elaborates the engineering value of the above algebraic structure for real-time rendering pipelines.

In computer graphics and real-time rendering, spatial filtering and denoising algorithms (such as SVGF and its derivative architectures) inherently rely heavily on various linear or convex combination operations (weighted summation, convolution, etc.).

The mathematical derivation above provides a powerful proof: *the synthesis operator $T$ is pure vector addition in the augmented signal space*. In a real-world pipeline, we only need to perform extremely low-cost ordinary linear blending of samples in the augmented representation $tilde(bold(x)) = (bold(x), |bold(x)|)$ (computing $T_"avg"$), and the final filtered result will rigorously conform to all algebraic axioms and the Irradiance Conservation law.

Because $T$ is inherently linear, *no post-hoc nonlinear correction is needed* — every operation in the filtering loop is a strict linear combination, and irradiance reconstruction $E(hat(arrow(n)))$ is deferred to a single pass at the shading output stage. This lays a solid theoretical foundation for designing lighting denoising pipelines that achieve both "mathematically rigorous unbiasedness" and "extremely high shading execution efficiency."

= Statistical Model

== Monte Carlo Representation and Maximum Entropy Encoding

The first principles of the synthesis operator $T$ only constrain the algebraic structure and contain no statistical semantics themselves. However, a real-world denoising pipeline inherently needs to process random signals based on Monte Carlo (MC) sampling. In order to connect it to the denoising pipeline while introducing the minimum amount of ad-hoc prior bias at the statistical level, we adopt the Maximum Entropy Principle @jaynes1957information from information theory to construct, for a single lighting state $L$, the probability distribution that it implicitly encodes.

In a real-world MC sampling pipeline, the output $bold(x)_i = hat(bold(d))_i dot E_i$ of a single ray bounce corresponds to the lighting state $(bold(x)_i, |bold(x)_i|)$, lying strictly on the boundary of the cone $cal(C)$ — because the empirical distribution of a single sample is a Dirac $delta$ function, Jensen's inequality takes equality. After operator filtering (accumulation), the resulting internal local lighting state $(bold(v), omega)$ has $omega > |bold(v)|$ due to the dispersion of the multi-sample empirical distribution, with Jensen's inequality holding strictly.

We treat $(bold(v), omega) = (E[bold(x)], E[ |bold(x)| ])$ as a sufficient observational constraint on the local lighting field. Based on the Maximum Entropy Principle, we seek a maximum-entropy probability density distribution $p(bold(x))$ defined on the continuous momentum space $bold(x) in RR^n$, satisfying:

$
    "Maximize" quad & H[p] = - integral_(RR^n) p(bold(x)) log p(bold(x)) d bold(x) \
  "Subject to" quad & integral_(RR^n) p(bold(x)) d bold(x) = 1 \
                    & integral_(RR^n) bold(x) p(bold(x)) d bold(x) = bold(v) \
                    & integral_(RR^n) |bold(x)| p(bold(x)) d bold(x) = omega
$

=== Analytic Distribution Form and Boundary Handling

Using variational calculus (Lagrange multiplier method), in the non-singular domain $omega > |bold(v)| > 0$, the unique analytic solution of this maximum entropy problem can be rigorously derived, belonging to the Exponential Family:

$
  p_L (bold(x)) = (beta^n (1 - kappa^2)^((n+1)/2)) / (|S^(n-1)| Gamma(n)) exp(-beta(|bold(x)| - kappa hat(bold(v)) dot bold(x)))
$

where $|S^(n-1)|$ is the surface area of the $(n-1)$-dimensional unit sphere (for 3D space $n=3$, we always have $|S^2| Gamma(3) = 8pi$). The parameter analytic forms are completely determined by the first-order observation matrix:

$ rho = (|bold(v)|) / omega $
$ kappa = (2n rho) / ( (n+1) + sqrt((n+1)^2 - 4 n rho^2) ) $
$ beta = (n + kappa^2) / (omega (1 - kappa^2)) $
$
  hat(bold(v)) = cases(
    bold(v) / (|bold(v)|) & "if" bold(v) != bold(0),
    bold(h)_("any") & "if" bold(v) = bold(0)
  )
$

*Numerical Safeguards for Boundary Asymptotic Behavior:* In natural physics, the following degenerate limits of the above continuous distribution occur:
1. *Zero-vector unbiased decay ($bold(v) = bold(0)$)*: Here $rho = 0, kappa = 0$, and the distribution degenerates into an isotropic Laplace decay field $p(bold(x)) prop exp(-beta |bold(x)|)$.
2. *Absolute darkness state ($omega = 0$)*: The system is in a strict energy-extinguished state, and the variance collapses entirely to a point-mass distribution (i.e., the Dirac $delta(bold(x))$ function). In Shader implementation, evaluation is directly bypassed via an `omega` division-by-zero guard.
3. *Un-denoised raw ray state ($omega = |bold(v)|$)*: Here the connected manifold boundary $rho=1, kappa=1$ causes $beta$ to tend to infinity, making the distribution an extremely narrow distribution along the $hat(bold(v))$ axis (Dirac degeneration). This precisely reflects the fact that a single sample, without any spatial fusion, is extremely impoverished in low-frequency information. In practical applications, a hard threshold clamp can ensure it always falls within the non-singular measure domain: $rho < 1 - epsilon$.

=== Variance Structure in the Original Sample Space $RR^n$

The covariance matrix $op("Cov")(bold(X))$ of the non-singular maximum entropy distribution in the original space $RR^n$ possesses a perfectly axisymmetric geometric character (an uncertainty ellipsoid of revolution):

$
  op("Cov")(bold(X)) = sigma_(perp, bold(X))^2 (bold(I)_n - hat(bold(v))hat(bold(v))^T) + sigma_(parallel, bold(X))^2 hat(bold(v))hat(bold(v))^T
$

where the eigen-variances in the perpendicular and parallel principal directions are, respectively:
$ sigma_(perp, bold(X))^2 = ((n+1) omega^2 (1 - kappa^2)) / (n + kappa^2)^2 $
$ sigma_(parallel, bold(X))^2 = ((n+1) omega^2 (1 + kappa^2)) / (n + kappa^2)^2 $

The corresponding systemic eigen-homogeneous total scalar variance is:
$ "Var"_("scalar")(bold(X)) = op("tr")(op("Cov")(bold(X))) = ((n+1) omega^2) / (n+kappa^2)^2 [ n + (2-n)kappa^2 ] $

After eliminating the intermediate parameter $kappa$, its pure closed-form analytic expression, determined solely by the input first-order statistics, is:
$
  "Var"_("scalar")(bold(X)) = (omega ( (n+1)omega + sqrt((n+1)^2 omega^2 - 4 n |bold(v)|^2) )) / (2n) - (n-1)/(n+1) |bold(v)|^2
$

For 3D rendering scenarios ($n=3$), the scalar variance simplifies maximally to:
$ "Var"_("scalar")(bold(X)) = (2omega^2 + omega sqrt(4omega^2 - 3|bold(v)|^2)) / 3 - 1/2 |bold(v)|^2 $

In a Shader, considering the amortization effect, if the expected effective temporal accumulation frame count of a sample is $N_("eff")$, then the residual variance of the current pixel's estimator is $"Var"_("estimator") = "Var"_("scalar")(bold(X)) / N_("eff")$. This term can directly serve as the adaptive dynamic bandwidth $sigma_c^2$ equivalently implemented by a bilateral filter.

=== Joint Variance in the Augmented Signal Space $(bold(X), |bold(X)|)$

Since the algorithmic carrier of lighting synthesis actually operates in the augmented signal space $bold(Y) = (bold(X), R) = (bold(X), |bold(X)|)$, its complete joint covariance block matrix is crucial for the spatiotemporal filter (it supports more complex covariance evaluation via the $delta$-method):

$ op("Cov")(bold(Y)) = mat(op("Cov")(bold(X)), op("Cov")(bold(X), R); op("Cov")(R, bold(X)), op("Var")(R)) $

where the cross-covariance vector between direction and energy is:
$ op("Cov")(bold(X), R) = (2(n+1) kappa omega^2) / (n + kappa^2)^2 hat(bold(v)) $

and the radial energy uncertainty variance is:
$ op("Var")(|bold(X)|) = omega^2 / (n+kappa^2)^2 [ n + (n+3) kappa^2 - kappa^4 ] $


=== Information-Geometric Dual Space and Divergence Measure

To measure the core similarity between two lighting states $L_1(bold(v)_1, omega_1, N_1)$ and $L_2(bold(v)_2, omega_2, N_2)$ in a spatial neighborhood or temporal history, we embed the lighting states into the information-geometric surface of the continuous probability manifold @amari2016information. Based on Legendre Duality, a single lighting state $L = (bold(v), omega)$ possesses a unique bidirectional dual representation axis system in the hyperspace $RR^(n+1)$:

- *Primal coordinate vector (expectation parameter space / breadth distribution):*
  $ bold(psi)(L) := vec(bold(v), omega) in RR^(n+1) $

- *Dual coordinate vector (natural parameter space / intensity distribution):*
  $
    bold(phi)(L) := vec(bold(theta), -beta) = (n+kappa^2) / (omega(1-kappa^2)) vec((n+kappa^2) / ((n+1)omega) bold(v), -1) in RR^(n+1)
  $

The above dual coordinates strictly equal the gradient of the system's negative Shannon entropy with respect to the primal coordinates: $bold(phi)(L) = - nabla_(bold(psi)) H(p)$. Under the standard inner product, this model exhibits an elegant *zero-sum conservation theorem*:
$ bold(phi)(L) dot bold(psi)(L) = bold(theta) dot bold(v) - beta omega equiv -n $

According to the Legendre coordinate inner product theorem for symmetric Bregman divergences, the two-sample Jeffreys Divergence of an exponential family distribution — with no partition function residual cancellation — equals the pairwise inner product of the pure coordinate differences:
$ D_J(L_1, L_2) = (bold(phi)(L_1) - bold(phi)(L_2)) dot (bold(psi)(L_1) - bold(psi)(L_2)) $

The theoretical divergence does not encompass sample reliability in denoising algorithms. When used for bilateral rejection weights in a denoising pipeline, we additionally introduce the harmonically accumulated effective estimator $W_("eff") = (N_1 N_2) / (N_1 + N_2)$ based on the effective signal-to-noise ratio (Fisher information diagonal weighting), thereby deriving a weighted similarity distance measure suitable for real-world implementation with branchless evaluation (Weighted Jeffreys Divergence):

$
  D_("WJ")(L_1, L_2) &= W_("eff") dot D_J(L_1, L_2) \
  &= - (N_1 N_2) / (N_1 + N_2) [ bold(phi)(L_1) dot bold(psi)(L_2) + bold(phi)(L_2) dot bold(psi)(L_1) + 2n ]
$

Expanding the antipodal scalar components based on the above harmonically weighted cross terms yields:
$
  D_("WJ")(L_1, L_2) = (N_1 N_2) / (N_1 + N_2) [ beta_1 omega_2 + beta_2 omega_1 - bold(theta)_1 dot bold(v)_2 - bold(theta)_2 dot bold(v)_1 - 2n ]
$

For 3D rendering space ($n=3$), the constant term $-2n$ is always $-6$. In actual pipeline execution, this measurer possesses a dual algebraic dynamic range characteristic: during the low spatiotemporal SPP accumulation phase, the measure value naturally has a relatively high degree of adaptive soft tolerance, enabling rapid signal fusion and reconstruction; while at high SPP or when structural contrast is pronounced, the distance measure becomes extremely strict and rapidly transitions to a hard partitioning strategy to prevent ghosting artifacts.

= Lighting Reconstruction

== Integral Modeling and Hemispherical Projection

In a 3D rendering scenario ($n=3$), let the unit normal vector of the local shading point surface be $arrow(n) in S^2$. Assuming an ideal diffuse (Lambertian) material, according to the definition of irradiance, we need to perform a cosine-weighted projection integral of the previously derived maximum entropy distribution $p_L (bold(x))$ over the hemispherical space.

We define $E(arrow(n))$ as the mathematical expectation of the hemispherical cosine projection:
$
  E(arrow(n)) = E_(p_L) [ max(0, bold(x) dot arrow(n)) ] = integral_(RR^3) max(0, bold(x) dot arrow(n)) p_L (bold(x)) d bold(x)
$

Substituting the analytic form of the 3D maximum entropy distribution (where the unit sphere area and gamma function product $|S^2| Gamma(3) = 8pi$):
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(RR^3) max(0, bold(x) dot arrow(n)) exp(-beta (abs(bold(x)) - kappa hat(bold(v)) dot bold(x))) d bold(x)
$

== Analytic Azimuthal Integration and One-Dimensional Reduction

To simplify this 3D spatial integral, we introduce spherical coordinates. Let $bold(x) = r arrow(u)$, where $r = abs(bold(x)) in [0, oo)$ and $arrow(u) in S^2$ is the unit direction vector (volume element satisfies $d bold(x) = r^2 d r d arrow(u)$).

Using positive homogeneity to peel the integral into radial and angular double form:
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(S^2) max(0, arrow(u) dot arrow(n)) [ integral_0^oo r^3 exp(-beta (1 - kappa hat(bold(v)) dot arrow(u)) r) d r ] d arrow(u)
$

Using the definite integral relation $integral_0^oo r^3 e^(-a r) d r = Gamma(4) / a^4 = 6 / a^4$ (since $kappa in [0, 1)$, the radial convergence factor $a = beta(1 - kappa hat(bold(v)) dot arrow(u)) > 0$ always holds), the radial distribution in the integral expression is rigorously integrated out:
$
  E(arrow(n)) = (3 (1 - kappa^2)^2) / (4 pi beta) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

Eliminating the intermediate parameter $beta = (3 + kappa^2) / (omega (1 - kappa^2))$, we obtain a projection integral that depends only on the macroscopic total energy $omega$ and the directional distribution characteristics:
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 pi (3 + kappa^2)) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

We establish a local coordinate system with the normal $arrow(n)$ as the $z$-axis, so $cos theta = arrow(u) dot arrow(n)$. The hemispherical truncation operator $max(0, cos theta)$ strictly restricts the integration domain to the upward-facing hemisphere $Omega_+ = { arrow(u) in S^2 | cos theta >= 0 }$.

Let $mu_0 = hat(bold(v)) dot arrow(n)$. In this basis, we represent the principal optical axis direction $hat(bold(v))$ projected as $(sin theta_0, 0, mu_0)^T$. Integrating over the azimuthal angle $phi in [0, 2pi]$ (applying first-order derivative recursion):
$ integral_0^(2pi) d phi / (A - B cos phi)^4 = pi (2A^3 + 3A B^2) / (A^2 - B^2)^(7/2) $

where the auxiliary elements are defined as:
$ A(z) = 1 - kappa mu_0 z, quad B(z)^2 = kappa^2 (1 - mu_0^2)(1 - z^2) $
$ A(z)^2 - B(z)^2 = kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) $

Let $z = cos theta in [0, 1]$. After substitution and simplification, the azimuthal angle can be eliminated, yielding the *most simplified univariate analytic integral form* with respect to the zenith cosine $z$:
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 (3 + kappa^2)) integral_0^1 (z (1 - kappa mu_0 z) [ 2 (1 - kappa mu_0 z)^2 + 3 kappa^2 (1 - mu_0^2)(1 - z^2) ]) / ([ kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) ]^(7/2)) d z
$

== Boundary Behavior Analysis and Symmetric/Antisymmetric Decoupling

Since the denominator of the above expression contains the fractional-order algebraic term $Q(z)^(7/2)$, its antiderivative form is extremely cumbersome in the general domain. To construct an efficient real-time reconstruction scheme, we define the normalized irradiance response function as $e(mu_0, kappa) := E(arrow(n)) / omega$, and decompose it into a symmetric component $e_S$ and an antisymmetric component $e_A$ with respect to the cosine angle $mu_0$:
$
  e_S (mu_0, kappa) = (e(mu_0, kappa) + e(-mu_0, kappa)) / 2, quad e_A (mu_0, kappa) = (e(mu_0, kappa) - e(-mu_0, kappa)) / 2
$

By performing boundary limit derivations on the above one-dimensional integral, the system exhibits the following extremely elegant and symmetric mathematical boundary closed-form solutions:

1. *Omnidirectional Isotropic Limit ($kappa arrow.r 0$)*:
  $ e(mu_0, 0) equiv 1/4 $
2. *Extreme Directional Limit ($kappa arrow.r 1$)*:
  $ e(mu_0, 1) = max(0, mu_0) $
3. *Optical Axis and Normal Perfectly Co-aligned ($mu_0 = 1$)*:
  $ e(1, kappa) = ((1+kappa)^3 (3 - kappa)) / (4 (3 + kappa^2)) $
4. *Optical Axis and Normal Perfectly Anti-aligned ($mu_0 = -1$)*:
  $ e(-1, kappa) = ((1-kappa)^3 (3 + kappa)) / (4 (3 + kappa^2)) $
5. *Optical Axis Coplanar with Surface Tangent Plane ($mu_0 = 0$)*:
  $ e(0, kappa) = (3 sqrt(1 - kappa^2)) / (4 (3 + kappa^2)) $

*Analytic Uniqueness Theorem for the Antisymmetric Component:*
Further analysis reveals that the physical essence of the antisymmetric component $e_A$ is to restore the hemispherical projection to a full-sphere projection. By performing the untruncated integral over the unit sphere $S^2$, one can rigorously prove that this component is a *strictly linear function* of $mu_0$ and $kappa$ for arbitrary values, with absolutely no approximation error:
$ e_A (mu_0, kappa) equiv mu_0 dot (2 kappa) / (3 + kappa^2) $

Since $e_A$ has been rigorously integrated out, the entire fitting error of the lighting reconstruction collapses and retracts entirely onto the symmetric part $e_S$.

== Physical Smoothness Correction and High-Precision Approximation

The symmetric component $e_S$ describes the evolution process from the isotropic edge $e_S(0, kappa)$ to the collinear-aligned edge $e_S(1, kappa)$. Since for $kappa < 1$, the maximum entropy probability density field is smoothly differentiable ($C^oo$ continuous) on the local manifold, the lighting response it produces must have a strictly zero first-order derivative at $mu_0 = 0$.

Only when the system degenerates to the extreme Dirac limit ($kappa arrow.r 1$) does the discontinuous first-order characteristic of the kink term $| \mu_0 |$ emerge. Based on this physical prior, the blending weight of the kink term in the linear transition function $t$ should not be linear, but should exhibit higher-order decay characteristics as $kappa$ weakens.

We introduce the higher-order characteristic weight $kappa^4$ to suppress the kink response in the mid-to-low frequency band, constructing the following transition function $t$ and symmetric part approximation:
$ t = (1 - kappa^4) mu_0^2 + kappa^4 | mu_0 | $
$ e_S (mu_0, kappa) approx e_S (0, kappa) + (e_S (1, kappa) - e_S (0, kappa)) dot t $

Substituting the boundary analytic values of $e_S (0, kappa)$ and $e_S (1, kappa)$ and combining, we obtain the *final reconstruction formula that simultaneously guarantees rigorous exactness at all limiting boundaries, physical field smoothness and continuity, and a global maximum relative error controlled within $0.4%$*:

$
  E(arrow(n)) approx (omega) / (4(3+kappa^2)) [ 3 sqrt(1 - kappa^2) + (3 + 6 kappa^2 - kappa^4 - 3 sqrt(1 - kappa^2)) dot ((1 - kappa^4) mu_0^2 + kappa^4 | mu_0 |) + 8 kappa mu_0 ]
$

#image("./assets/image.png")

#text(size: 10pt, fill: luma(100))[
  *Normalization note:* the above reconstruction formula outputs irradiance values under the ALICE convention. Because single-sample probe encoding does not incorporate the MC integrator's sampling PDF factor (for cosine-weighted sampling $p(omega) = cos theta / pi$, each sample represents a solid angle of $pi / (N cos theta)$), the isotropic limit yields $E_"ALICE" = omega / 4$ versus the physical irradiance $E_"physical" = pi bar(L)$ — a calibration ratio of $4pi$. In the full rendering pipeline this constant factor is absorbed by tone mapping and exposure control; if physical-unit alignment with the specular reflection channel is desired, the calibration coefficient should be applied at the compositing stage.
]

== Irradiance Reconstruction — HLSL Implementation

The above algebraically restructured formula contains only basic arithmetic instructions, avoiding expensive transcendental functions (such as $sin, cos$) or numerical integration overhead, making it highly suitable for modern GPU rendering architectures. The following is the core logic executed in the actual Shader:

```hlsl
// High-precision O(1) diffuse irradiance reconstruction based on maximum entropy distribution
// Parameters:
//   v     - direction vector of the lighting after spatial filtering (v = L.v)
//   omega - total irradiance after spatial filtering
//   N     - surface unit normal vector of the current pixel
float ReconstructDiffuseLighting(float3 v, float omega, float3 N)
{
    // 0. Minimal energy boundary protection
    if (omega < 1e-6f) return 0.0f;

    float len_v = length(v);
    if (len_v < 1e-6f)
    {
        // Corresponds to the isotropic limit case (e_isotropic = 0.25)
        return omega * 0.25f;
    }

    float3 v_hat = v / len_v;
    float rho = min(len_v / omega, 0.999f); // Clamp to prevent division by zero

    // 1. Fast fitting of the characteristic parameter kappa (for 3D measure space n = 3)
    float sqrt_term = sqrt(16.0f - 12.0f * rho * rho);
    float kappa = (6.0f * rho) / (4.0f + sqrt_term);

    // 2. Cosine projection relation
    float mu_0 = dot(v_hat, N);
    float abs_mu_0 = abs(mu_0);

    // 3. Extract characteristic terms and common denominator
    float kappa_sq = kappa * kappa;
    float one_minus_kappa_sq = max(0.0f, 1.0f - kappa_sq);
    float sqrt_one_minus_kappa_sq = sqrt(one_minus_kappa_sq);

    float denom_shared = 3.0f + kappa_sq;

    // 4. Compute boundary components of the symmetric part
    float e_S0_num = 3.0f * sqrt_one_minus_kappa_sq;
    float e_S1_num = 3.0f + 6.0f * kappa_sq - kappa_sq * kappa_sq;

    // 5. Apply higher-order smooth interpolation function transition (guarantees physical C1/C2 continuity)
    float kappa_fourth = kappa_sq * kappa_sq;
    float t = (1.0f - kappa_fourth) * mu_0 * mu_0 + kappa_fourth * abs_mu_0;

    // 6. Combine symmetric part with unbiased antisymmetric part, compute final irradiance
    float e_S_num = lerp(e_S0_num, e_S1_num, t);
    float final_numerator = e_S_num + 8.0f * kappa * mu_0;
    float irradiance = omega * (final_numerator / (4.0f * denom_shared));

    return max(0.0f, irradiance);
}
```

= Denoising Pipeline Implementation

== Pipeline Overview

The ALICE denoising pipeline adopts a multi-pass architecture, performing progressive filtering of the diffuse lighting signal in the spatiotemporal domain. The overall pipeline consists of the following core passes (see #link("https://github.com/sjrsjz/Dirt-RT/tree/better-denoiser-dev/shaders/post", [Dirt RT/shaders/post])):

#figure(caption: [ALICE Denoising Pipeline Data Flow])[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      [*Pass*], [*Shader*], [*Function*],
      [100], [fragment], [Temporal accumulation: reprojects history frames using motion vectors, performs temporal recursive filtering in ALICE augmented space],
      [swap2], [compute], [Variance pre-filtering: $5 times 5$ geometry-aware bilateral filter smooths raw ALICE variance; $3 sigma$ energy clamping],
      [300_cs], [compute], [Spatial filtering L1—L3: à-trous wavelet decomposition $R_0 in {1, 2, 4}$, accelerated via shared memory],
      [300], [fragment], [Spatial filtering L4—L6: à-trous wavelet decomposition $R_0 in {8, 16, 32}$, rotation jitter decorrelation],
      [swap3], [compute], [Buffer swap: writes filtered results back to the main buffer, completes double-buffer flip, saves historical statistics],
    )
  ]
]

Pass 100 (Temporal Accumulation) uses motion vectors to reproject the history frame's ALICE encoding to the current frame, performing exponential moving average in the augmented space. The details of this pass are beyond the scope of this section, which focuses on the three core passes of spatial-domain denoising: swap2 (variance pre-filtering), 300/300_cs (spatial filtering), and swap3 (buffer swap).

#text(size: 11pt, fill: rgb("#8B0000"))[
  *⚠ Critical Note:* Although this denoiser references the à-trous wavelet decomposition framework of SVGF @schied2017svgf in its architecture, it differs fundamentally from standard SVGF in key dimensions including *signal domain, weight functions, variance estimation, and preprocessing strategy*. The following sections explicitly mark these differences at the relevant positions. A summary of the core differences is provided in the table below.
]

#figure(caption: [Key Differences between the ALICE Denoiser and Standard SVGF])[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      [*Dimension*], [*Standard SVGF*], [*This Denoiser*],
      [*Signal Domain*],
      [RGB 3-channel color (radiometric space)],
      [ALICE augmented representation $(bold(v), omega)$ (4D linear space), chrominance $("Co", "Cg")$ filtered independently],

      [*Synthesis Operator*],
      [Nonlinear (must handle nonlinear combinations in color space)],
      [Purely linear $T$ operator — sample accumulation in augmented space is simply vector addition, guaranteed by the ALICE algebraic structure],

      [*Variance Estimation*],
      [Local empirical variance over color channels],
      [Closed-form theoretical variance $"Var"_"scalar"(bold(X))$ of the ALICE maximum entropy distribution (depends only on first-order moments $(bold(v), omega)$), with a dedicated $5 times 5$ pre-filter pass],

      [*Luminance Weight*],
      [RGB luminance gradient + variance normalization],
      [ALICE vector space distance $|bold(v)_"center" - bold(v)_"sample"|$ + pre-filtered variance normalization],

      [*Energy Clamping*], [None], [$3 sigma$ energy clamping (swap2), preserving $rho = (|bold(v)|)/omega$ unchanged],
      [*Temporal Accumulation*], [Exponential moving average in RGB color space], [Direct linear accumulation in ALICE augmented space, $w$ is exactly the effective frame count $N_"eff"$],
      [*Output Signal*],
      [Filtered RGB color],
      [Filtered $(bold(v), omega) + ("Co", "Cg")$, irradiance reconstruction ($E(arrow(n))$) deferred to a single shading stage],
    )
  ]
]

#text(size: 12pt, fill: rgb("#1A5276"))[
  *Engineering Characteristics of the ALICE Denoiser:* \
  Within the *diffuse channel*, the ALICE denoiser achieves excellent joint denoising results without needing to separate Direct Illumination (DI) from Global Illumination (GI) — the ALICE probe encodes a material-independent incident light field, where DI (sun contribution) and GI (indirect bounces) accumulate naturally in the augmented space. Specular reflection and refraction channels use independent denoising pipelines (reflectIlluminationBuffer / refractIlluminationBuffer), architecturally decoupled from ALICE. \
  For purely diffuse or opaque dielectric materials, the diffuse channel handles the vast majority of the denoising load; for pure metals and other materials with negligible diffuse response, the specular reflection channel operates independently. \
  Actual performance: *contact shadows converge to clearly recognizable quality within 5—10 frames* (approximately 80—170 ms at 60 fps), making the convergence process virtually imperceptible to human vision.
]

== Unified Diffuse Buffer

The ALICE denoising pipeline uses a unified SSBO (Shader Storage Buffer Object) on the host side to manage diffuse data, replacing the traditional multi-texture approach. The per-pixel data structure `UnifiedDiffuseElement` contains 18 `float`s (72 bytes), covering the following functional domains:

+ *RT Output Domain (12B)*: The raw ALICE encoding of the current frame's ray-traced output (`rt_aliceY_xy`, `rt_aliceY_zw`, `rt_CoCg`), written by `ray0.rgen`, read by Pass 100.
+ *Current Geometry Domain (20B)*: World-space position $(p_x, p_y, p_z)$ + octahedron-compressed normal `oct_n` + secondary normal `oct_n2`.
+ *History Geometry Domain (16B)*: The previous frame's world-space position and compressed normal, used for edge-stop decisions during temporal reprojection. This domain is independent of the current geometry domain because `ray0.rgen` overwrites the current geometry domain every frame while the history geometry must persist across frames.
+ *Temporal History Domain (14B)*: The previous frame's accumulated ALICE encoding and accumulation weight, written by swap3, read by Pass 100.
+ *Swap Buffer Domain (14B)*: The current frame's to-be-filtered / already-filtered ALICE encoding and weight, serving as the data bus between Pass 100 → swap2 → 300 → swap3.

ALICE encodings are stored using half-precision floating-point (float16) compression: each lighting state $(bold(v), omega)$ as a `vec4` is packed into two `float`s (via `packHalf2x16` / `unpackHalf2x16`), and the chrominance component `CoCg` is packed into one `float`, making a total of 3 `float`s to fully represent a single ALICE lighting state. The compression/decompression interface is as follows:

```hlsl
vec3 packAlice(AliceEncoding encoded) {
    float s0 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.x, encoded.aliceY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.z, encoded.aliceY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(encoded.CoCg.x, encoded.CoCg.y)));
    return vec3(s0, s1, s2);
}

AliceEncoding unpackAlice(float s0, float s1, float s2) {
    AliceEncoding encoded;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    encoded.aliceY = vec4(v0.x, v0.y, v1.x, v1.y);
    encoded.CoCg = v2;
    return encoded;
}
```

== Variance Pre-Filtering (swap2)

=== Design Motivation

The spatial filter of this denoiser critically depends on per-pixel variance estimates $sigma^2$. However, the raw variance produced by single-frame Monte Carlo sampling contains extremely high-frequency noise; using it directly to guide à-trous filtering would cause: (1) the luminance weight $w_"luma" = Delta E / sigma$ to become unstable at noisy pixels, producing blocky artifacts; (2) low-variance regions being misidentified as high-confidence, leading to edge blurring.

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* Standard SVGF uses local empirical variance over color channels, recomputed at each à-trous level as $sigma^2 = max(0, 1/n sum x_i^2 - mu^2)$. This denoiser instead uses the closed-form theoretical variance $"Var"_"scalar"(bold(X))$ of the ALICE maximum entropy distribution, which depends only on the first-order moments $(bold(v), omega)$ and the effective frame count $N_"eff"$, requiring no storage of second-order moments. More importantly, we introduce a dedicated *variance pre-filter pass* (swap2) that performs $5 times 5$ geometry-aware bilateral smoothing on the raw variance before entering the à-trous iteration. This preprocessing step is one of the key design elements distinguishing this denoiser from standard SVGF.
]

Therefore, before entering the à-trous iteration, a mild pre-smoothing of the raw ALICE variance is required. swap2 performs $5 times 5$ geometry-aware bilateral filtering on $16 times 16$ workgroups, simultaneously completing $3 sigma$ energy clamping and data push to the colortex outputs.

=== Cooperative Shared-Memory Loading

swap2 uses a $20 times 20$ shared memory tile ($16 times 16$ workgroup + 2px halo), with 256 threads sharing the loading of 400 pixels via Round-Robin scheduling. Each tile element contains position, normal, raw variance, and total energy $omega$.

The raw variance is computed on-the-fly by `alice_estimator_variance(aliceY, weight)`, where `weight` is the temporally accumulated effective frame count $N_"eff"$. The variance formula uses the 3D scalar variance closed-form solution derived earlier (see the section "Variance Structure in the Original Sample Space $RR^n$"):

$ "Var"_"scalar"(bold(X)) = (2omega^2 + omega sqrt(4omega^2 - 3|bold(v)|^2)) / 3 - 1/2 |bold(v)|^2 $

$ sigma_"raw"^2 = "Var"_"scalar"(bold(X)) / max(N_"eff", 1) $

=== 3-Sigma Energy Clamping

Before performing spatial filtering on the ALICE encoding, swap2 first applies $3 sigma$ clamping to the output energy $omega$ of the center pixel. In a $5 times 5$ neighborhood, the weighted mean $mu_omega$ and standard deviation $sigma_omega$ are computed (weights $w_"kernel" dot w_"geom"$), and the center pixel's $omega$ is clamped to $[mu_omega - 3 sigma_omega, mu_omega + 3 sigma_omega]$:

$
  "scale" = omega_"clamped" / max(omega_"center", 10^(-8)), quad bold(v)' = bold(v) dot "scale", quad ("Co", "Cg")' = ("Co", "Cg") dot "scale"
$

After clamping, the entire ALICE vector is proportionally scaled to preserve the anisotropy $rho = (|bold(v)|) / omega$. This operation preserves the cone constraint $omega >= |bold(v)|$ while preventing extreme energy outliers caused by insufficient temporal accumulation from propagating through the subsequent à-trous filtering.

=== 5×5 Geometry-Aware Variance Filtering

The variance filter uses a B-spline kernel $h(x) in {1.0, 0.66667, 0.44444}$ with $5 times 5$ separable weights:

$ w_"kernel"(k_x, k_y) = h(|k_x|) dot h(|k_y|) $

The geometric weights adopt the standard SVGF formula:

$ w_"normal" = (op("clamp")(bold(n)_"center" dot bold(n)_"sample", 0, 1))^(gamma_n) $
$ w_"depth" = exp(-(|(bold(p)_"sample" - bold(p)_"center") dot bold(n)_"center"|) / (sigma_p dot "footprint")) $

The total weight is the product of three factors $w = w_"kernel" dot w_"normal" dot w_"depth"$, and the filtered variance is the weighted average (with an optional conservative mode using $max("filtered", "center")$ to guarantee variance is never underestimated). If computational cost is a concern, the normal weight can be approximated using an $exp$ form.

=== Gaussian Curvature Edge Marking (Optional, Non-Essential)

#text(fill: rgb("#8B0000"))[
  *⚠ Optional Feature:* Gaussian curvature edge marking is an *experimental auxiliary feature* of this denoiser, controlled by the compile-time macro `ENABLE_GAUSSIAN_FILTER`. In most scenarios, the basic geometric weights (normal + depth) are already sufficient to provide reliable edge stopping. This feature is not a necessary component of the denoiser, and disabling it does not affect core denoising quality. This procedure is only applicable to fragmented sub-pixel geometry that TAA cannot correctly handle, and its use should be avoided in practice.
]

At geometric edges (such as block corners, discontinuity boundaries), normal- and depth-based edge stopping cannot reliably separate lighting signals on either side — because the geometry buffer itself is already fractured at the edge. swap2 uses $3 times 3$ finite differences to compute the local surface Gaussian curvature $K$:

$
  E = (D_x bold(p))^2, quad F = (D_x bold(p)) dot (D_y bold(p)), quad G = (D_y bold(p))^2
$
$
  L = D_(x x) bold(p) dot bold(n), quad M = D_(x y) bold(p) dot bold(n), quad N = D_(y y) bold(p) dot bold(n)
$
$ K = (L N - M^2) / (E G - F^2) $

The first-order / second-order partial derivatives are estimated via central differences ($D_x$ using $plus.minus 1$ half-neighbor difference, $D_(x x)$ using the second-order central difference of the three points $+1, 0, -1$). When $|K| > tau$ (curvature threshold, defined by the `CURVATURE_THRESHOLD` macro), the pixel's $omega$ is marked as negative. In subsequent 300/300_cs passes, detecting $omega < 0$ automatically zeroes the geometric weight (`geomValid = 0`), relying solely on the luminance weight for denoising and avoiding incorrect blending of signals across geometric discontinuities.

== Spatial Filter (300 / 300_cs)

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* This denoiser only references SVGF in the *iterative architecture* of the à-trous wavelet decomposition. The core of the weight functions — signal representation, distance metric, variance source — is entirely redesigned around the ALICE encoding and is fundamentally different from standard SVGF. Details follow.
]

=== À-Trous Wavelet Decomposition Architecture

The spatial filter uses a 6-level à-trous wavelet iteration (iterative architecture referencing @schied2017svgf), with each level using a $3 times 3$ filter kernel and the step size $R_0$ doubling at each level:

#figure(caption: [À-Trous Iteration Level Configuration])[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      [*Level (STEP)*], [*Step Size $R_0$*], [*Shader*], [*Effective Radius*],
      [1], [$1$], [300_cs (Compute)], [1 px],
      [2], [$2$], [300_cs (Compute)], [3 px],
      [3], [$4$], [300_cs (Compute)], [7 px],
      [4], [$8$], [300 (Fragment)], [15 px],
      [5], [$16$], [300 (Fragment)], [31 px],
      [6], [$32$], [300 (Fragment)], [63 px],
    )
  ]
]

The first 3 levels ($R_0 <= 4$) execute in a compute shader (300_cs), leveraging shared memory (LDS) to reduce redundant texture reads; the last 3 levels ($R_0 >= 8$) fall back to a fragment shader (300), because at large step sizes the neighborhood tile overlap rate is low (texture reads across threads no longer overlap significantly) and the LDS benefit diminishes.

=== Compute Shader Shared-Memory Optimization

The core optimization of 300_cs lies in exploiting data reuse within a workgroup. For a $16 times 16$ workgroup and step size $R_0$, the required texture tile size is $(16 + 2R_0)^2$:

#figure(caption: [Tile Sizes and Shared Memory Usage for Each $R_0$ Level])[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      [*$R_0$*], [*Tile Size*], [*Geometry (vec4)*], [*Lighting (vec4)*],
      [1], [$18 times 18$], [324], [324],
      [2], [$20 times 20$], [400], [400],
      [4], [$24 times 24$], [576], [576],
    )
  ]
]

The shared memory usage for all tile sizes (maximum $576 times 2 times 16 = 18.0$ KB) is well below the typical GPU LDS limit of 32—64 KB. 256 threads share the loading task via Round-Robin scheduling (each thread loads $ceil("TILE_AREA" / 256)$ elements), with out-of-bounds pixels having their variance written as negative values as a sky mask. After loading, synchronization via `barrier()` + `memoryBarrierShared()` ensures that all subsequent $3 times 3$ à-trous sampling loops read entirely from shared memory, reducing the global texture read count by approximately 3—6×.

=== Tri-Lateral Weight Function

The composite weight of each neighborhood sample is formed by the product of three factors.

*Kernel Weight*: A B-spline kernel $h in {1.0, 0.66667}$, distinguishing only the center and cross-shaped neighbors:

$ w_"kernel"(i, j) = h(|i|) dot h(|j|) $

*Geometry Weight*:

$
  w_"geom" = gamma_n (1 - bold(n)_"center" dot bold(n)_"sample") + (|(bold(p)_"sample" - bold(p)_"center") dot bold(n)_"center"|) / (sigma_p dot "footprint") dot "geomValid"
$

where:
- $gamma_n$ is the normal sensitivity parameter (`SVGF_NORMAL_POWER`), controlling the penalty strength for normal differences;
- $sigma_p$ is the position sensitivity parameter (`SVGF_POSITION_PARAM`);
- $"footprint" = max("distToCam" / "resolution.y", 10^(-4))$ is the world-space footprint size of the pixel, making the depth term independent of screen resolution;
- `geomValid` is controlled by the Gaussian curvature flag: when the curvature exceeds the threshold, this term is zeroed, downgrading the filter to purely luminance-driven.

*Luminance Weight*:

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* Standard SVGF's luminance weight is based on the RGB color space gradient $|L_i - L_j|$ (where $L_i$ is the pixel luminance). This denoiser measures differences in the ALICE augmented space — using the Euclidean distance of the direction vector $|bold(v)_"center" - bold(v)_"sample"|$. The mathematical justification for this choice is: $bold(v)$ is precisely the linearly additive signal component in the augmented space $cal(C)$, and its Euclidean distance directly measures the difference in the joint direction-energy space of the lighting, without going through an irradiance reconstruction step. Furthermore, since the variance pre-filter (swap2) has already smoothed $sigma^2$, the center pixel's pre-filtered variance is used directly as the normalization baseline (standard SVGF uses locally computed $sigma_"center"^2 + sigma_"sample"^2$ at each à-trous level):
]

$ w_"luma" = (|bold(v)_"center" - bold(v)_"sample"|) / sqrt(sigma_"center"^2) dot phi_l $

where $phi_l$ is the luminance sensitivity parameter (`SVGF_PHI_L`).

=== Critical Note on the Asymmetric $sigma^2$ Choice

The use of the asymmetric form $sigma_"center"^2$ rather than the symmetric form $sigma_"center"^2 + sigma_"sample"^2$ is a core design decision of this denoiser, verified through extensive experimentation. The logic chain behind it critically depends on the existence of the variance pre-filter (swap2):

*Case 1: Without Variance Pre-Filtering (Standard SVGF Scheme)*

If the raw, un-pre-smoothed variance is used directly in the à-trous filter, the symmetric form $max(sigma_"center"^2 + sigma_"sample"^2, 10^(-8))$ must be adopted. The reason: the raw variance signal is extremely noisy, and single-pixel variance estimates are highly unstable. If only $sigma_"center"^2$ is taken, the center pixel's variance may oscillate violently between noise peaks and troughs — when $sigma_"center"^2$ happens to be extremely small (a variance trough) while $|bold(v)_"center" - bold(v)_"sample"|$ is large due to noise, the luminance weight $w_"luma"$ inflates abnormally, causing neighborhood samples to be over-trusted, resulting in *black speckle artifacts and severe energy loss* (denoiser collapse). The introduction of $sigma_"sample"^2$ is essentially a form of statistical smoothing — the superposition of two independent noise terms reduces the weight's sensitivity to single-point variance anomalies.

*Case 2: With Variance Pre-Filtering (This Denoiser's Scheme)*

The $5 times 5$ bilateral smoothing of swap2 has already substantially eliminated high-frequency noise from the variance signal, and $sigma_"center"^2$ itself is already sufficiently stable. If the symmetric form $sigma_"center"^2 + sigma_"sample"^2$ continues to be used at this point, not only is the stabilizing effect of $sigma_"sample"^2$ already negligible — more critically, $sigma_"sample"^2$ and $sigma_"center"^2$, though pre-filtered, still retain residual minute differences spatially. These differences fluctuate across frames as the sampling pattern changes, and after two-sample superposition, they produce *extremely severe temporal flickering side effects* in $w_"luma"$. This flickering cannot be fully eliminated by temporal accumulation because the inter-frame fluctuation of the luminance weights directly alters the effective kernel shape at every à-trous level.

*Conclusion:* Variance pre-filtering and single-sample $sigma_"center"^2$ are a tightly coupled design pair — the pre-filter eliminates the instability of $sigma_"center"^2$, thereby making it safe to remove the $sigma_"sample"^2$ term, which in turn eradicates the temporal flickering caused by two-sample variance superposition. This is yet another critical and non-obvious point of divergence from standard SVGF.

=== Parameter Sensitivity of the Variance Floor

In the actual implementation, $sigma^2$ is clamped to a lower bound $epsilon$ before entering $1/sqrt(sigma^2)$:

$ sigma^2 = max(sigma_"center"^2, epsilon), quad epsilon = 10^(-9) $

(In the symmetric scheme, this would be $max(sigma_"center"^2 + sigma_"sample"^2, 10^(-8))$, with the same principle.)

$epsilon$ is the most sensitive parameter in this denoiser governing the tradeoff between *contact shadow quality* and *stability in dark regions*. Its mechanism of action is as follows:

$epsilon$ controls the upper bound of $1/sqrt(sigma^2)$ — since $1/sqrt(sigma^2) <= 1/sqrt(epsilon)$. When $sigma_"center"^2$ naturally tends to extremely small values in dark regions, $1/sqrt(sigma^2)$ would diverge to infinity without an upper bound, causing $w_"luma"$ to completely dominate the composite weight and the denoiser to over-respond to any minute $bold(v)$ differences.

- *$epsilon$ too large (e.g., $10^(-6)$)*: The denoiser becomes "sluggish" — $1/sqrt(sigma^2)$ is suppressed, and $w_"luma"$ in dark regions cannot distinguish genuine lighting boundaries from noise; *contact shadows in low-energy regions are blurred*.
- *$epsilon$ too small (e.g., $10^(-12)$)*: The denoiser becomes "overly sensitive" — $1/sqrt(sigma^2)$ is extremely high in dark regions, and $w_"luma"$ over-responds to noise fluctuations, manifesting as *denoising collapse* (structural black spots in dark regions) or *shadow creeping* (inter-frame noise amplified by weights forming slowly drifting artifacts).
- *$epsilon = 10^(-9)$*: The balance point selected after extensive experimentation across diverse scenes (indoor dark corners, dense forest shadows, caves), providing sufficient numerical stability for dark regions while preserving contact shadow sharpness.

*Composite Weight*: The luminance weight appears in a "bilateral enhancement" form at two positions in the product — as an exponential decay term penalizing large energy differences, and simultaneously as a prefactor providing mild adaptive enhancement:

$ w_0 = w_"kernel" dot (1 + w_"luma") dot exp(-(w_"geom" + w_"luma")) $

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* Standard SVGF's composite weight is $w_"kernel" dot exp(-(w_"geom" + w_"luma"))$, i.e., relying solely on exponential decay. The $(1 + w_"luma")$ prefactor introduced by this denoiser additionally provides first-order luminance-adaptive enhancement, delivering faster convergence and stronger denoising capability in high-variance regions (where $w_"luma"$ is large). This modification is particular to the ALICE augmented space — because $w_"luma"$ is based on the Euclidean distance of $bold(v)$ rather than an RGB gradient, its numerical range and statistical properties differ from standard SVGF.
]

=== Sample Accumulation in ALICE Augmented Space

#text(fill: rgb("#8B0000"))[
  *Fundamental difference from standard SVGF:* This is the most fundamental point of divergence between this denoiser and standard SVGF. Standard SVGF accumulates weighted color values $sum_i w_i bold(c)_i$ in RGB color space — linear combinations in color space do not correspond to physical lighting synthesis. This denoiser accumulates in the ALICE augmented space $cal(C)$ — guaranteed by the ALICE first principles, the $T$ operator in the augmented space is pure vector addition, so $sum_i w_i tilde(bold(x))_i$ is mathematically strictly equivalent to the physical synthesis of lighting. The denoiser never needs to handle nonlinear synthesis logic at any step. This is the core advantage of ALICE encoding over traditional RGB denoising.
]

Neighborhood samples are synthesized through pure linear accumulation in the ALICE augmented space (corresponding to the linearity of the $T$ operator derived earlier):

$
  bold(v)_"accum" = sum_i w_i bold(v)_i, quad omega_"accum" = sum_i w_i omega_i, quad ("Co", "Cg")_"accum" = sum_i w_i ("Co", "Cg")_i
$

After normalization, $bold(v)_"out" = bold(v)_"accum" / sum_i w_i$, and similarly $omega$ and chrominance are each divided by the total weight. The variance is propagated to the next level via the variance propagation formula for independent-sample weighted means:

$ sigma_"out"^2 = (sum_i w_i^2 sigma_i^2) / ((sum_i w_i)^2) $

=== Rotation Jitter

At $R_0 >= 8$ (STEP ≥ 4) levels, the fixed axis-aligned sampling pattern can produce structured grid artifacts. To eliminate these artifacts, atrous_denoise_diffuse.glsl introduces random rotation (Rotation Jitter) at each level:

$ bold(d)_"rotated" = R(theta) dot bold(d)_"aligned", quad theta = 2 pi dot "rand"("pix" + R_0) $

The rotation matrix $R(theta)$ is a standard 2D rotation, with the random seed jointly hashed from the pixel coordinates and the current level step size $R_0$, ensuring: (1) rotation angles are independently distributed across different pixels; (2) rotation angles are mutually uncorrelated across different levels; (3) the rotation angle for the same pixel at the same level remains fixed across frames (to avoid temporal flickering). Sampling coordinates are rounded (`round`) to prevent floating-point truncation errors from causing out-of-bounds access.

=== Special Handling of the Final Pass

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* Standard SVGF's final level outputs only the filtered result. This denoiser additionally outputs a large-radius blurred version to an independent texture (`colortex5`), used by swap3 to accelerate the convergence of low-confidence pixels. This dual-output design does not exist in standard SVGF.
]

The final level (STEP = 6), in addition to outputting the regular filtered result (`colortex4`), also outputs a "large-radius post-blurred" result to `colortex5` (`out_light_sample_blurred`). This result is read by swap3 for fast convergence of low-weight pixels. Additionally, the Gaussian curvature flag (the sign of $omega$, a non-essential feature) is only propagated forward in non-final passes; the final pass output forces the absolute value to ensure that subsequent irradiance reconstruction is not disturbed by the sign marker.

== Buffer Swap (swap3)

swap3 is the bridge connecting the current frame to the next frame, responsible for the following key tasks:

*Double-Buffer Flip*: Copies `data_swap` (the current frame's filtered result) to `data` (which will serve as the next frame's history), for Pass 100 to read during temporal reprojection in the next frame. This operation closes the data flow loop in the temporal dimension.

*Historical Statistics Preservation*: $w_"prev" = w$, saving the current frame's effective accumulation frame count as the historical weight. This weight is used in the next frame's Pass 100 for: (1) computing the temporal blending factor $alpha$; (2) evaluating the reliability of the reprojected sample.

*Filtered Result Read-Back*: Reads the post-blurred ALICE encoding output by the final pass from colortex5, writing it into `data_swap` as the starting point for the next frame's spatial filtering.

*Low-Weight Blending Acceleration (inspired by NRD @nvidia2021nrd, non-standard SVGF procedure)*:

#text(fill: rgb("#8B0000"))[
  *Difference from standard SVGF:* Standard SVGF's temporal-spatial relationship is unidirectional (temporal accumulation → spatial filtering), and the temporal history is never modified after spatial filtering. This denoiser performs a reverse blend in swap3 — using the large-radius spatial filtering result to backfill low-confidence temporal history. This design borrows ideas from NRD (NVIDIA Real-time Denoisers) @nvidia2021nrd and does not exist in standard SVGF.
]

When the historical accumulation is insufficient ($w < 1.0$), the current historical data is gently blended toward the spatial filtering result:

$ "data" = "mix"("data", "blurred_alice", "clamp"(1 / max(w, 1.0), 0, 1)) $

This operation provides additional convergence acceleration during cold starts, disocclusions, or temporal accumulation breakdowns caused by rapid camera motion. In practice, the impact of this blending on denoising quality is mild, but it helps eliminate isolated flickering pixels. The parameter $1/max(w, 1.0)$ ensures that the blending intensity is inversely proportional to confidence: the smaller $w$ is, the more the result is biased toward trusting the spatial filtering result. Likewise, this operation is not essential and, when disabled, does not significantly affect final quality in most scenarios.

== Data Flow Summary

Integrating the three passes above, the complete data flow of the ALICE denoising pipeline is as follows:

1. *Pass 100 (Temporal Accumulation)*: Reads the historical ALICE encoding $arrow(L)_"hist"$ from the SSBO, performs weighted averaging with the current frame's RT output $arrow(L)_"curr"$ in the augmented space, and outputs $arrow(L)_"temp"$ along with the accumulation weight $w$.
2. *swap2 (Variance Pre-Filtering)*: Reads $arrow(L)_"temp"$ and $w$, computes the raw variance $sigma_"raw"^2$, performs $5 times 5$ bilateral smoothing + $3 sigma$ clamping + Gaussian curvature marking, and pushes the data to colortex3 (geometry) and colortex4 (ALICE + variance).
3. *300_cs (Spatial Filtering L1—L3)*: Cooperatively loads from colortex3/4 into LDS, performs à-trous filtering with $R_0 in {1, 2, 4}$, and writes the result back to colortex4.
4. *300 (Spatial Filtering L4—L6)*: Directly texelFetch from colortex3/4, performs à-trous filtering with $R_0 in {8, 16, 32}$ (including rotation jitter), and additionally outputs the final level to colortex5.
5. *swap3 (Buffer Swap)*: Double-buffer flip, saves historical statistics, reads back the blurred result, and closes the data flow loop. The next frame restarts from step 1.

In compressed representation, the spatial filtering stage of the ALICE denoiser requires only two $"vec4"$s (32 bytes total) to fully represent all necessary geometric information (one $"vec4"$) and lighting information (one $"vec4"$).

= Naming
This lighting encoding scheme is named Asymmetric Laplace Isomorphic Conic Encoding (abbreviated as ALICE). The name reflects its core mathematical and physical structure:

- *Asymmetric Laplace*: the probability distribution derived from the maximum entropy principle belongs to the asymmetric Laplace distribution family;
- *Isomorphic*: the encoding space $cal(C)$ is strictly isomorphic to the thermodynamic state space of a drifting massless photon gas in natural units ($c = 1$) — ALICE's directional moment is the photon gas's collective momentum, irradiance is the photon gas's total energy, and the maximum entropy distribution is the Maxwell-Jüttner distribution;
- *Conic*: the state space is the convex cone $cal(C) = {(bold(v), omega) | omega >= |bold(v)|}$, whose cone constraint is naturally guaranteed by Jensen's inequality $E[ |bold(x)| ] >= |E[bold(x)]|$.

= Physical Correspondence
Although ALICE is derived entirely from first principles, it is strictly equivalent to the *Relativistic Statistical Mechanics* model in physics. Specifically, the maximum entropy distribution of ALICE corresponds exactly, in physical terms, to a *Drifting Massless Gas* (i.e., a drifting photon gas) in local thermodynamic equilibrium.

== Maxwell-Jüttner Distribution
When we do not impose a monochromatic (fixed-wavelength) constraint on the photons, but instead allow them to distribute freely in the 3D continuous momentum space $RR^3$, applying the maximum entropy constraints on their macroscopic energy $omega$ and macroscopic momentum $bold(v)$ exactly yields the *Maxwell-Jüttner distribution with a drift velocity* from special relativity @juttner1911maxwellsche.

Within ALICE's mathematical formulas, there exists an extremely rigorous and elegant physical quantity mapping dictionary:

+ *Phase Space and Momentum* \
  The mathematical state vector $bold(x)$ strictly corresponds to a single photon's momentum $bold(p)$ (or equivalently, the energy $E/c$).

+ *Collective Drift Velocity* \
  The anisotropy $rho = (|bold(v)|) / omega$ and the maximum entropy characteristic parameter $kappa$ are physically equivalent. They represent the *dimensionless collective drift velocity (Drift Velocity Ratio, $v_"drift" / c$)* of this photon gas ensemble moving through space as a whole.

+ *Thermodynamic Temperature* \
  The natural parameter $beta$ corresponds to the inverse of the *effective kinetic temperature* of the photon gas in the laboratory (camera) reference frame, i.e., $beta = c / (k_B T_"lab")$.

+ *Relativistic Doppler Effect* \
  The core algebraic term $1 - kappa hat(bold(v)) dot arrow(u)$ appearing in the angular integral is precisely the *Relativistic Doppler Factor* from special relativity. Due to the high-speed collective drift of the photon gas, the light field energy undergoes extreme relativistic concentration in the forward direction (Relativistic Beaming).

== Morphological Evolution of the Light Field and Thermodynamic Interpretation
Through this physical mapping, various complex macroscopic lighting phenomena encountered in real-time rendering can be endowed with intuitive and rigorous microscopic thermodynamic interpretations:

- *Perfectly Diffuse Ambient Light ($rho = 0, kappa = 0$)* \
  The drift velocity is zero, and the system is in a globally thermally equilibrated, isotropic state. Here $bold(v) = E[bold(x)] = bold(0)$, the Jensen gap reaches its maximum $I = omega$, and all energy manifests as random thermal motion. This is equivalent to directionless uniform skylight or extremely well-converged multi-bounce low-frequency GI.

- *Perfectly Directional Light ($rho arrow.r 1, kappa arrow.r 1$)* \
  The collective drift velocity of the photon gas approaches the speed of light. Here the Jensen gap $I = omega - |bold(v)| arrow.r 0$, the system temperature contracts toward absolute zero, the distribution degenerates into a Dirac $delta$ function along the drift direction, and all energy is converted into uniform directional kinetic energy. Macroscopically, this manifests as an absolutely parallel, intense direct beam of light (such as a high-frequency solar beam or laser).

- *Soft Shadows and Penumbra Transitions ($0 < kappa < 1$)* \
  At the edges of soft shadows in practical scenes, the light field is in an intermediate non-equilibrium state between directional flow and thermal scattering. ALICE, through the parameter $kappa$, can extremely smoothly bridge these two extreme regimes, achieving physically self-consistent Contact Hardening and smooth soft-shadow gradients.


= Screen-Space Path Rebuilding Importance Sampling

#figure(caption: [First-moment distribution of the light field obtained after denoising])[
  #align(center)[
    #image("./assets/image-2.png")
  ]
]

== Physical Motivation and Prior Distribution

In real-time path tracing, although Next Event Estimation (NEE) can effectively reduce the variance of direct lighting, for complex secondary bounces (such as deep corridors or room interiors with tiny windows), blind cosine-weighted sampling has extreme difficulty hitting effective light sources, causing indirect lighting to produce destructively high-frequency, long-tail noise (Fireflies).

Since we have already extracted and reconstructed the maximum entropy distribution state $(bold(v), omega)$ of the light field in the spatiotemporal domain using ALICE encoding within the denoising pipeline, we can naturally use it as *prior knowledge (Prior)* to perform Path Guiding over the hemispherical space when casting rays in the next frame.

Based on the maximum entropy angular energy density, we construct a guiding probability density function (PDF) defined on the full unit sphere $S^2$:
$ p_"ALICE" (arrow(u)) = C / ((1 - kappa hat(bold(v)) dot arrow(u))^4) $
where $hat(bold(v))$ is the guiding principal axis reconstructed from the previous frame, and $kappa$ is the corresponding characteristic parameter.

== Full-Sphere Integral and Normalization Constant

To make it a strict probability density function, we need to solve for the normalization constant $C$ over the full solid angle. Let $mu = hat(bold(v)) dot arrow(u) = cos theta$ and let the azimuthal angle be $phi$. The integral proceeds as follows:
$
                      integral_(S^2) p_"ALICE" (arrow(u)) d arrow(u) & = 1 \
  C integral_0^(2pi) d phi integral_(-1)^1 1 / (1 - kappa mu)^4 d mu & = 1
$

The azimuthal integral yields $2pi$, and evaluating the definite integral over $mu$:
$ 2pi C [ 1 / (3 kappa (1 - kappa mu)^3) ]_(-1)^1 = 1 $
$ (2pi C) / (3 kappa) ( 1 / (1 - kappa)^3 - 1 / (1 + kappa)^3 ) = 1 $

Simplifying the bracketed term through common denominators:
$
  ((1+kappa)^3 - (1-kappa)^3) / ((1-kappa^2)^3) = (2kappa^3 + 6kappa) / ((1-kappa^2)^3) = (2kappa(kappa^2 + 3)) / ((1-kappa^2)^3)
$

Substituting back yields the algebraic normalization constant:
$ C = (3(1-kappa^2)^3) / (4pi(3+kappa^2)) $


== Rigorous Analytic Inverse Transform Sampling

To achieve efficient importance sampling with zero rejection rate on the GPU, we solve for the cumulative distribution function (CDF) of the marginal probability density distribution $p(mu) = 2pi C / (1 - kappa mu)^4$:
$ F(mu) = integral_(-1)^mu p(x) d x = (2pi C) / (3 kappa) ( 1 / (1 - kappa mu)^3 - 1 / (1 + kappa)^3 ) $

The normalization condition yields $F(1) = 1$. To generate the sampling angle $mu$ from a uniformly distributed random number $xi_1 in [0, 1)$, we set $F(mu) / F(1) = xi_1$:
$ ( 1 / (1 - kappa mu)^3 - 1 / (1 + kappa)^3 ) / ( 1 / (1 - kappa)^3 - 1 / (1 + kappa)^3 ) = xi_1 $

For efficient evaluation in a Shader, we define the boundary constants $a$ and $b$:
$ a = 1 / (1 + kappa)^3, quad b = 1 / (1 - kappa)^3 $
Substituting and simplifying:
$ ( (1-kappa mu)^(-3) - a ) / (b - a) = xi_1 $
$ (1 - kappa mu)^(-3) = a + xi_1 (b - a) = op("lerp")(a, b, xi_1) $

Taking the $-1/3$ power of both sides yields the analytic inverse mapping equation, implementable on the GPU in just two lines of code:
$ mu = (1 - [ op("lerp")(a, b, xi_1) ]^(-1/3)) / kappa $

Combined with the azimuthal angle $phi = 2pi xi_2$ uniformly generated from $xi_2$, we can directly sample a ray direction that conforms exactly to the ALICE probability distribution in $O(1)$ time.

== Dynamic Multiple Importance Sampling

Although ALICE provides guidance that approximates the true light field extremely closely, in dynamic scenes with drastic occlusion changes, the prior guidance from the previous frame may become invalid (e.g., sudden light source movement or camera teleportation). To guarantee the absolute unbiasedness of the rendering equation and avoid division-by-zero variance explosions, we blend ALICE sampling with classical cosine-weighted sampling via Multiple Importance Sampling (MIS) @veach1995optimally.

Algebraically and physically, the dimensionless drift velocity of ALICE, $rho = (|bold(v)|) / omega$, reflects the "directional confidence" of the light field. Therefore, we directly couple ALICE's mixture probability weight $P_"guide"$ to $rho$:
$
  P_"guide" = cases(
    0.975 dot rho & "if" |bold(v)| > 10^(-8),
    0 & "if" |bold(v)| <= 10^(-8)
  )
$
*Design Intent:* When the light field tends toward diffuse ambient ($rho arrow.r 0$), the system automatically degenerates to cosine sampling; when the light field exhibits strong directionality ($rho arrow.r 1$), the system invests $97.5%$ of the computational effort into ALICE guidance, retaining $2.5%$ cosine sampling as a safety floor.

Ultimately, the mixed probability density function for the next ray is:
$ p_"mix"(arrow(u)) = (1 - P_"guide") dot p_"cos"(arrow(u)) + P_"guide" dot p_"ALICE"(arrow(u)) $
where $p_"cos"(arrow(u)) = max(0, arrow(n) dot arrow(u)) / pi$.

In the rendering equation's Monte Carlo integrator, the diffuse BSDF contribution (product factor) is:
$ W_"BSDF" = (f_r dot max(0, arrow(n) dot arrow(u))) / p_"mix"(arrow(u)) = (p_"cos"(arrow(u))) / p_"mix"(arrow(u)) $

The above theory ultimately converges to an extremely minimal code architecture: the real-world pipeline, without requiring any complex octree or neural grid caches, uses only the spatiotemporal filtering byproduct $(bold(v), omega)$ from the previous frame to accomplish, with extremely low ALU overhead, a theoretically complete and physically unbiased path guiding step.

#bibliography("./references.bib", title: "References", style: "ieee")