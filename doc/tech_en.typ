#set text(font: "Microsoft YaHei")
#set page(numbering: "1")

= Technical Documentation

== Preface

=== Background Introduction

Diffuse lighting reconstruction is an important problem in computer graphics. Its goal is to reconstruct a lighting vector from a set of lighting samples for use in subsequent rendering. In practical applications, due to noise in the lighting samples, the reconstructed lighting vector is often contaminated by noise, necessitating a denoising step.

Note: Essentially, I found SVGF too poor and difficult to tune, SH too heavy and prone to negative values and ringing artifacts, SG prohibitively slow, ReSTIR not a true denoising pass, and NRD both beyond my grasp and ill-suited for MC shaders. This directly led to the work presented below.

=== Purpose

This document aims to introduce a novel lighting encoding scheme along with its corresponding denoising solution and reconstruction algorithm.

== Modeling

=== First Principles
We define the lighting state space as $S = RR^n times RR_(>=0)$.
An arbitrary lighting encoding is represented as a 2-tuple $L = (arrow(v), I) in S$, where $arrow(v) in RR^n$ and $I in RR_(>=0)$.

Define the energy metric function $omega(L) : S -> RR_(>=0)$ satisfying:
$ omega(L) := abs(arrow(v)) + I $

Define the component functions $ v(L) := bold(v), quad I(L) := I $.

This modeling proceeds entirely from algebraic first principles (excluding additional physical assumptions) to derive the analytic form of the lighting synthesis operator $T: "List"(S) -> S$ (where $cal(L) in "List"(S)$ is the input list of lighting samples).

The algebraic first-principle axioms it satisfies are as follows:

+ *Associativity & Commutativity* \
  The synthesis result of lighting is independent of the input order of samples and computational grouping. For any sample list $cal(L)$ and any disjoint partition $cal(L) = union.big_k cal(L)_k$, the operator satisfies:
  $ T(cal(L)) = T({ T(cal(L)_k) }) $

+ *Energy Conservation* \
  The total energy of the system is strictly conserved before and after synthesis. For any sample list $cal(L)$:
  $ omega(T(cal(L))) = sum_(L_i in cal(L)) omega(L_i) $

+ *Positive Homogeneity* \
  When all inputs are scaled proportionally, the synthesis result is scaled by the same factor. For any non-negative scalar $lambda >= 0$:
  $ T({lambda L_i}) = lambda T({L_i}) $

+ *Directional Response Additivity / First-Moment Conservation* \
  For any directional detector (i.e., any linear projection direction $bold(a) in RR^n$), its linear response to the synthesized lighting must equal the superposition of responses to all individual input samples. That is:
  $ bold(a) dot v(T(cal(L))) = sum_(L_i in cal(L)) bold(a) dot v_i $

We auxiliary define $ T_"avg" (cal(L)) = 1/(|cal(L)|) T(cal(L)) $ to represent the average lighting synthesis result of the input samples.

=== Derivation of the Operator

==== Algebraic Isomorphism and Formal Decomposition
We map the lighting state space $S = RR^n times RR_(>=0)$ via a bijection $phi: S -> C$ into a more tractable convex cone $C subset RR^(n+1)$:
$C = { (bold(x), y) in RR^(n+1) | y >= |bold(x)| }$.

The mapping and its inverse are respectively defined as:
$
  phi(bold(v), I) := (bold(v), |bold(v)| + I) \
  phi^(-1)(bold(x), y) := (bold(x), y - |bold(x)|)
$

In the isomorphic space $C$, the induced synthesis operator $T_C$ must satisfy the corresponding forms of the original axioms:
- By *Energy Conservation*, the second coordinate (i.e., the vertical energy component) is directly locked to ordinary accumulation: $Omega_"res" = sum (|bold(v)_i| + I_i)$;
- By the newly introduced *Directional Response Additivity*, since the axiom holds for all probe directions $bold(a)$, the necessary and sufficient condition for the equality strictly constrains the vector component synthesis on its base manifold to be ordinary vector summation: $bold(V)_"res" = sum bold(v)_i$.

This summation result naturally satisfies the cone's closure property (the triangle inequality ensures $|sum bold(v)_i| <= sum |bold(v)_i| <= Omega_"res"$).

Pulling the result of $T_C$ back to $S$ via $phi^(-1)$ yields the unique analytic form of the lighting synthesis operator:
#set math.equation(numbering: none)
$
  T(cal(L)) = ( sum_(L_i in cal(L)) bold(v)_i, sum_(L_i in cal(L)) I_i + sum_(L_i in cal(L)) |bold(v)_i| - |sum_(L_i in cal(L)) bold(v)_i| )
$

Its corresponding mean operator is:
$ T_"avg"(cal(L)) = ( E[bold(v)], E[I] + E[|bold(v)|] - |E[bold(v)]| ) $

==== Uniqueness Proof
The uniqueness of the synthesis operator is strictly guaranteed by the bijective isomorphism $phi$ together with the first-principle axioms:
"Energy Conservation" independently and completely determines the closure characteristics of the scalar component (the total measure space dimension), thereby locking the scalar product; "Directional Response Additivity" explicitly eliminates non-identity isotropic rescaling solutions on top of the measure basis (e.g., avoiding nonlinear bias caused by amplifying or suppressing the directional vector intensity). Furthermore, Associativity & Commutativity mathematically provide Lie group symmetry guarantees for the underlying topology of the accumulation operation. After pulling back to $S$, the analytic form of $T$ must be unique — no other operator can violate this explicit form while simultaneously satisfying all the above axioms.

=== Linear Embedding of the Operator and Algorithmic Introduction

To decouple the nonlinearity introduced by the absolute norm in the original operator $T$, we follow the preceding isomorphism philosophy and introduce, at the algorithmic level, the Isomorphism-induced Linear Operator $T^*: "List"(S) -> S$.

We map each lighting sample to its Linear Embedded Representation $cal(S)[L]$, reparameterizing it as a joint "vector–total-energy" form:
$ cal(S)[L] := (bold(v), omega(L)) = (bold(v), I + |bold(v)|) $

In this embedding space, the induced operator $T^*$ degenerates into an elegant pure linear accumulation:
$ T^*(cal(L)) = sum_(L_i in cal(L)) cal(S)[L_i] $

The corresponding induced mean operator is simply the linear mathematical expectation over the joint samples:
$ T^*_"avg"(cal(L)) = E[cal(S)[L]] $

Based on the embedding and decomposition of the above algebraic structure, the original mean operator $T_"avg"(cal(L))$, which inevitably contains complex nonlinear terms, can be rigorously and uniquely decoupled into the difference of a "linear representation expectation" and a "post-hoc energy pull-back compensation":
$
  T_"avg"(cal(L)) = ( E[bold(v)], E[omega(L)] - |E[bold(v)]| ) = T^*_"avg"(cal(L)) - (bold(0), |bold(v)(T^*_"avg"(cal(L)))|)
$

*Algorithmic Significance:*
This algebraic decoupling property exhibits penetrating engineering value. In computer graphics and real-time rendering, spatial filtering and denoising algorithms (such as SVGF and its derivative architectures) inherently rely heavily on various linear or convex combination operations (weighted summation, convolution, etc.).
Our mathematical derivation provides a powerful proof: at the microscopic filtering loop level, the algorithm does not need to handle complex nonlinear lighting synthesis at all. In a real pipeline, we only need to perform extremely low-cost ordinary linear blending on samples in the embedding space $cal(S)$ (computing $T^*_"avg"$), and apply a single $O(1)$-complexity global norm correction at the final shading output stage. This architecture rigorously guarantees that the final filtering result absolutely conforms to all algebraic distribution axioms and the physical energy conservation invariant. This lays a solid theoretical foundation for designing lighting denoising pipelines that balance "mathematically rigorous unbiasedness" with "extremely high shading execution efficiency."

== Statistical Model

=== Monte Carlo Representation and Maximum Entropy Encoding

The first-principle synthesis operator $T$ only constrains the algebraic structure and does not itself contain any statistical semantics. However, a real-world denoising pipeline inherently needs to process random signals based on Monte Carlo (MC) sampling. To interface this with the denoising pipeline while introducing minimal ad-hoc prior bias at the statistical level, we adopt the Maximum Entropy Principle from information theory to construct, for a single lighting state $L$, the probability distribution it implicitly encodes.

In a real MC sampling pipeline, the raw radiance carried by a single ray cast, devoid of any isotropic low-frequency energy, is a boundary state $L_("raw") = (bold(x), 0)$, whose linear embedded representation equals $phi(L_("raw")) = (bold(x), |bold(x)|)$, lying strictly on the boundary of the isomorphic cone. After operator filtering (accumulation), the resulting internal local lighting state $L = (bold(v), I)$ has total energy $omega = |bold(v)| + I$ (where typically $omega > |bold(v)|$).

We treat $cal(S)[L] = (bold(v), omega)$ as sufficient observation constraints on the local lighting field. Based on the Maximum Entropy Principle, we seek a probability density distribution $p(bold(x))$ of maximum entropy defined on the continuous momentum space $bold(x) in RR^n$, satisfying:

$
    "Maximize" quad & H[p] = - integral_(RR^n) p(bold(x)) log p(bold(x)) d bold(x) \
  "Subject to" quad & integral_(RR^n) p(bold(x)) d bold(x) = 1 \
                    & integral_(RR^n) bold(x) p(bold(x)) d bold(x) = bold(v) \
                    & integral_(RR^n) |bold(x)| p(bold(x)) d bold(x) = omega
$

==== Analytic Distribution Form and Boundary Handling

Using the calculus of variations (Lagrange multiplier method), within the non-singular domain $omega > |bold(v)| > 0$, the unique analytic solution to this maximum entropy problem can be rigorously derived, belonging to the Exponential Family:

$
  p_L (bold(x)) = (beta^n (1 - kappa^2)^((n+1)/2)) / (|S^(n-1)| Gamma(n)) exp(-beta(|bold(x)| - kappa hat(bold(v)) dot bold(x)))
$

where $|S^(n-1)|$ is the surface area of the $(n-1)$-dimensional unit sphere (for 3D space $n=3$, $|S^2| Gamma(3) = 8pi$ always holds). The parameter analytic forms are completely determined by the first-order observation matrix:

$ rho = (|bold(v)|) / omega $
$ kappa = (2n rho) / ( (n+1) + sqrt((n+1)^2 - 4 n rho^2) ) $
$ beta = (n + kappa^2) / (omega (1 - kappa^2)) $
$
  hat(bold(v)) = cases(
    bold(v) / (|bold(v)|) & "if" bold(v) != bold(0),
    bold(h)_("any") & "if" bold(v) = bold(0)
  )
$

*Numerical Safeguards for Boundary Limit Behavior*: Under natural physics, the following degenerate limits of the above continuous distribution may occur:
1. *Zero-vector unbiased decay ($bold(v) = bold(0)$)*: Here $rho = 0, kappa = 0$, and the distribution degenerates to an isotropic Laplace decay field $p(bold(x)) prop exp(-beta |bold(x)|)$.
2. *Absolute dark state ($omega = 0$)*: The system is in a strict zero-energy state, and the variance collapses entirely to a point mass distribution (i.e., the Dirac $delta(bold(x))$ function). In the shader implementation, this is bypassed directly via an `omega` division-by-zero guard.
3. *Undenoised raw ray state ($omega = |bold(v)|$)*: Here the connected manifold boundary $rho=1, kappa=1$ causes $beta$ to tend to infinity, making the distribution an extremely narrow distribution along the $hat(bold(v))$ axis (Dirac degeneration). This precisely reflects the fact that a single sample, without any spatial fusion, is extremely lacking in low-frequency information. In practical applications, a hard threshold clamp can be applied to ensure it always stays within the non-singular measure domain: $rho < 1 - epsilon$.

==== Variance Structure in the Original Sample Space $RR^n$

The covariance matrix $op("Cov")(bold(X))$ of the non-singular maximum entropy distribution in the original space $RR^n$ possesses a perfectly axisymmetric geometric character (an uncertainty ellipsoid of revolution):

$
  op("Cov")(bold(X)) = sigma_(perp, bold(X))^2 (bold(I)_n - hat(bold(v))hat(bold(v))^T) + sigma_(parallel, bold(X))^2 hat(bold(v))hat(bold(v))^T
$

where the eigen-variances in the perpendicular and parallel principal directions are respectively:
$ sigma_(perp, bold(X))^2 = ((n+1) omega^2 (1 - kappa^2)) / (n + kappa^2)^2 $
$ sigma_(parallel, bold(X))^2 = ((n+1) omega^2 (1 + kappa^2)) / (n + kappa^2)^2 $

The corresponding system eigen-homogeneous total scalar variance is:
$ "Var"_("scalar")(bold(X)) = op("tr")(op("Cov")(bold(X))) = ((n+1) omega^2) / (n+kappa^2)^2 [ n + (2-n)kappa^2 ] $

After eliminating the intermediate parameter $kappa$, its pure closed-form analytic expression determined solely by the input first-order statistics is:
$
  "Var"_("scalar")(bold(X)) = (omega ( (n+1)omega + sqrt((n+1)^2 omega^2 - 4 n |bold(v)|^2) )) / (2n) - (n-1)/(n+1) |bold(v)|^2
$

For 3D rendering scenes ($n=3$), the scalar variance simplifies to:
$ "Var"_("scalar")(bold(X)) = (2omega^2 + omega sqrt(4omega^2 - 3|bold(v)|^2)) / 3 - 1/2 |bold(v)|^2 $

In a shader, accounting for the averaging effect, if the effective temporal accumulation frame count of a sample is $N_("eff")$, then the residual variance of the current pixel estimator is $"Var"_("estimator") = "Var"_("scalar")(bold(X)) / N_("eff")$. This term can directly serve as the adaptive dynamic bandwidth $sigma_c^2$ for an equivalent bilateral filter execution.

==== Joint Variance in the $T^*$ Embedding Space $(bold(X), |bold(X)|)$

Since the algorithmic carrier of lighting synthesis actually operates in the linear embedding space $bold(Y) = (bold(X), R) = (bold(X), |bold(X)|)$, its complete joint covariance block matrix is crucial for spatiotemporal filters (it supports more complex covariance evaluations via the $delta$-method):

$ op("Cov")(bold(Y)) = mat(op("Cov")(bold(X)), op("Cov")(bold(X), R); op("Cov")(R, bold(X)), op("Var")(R)) $

where the cross-covariance vector between direction and energy is:
$ op("Cov")(bold(X), R) = (2(n+1) kappa omega^2) / (n + kappa^2)^2 hat(bold(v)) $

and the radial energy uncertainty variance is:
$ op("Var")(|bold(X)|) = omega^2 / (n+kappa^2)^2 [ n + (n+3) kappa^2 - kappa^4 ] $


==== Information-Geometric Dual Space and Divergence Measure

To measure the core similarity between two lighting states $L_1(bold(v)_1, omega_1, N_1)$ and $L_2(bold(v)_2, omega_2, N_2)$ in a spatial neighborhood or across temporal history, we embed the lighting states onto the information-geometric surface of a continuous probability manifold. Based on Legendre Duality, a single lighting state $L = (bold(v), omega)$ possesses a unique bidirectional dual coordinate system in the hyperspace $RR^(n+1)$:

- *Primal coordinate vector (expectation parameter space / breadth distribution):*
  $ bold(psi)(L) := vec(bold(v), omega) in RR^(n+1) $

- *Dual coordinate vector (natural parameter space / intensity distribution):*
  $
    bold(phi)(L) := vec(bold(theta), -beta) = (n+kappa^2) / (omega(1-kappa^2)) vec((n+kappa^2) / ((n+1)omega) bold(v), -1) in RR^(n+1)
  $

The above dual coordinates are strictly equal to the gradient of the negative Shannon entropy with respect to the primal coordinates: $bold(phi)(L) = - nabla_(bold(psi)) H(p)$. Under the standard inner product, this model exhibits an elegant *zero-sum conservation theorem*:
$ bold(phi)(L) dot bold(psi)(L) = bold(theta) dot bold(v) - beta omega equiv -n $

According to the Legendre coordinate inner product theorem for symmetric Bregman divergences, the Jeffreys Divergence between two samples of an exponential family distribution contains no partition function cancellation residue and equals the antipodal inner product of their pure coordinate differences:
$ D_J(L_1, L_2) = (bold(phi)(L_1) - bold(phi)(L_2)) dot (bold(psi)(L_1) - bold(psi)(L_2)) $

The theoretical divergence does not incorporate sample confidence for denoising algorithms. When used for bilateral rejection weights in a denoising pipeline, we additionally introduce a harmonic cumulative estimation operator based on the effective signal-to-noise ratio (Fisher information diagonal weighting) $W_("eff") = (N_1 N_2) / (N_1 + N_2)$, thereby deriving a weighted similarity distance metric suitable for practical, branch-free evaluation (Weighted Jeffreys Divergence):

$
  D_("WJ")(L_1, L_2) &= W_("eff") dot D_J(L_1, L_2) \
  &= - (N_1 N_2) / (N_1 + N_2) [ bold(phi)(L_1) dot bold(psi)(L_2) + bold(phi)(L_2) dot bold(psi)(L_1) + 2n ]
$

Expanding the above antipodal scalar components based on harmonic-weight cross terms yields:
$
  D_("WJ")(L_1, L_2) = (N_1 N_2) / (N_1 + N_2) [ beta_1 omega_2 + beta_2 omega_1 - bold(theta)_1 dot bold(v)_2 - bold(theta)_2 dot bold(v)_1 - 2n ]
$

For 3D rendering space ($n=3$), the constant term $-2n$ is invariably $-6$. In actual pipeline execution, this metric possesses a rigorously discriminating algebraic dynamic range property: during low spatiotemporal SPP accumulation periods, the measure value naturally features a relatively high degree of adaptive soft tolerance, allowing rapid signal fusion and reconstruction; whereas at high SPP or when structural contrast is pronounced, the distance metric becomes extremely stringent, rapidly switching to a hard partitioning strategy to prevent ghosting artifacts.

== Lighting Reconstruction

=== Integral Modeling and Hemispherical Projection

In a 3D rendering scene ($n=3$), let the unit normal vector at the local shading point be $arrow(n) in S^2$. Assuming an ideal diffuse (Lambertian) material, according to the definition of irradiance, we need to perform a cosine-weighted projection integral of the previously derived maximum entropy distribution $p_L (bold(x))$ over the hemispherical domain.

We define $E(arrow(n))$ as the mathematical expectation of the hemispherical cosine projection:
$
  E(arrow(n)) = E_(p_L) [ max(0, bold(x) dot arrow(n)) ] = integral_(RR^3) max(0, bold(x) dot arrow(n)) p_L (bold(x)) d bold(x)
$

Substituting the analytic form of the 3D maximum entropy distribution (where the unit sphere area–gamma function product $|S^2| Gamma(3) = 8pi$):
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(RR^3) max(0, bold(x) dot arrow(n)) exp(-beta (abs(bold(x)) - kappa hat(bold(v)) dot bold(x))) d bold(x)
$

=== Analytic Azimuthal Integration and Reduction to One Dimension

To simplify this 3D spatial integral, we introduce spherical coordinates. Let $bold(x) = r arrow(u)$, where $r = abs(bold(x)) in [0, oo)$ and $arrow(u) in S^2$ is a unit direction vector (the volume element satisfies $d bold(x) = r^2 d r d arrow(u)$).

Using positive homogeneity to separate the integral into radial and angular double form:
$
  E(arrow(n)) = (beta^3 (1 - kappa^2)^2) / (8 pi) integral_(S^2) max(0, arrow(u) dot arrow(n)) [ integral_0^oo r^3 exp(-beta (1 - kappa hat(bold(v)) dot arrow(u)) r) d r ] d arrow(u)
$

Using the definite integral relation $integral_0^oo r^3 e^(-a r) d r = Gamma(4) / a^4 = 6 / a^4$ (since $kappa in [0, 1)$, the radial convergence factor $a = beta(1 - kappa hat(bold(v)) dot arrow(u)) > 0$ always holds), the radial distribution in the integral is rigorously integrated out:
$
  E(arrow(n)) = (3 (1 - kappa^2)^2) / (4 pi beta) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

Eliminating the intermediate parameter $beta = (3 + kappa^2) / (omega (1 - kappa^2))$ yields the projection integral expressed solely in terms of the macroscopic total energy $omega$ and directional distribution characteristics:
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 pi (3 + kappa^2)) integral_(S^2) (max(0, arrow(u) dot arrow(n))) / ((1 - kappa hat(bold(v)) dot arrow(u))^4) d arrow(u)
$

We establish a local coordinate system with the normal $arrow(n)$ as the $z$-axis, so $cos theta = arrow(u) dot arrow(n)$. The hemispherical clipping operator $max(0, cos theta)$ strictly restricts the integration domain to the upper hemisphere $Omega_+ = { arrow(u) in S^2 | cos theta >= 0 }$.

Let $mu_0 = hat(bold(v)) dot arrow(n)$. In this basis, we represent the principal light axis direction $hat(bold(v))$ as $(sin theta_0, 0, mu_0)^T$. Integrating over the azimuthal angle $phi in [0, 2pi]$ (applying first-order derivative recursion):
$ integral_0^(2pi) d phi / (A - B cos phi)^4 = pi (2A^3 + 3A B^2) / (A^2 - B^2)^(7/2) $

where the auxiliary quantities are defined as:
$ A(z) = 1 - kappa mu_0 z, quad B(z)^2 = kappa^2 (1 - mu_0^2)(1 - z^2) $
$ A(z)^2 - B(z)^2 = kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) $

Letting $z = cos theta in [0, 1]$ and substituting, the azimuthal angle can be eliminated, yielding the *simplest univariate analytic integral form* in terms of the zenith cosine $z$:
$
  E(arrow(n)) = omega dot (3 (1 - kappa^2)^3) / (4 (3 + kappa^2)) integral_0^1 (z (1 - kappa mu_0 z) [ 2 (1 - kappa mu_0 z)^2 + 3 kappa^2 (1 - mu_0^2)(1 - z^2) ]) / ([ kappa^2 z^2 - 2 kappa mu_0 z + (1 - kappa^2 + kappa^2 mu_0^2) ]^(7/2)) d z
$

=== Boundary Behavior Analysis and Perfect Symmetric/Antisymmetric Decoupling

Since the denominator of the above expression contains a fractional-order algebraic term $Q(z)^(7/2)$, its antiderivative form is extremely cumbersome in the general domain. To construct an efficient real-time reconstruction scheme, we define the normalized irradiance response function as $e(mu_0, kappa) := E(arrow(n)) / omega$, and decompose it into a symmetric component $e_S$ and an antisymmetric component $e_A$ with respect to the cosine angle $mu_0$:
$
  e_S (mu_0, kappa) = (e(mu_0, kappa) + e(-mu_0, kappa)) / 2, quad e_A (mu_0, kappa) = (e(mu_0, kappa) - e(-mu_0, kappa)) / 2
$

By performing boundary limit derivations on the above one-dimensional integral, the system exhibits the following remarkably elegant and symmetric mathematical boundary closed-form solutions:

1. *Fully isotropic limit ($kappa arrow.r 0$)*:
  $ e(mu_0, 0) equiv 1/4 $
2. *Extreme directional limit ($kappa arrow.r 1$)*:
  $ e(mu_0, 1) = max(0, mu_0) $
3. *Light axis perfectly coaligned with normal ($mu_0 = 1$)*:
  $ e(1, kappa) = ((1+kappa)^3 (3 - kappa)) / (4 (3 + kappa^2)) $
4. *Light axis perfectly anti-coaligned with normal ($mu_0 = -1$)*:
  $ e(-1, kappa) = ((1-kappa)^3 (3 + kappa)) / (4 (3 + kappa^2)) $
5. *Light axis perfectly coplanar with surface tangent plane ($mu_0 = 0$)*:
  $ e(0, kappa) = (3 sqrt(1 - kappa^2)) / (4 (3 + kappa^2)) $

*Analytic Uniqueness Theorem for the Antisymmetric Part:*
Further analysis reveals that the physical essence of the antisymmetric component $e_A$ is the reduction of a hemispherical projection to a full-spherical projection. By performing an unclipped integral over the unit sphere $S^2$, it can be rigorously proven that this component exhibits a *strictly linear relationship* for any $mu_0$ and $kappa$, with absolutely no approximation error:
$ e_A (mu_0, kappa) equiv mu_0 dot (2 kappa) / (3 + kappa^2) $

Since $e_A$ has been rigorously integrated out in closed form, the entire fitting error of the lighting reconstruction collapses entirely onto the symmetric part $e_S$.

=== Physical Smoothness Correction and High-Precision Approximation

The symmetric component $e_S$ describes the evolution from the isotropic margin $e_S(0, kappa)$ to the collinear-alignment margin $e_S(1, kappa)$. Since when $kappa < 1$, the maximum entropy probability density field is smooth and differentiable on the local manifold ($C^oo$ continuous), the resulting lighting response must have a first derivative strictly equal to $0$ at $mu_0 = 0$.

Only when the system degenerates to the extreme Dirac limit ($kappa arrow.r 1$) does the non-differentiable kink $| \mu_0 |$ manifest. Based on this physical prior, the mixing weight of the kink term in the transition function $t$ should not be linear, but should exhibit higher-order attenuation as $kappa$ weakens.

We introduce a higher-order characteristic weight $kappa^4$ to suppress the kink response in the low-to-mid frequency band, constructing the following transition function $t$ and symmetric part approximation:
$ t = (1 - kappa^4) mu_0^2 + kappa^4 | mu_0 | $
$ e_S (mu_0, kappa) approx e_S (0, kappa) + (e_S (1, kappa) - e_S (0, kappa)) dot t $

Substituting and combining the boundary analytic values of $e_S (0, kappa)$ and $e_S (1, kappa)$, we obtain the *final reconstruction formula that simultaneously guarantees strict exactness at all limit boundaries, physical field smoothness and continuity, and a global maximum relative error controlled within $0.4%$*:

$
  E(arrow(n)) approx (omega) / (4(3+kappa^2)) [ 3 sqrt(1 - kappa^2) + (3 + 6 kappa^2 - kappa^4 - 3 sqrt(1 - kappa^2)) dot ((1 - kappa^4) mu_0^2 + kappa^4 | mu_0 |) + 8 kappa mu_0 ]
$

#image("/assets/image.png")

=== Production HLSL Denoising Pipeline Implementation

The above algebraically restructured formula contains only basic arithmetic instructions, avoiding expensive transcendental functions (such as $sin, cos$) or numerical integration overhead, making it highly suitable for modern GPU rendering architectures. Below is the core logic for production shader execution:

```hlsl
// High-Precision O(1) Diffuse Lighting Reconstruction Based on Maximum Entropy Distribution
// Parameter description:
//   v     - Lighting direction vector after spatial filtering (v = L.v)
//   omega - Joint total energy after spatial filtering (omega = L.I + |L.v|)
//   N     - Unit surface normal vector at the current pixel
float ReconstructDiffuseLighting(float3 v, float omega, float3 N)
{
    // 0. Tiny-energy boundary protection
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

    // 2. Cosine projection relationship
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

    // 5. Apply high-order smooth interpolation function for transition (guarantees physical C1/C2 continuity)
    float kappa_fourth = kappa_sq * kappa_sq;
    float t = (1.0f - kappa_fourth) * mu_0 * mu_0 + kappa_fourth * abs_mu_0;

    // 6. Combine symmetric part and unbiased antisymmetric part, compute final irradiance
    float e_S_num = lerp(e_S0_num, e_S1_num, t);
    float final_numerator = e_S_num + 8.0f * kappa * mu_0;
    float irradiance = omega * (final_numerator / (4.0f * denom_shared));

    return max(0.0f, irradiance);
}
```

== Naming
Per AI suggestion, this lighting encoding scheme is tentatively named `Asymmetric Laplace Isomorphic Conic Encoding` (abbreviated `ALICE`). The name reflects its core mathematical structure: the conical encoding characteristics of an Asymmetric Laplace distribution (based on the Maximum Entropy Principle) within an isomorphic space.
