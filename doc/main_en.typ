#set text(font: "Microsoft YaHei")

= Technical Documentation

== Preface

=== Background Introduction

Diffuse lighting reconstruction is an important problem in computer graphics. Its purpose is to reconstruct a lighting vector from a set of lighting samples for subsequent rendering. In practical applications, due to the noise in the lighting samples, the reconstructed lighting vector is often affected by noise, so it is necessary to denoise the lighting vector.

=== Purpose

This document aims to introduce the algorithm for diffuse lighting reconstruction, including the encoding, denoising, and reconstruction of the lighting vector, as well as the variance estimation of the denoised lighting vector.

==== Note:

This algorithm was inspired by the first-order spherical harmonics lighting approximation. It is unclear whether there is a similar algorithm, and it may be original. If it is not original, please point it out in the issue.

== Diffuse Lighting Denoising and Reconstruction Algorithm

=== Model

==== Definition
Assume a set of $N+1$ dimensional vectors $V_i$ (the vector set itself is denoted as $V$, the $(N+1)$-th dimension is non-negative) and an energy measurement function $w(v):RR^(N+1)->RR$, $w$ is defined as $ w(v):=sqrt(sum_(i=1)^N v_i^2)+v_(N+1) $

Define the mapping $ T:"List"(RR^(N+1))->RR^(N+1) $ 

where _List_ is an ordered set, $T(V)$ is the mapping of all vectors in $V$ to $RR^(N+1)$, and $T$ satisfies:

+ #box(width: 100%,
[
  Energy Conservation
  
  For any $V in "List"(RR^(N+1))$,
  
  $ w(T(V)) = sum_(i=1)^n w(V_i) $
])
+ #box(width: 100%,
[

  Associativity
  
  For any $x in RR^(N+1)$ and $V in "List"(RR^(N+1))$,

  $ T({x,V}) = T({x,T(V)}) $
])
+ #box(width: 100%, 
[
  Linearity
  
  For any $x in RR^(N+1)$, $n in NN_+$,

  $ T({x,x,...,x}) = n x $

where ${x,x,...,x}$ is an ordered set of $n$ $x$
])

Let $ V_i = (arrow(v)_i, I_i) $ $ Lambda(V_i) := v_i, I(V_i) := I_i $

where
  $ arrow(v)_i in RR^N , I_i in RR_+ $

Then the above mapping $T$ exists, and its form is:

$ T(V) = (sum_(i=1)^n v_i, sum_(i=1)^n omega(V_i) - |sum_(i=1)^n v_i|) $

where

$ omega(V_i) := |v_i| + I_i $

==== Physical Meaning

Select $N=3$, take $V$ as the set of lighting samples, each sample $V_i$ is a lighting vector, the first three dimensions of $V_i$ are directional lighting, the fourth dimension is ambient lighting, $w(V_i)$ is the energy of the lighting, $T$ is the lighting synthesis operator, $T(V)$ is the synthesized lighting, and $T$ satisfies energy conservation, associativity, and linearity.

=== Algorithm

Take the input of the algorithm as a set of lighting vectors $V$ input into the BRDF model, and the output as a lighting vector $T(V)$. The goal of the algorithm is to reduce the noise of the lighting vector and reconstruct the lighting vector.

==== Algorithm Process

+ Encode $V_i$ as $V_i = (I*arrow(d), 0)$, where $I$ is the sampled lighting intensity, $arrow(d)$ is the sampled incident lighting direction, and $0$ is the ambient lighting intensity
+ Filter $V_i$ in the time domain to obtain $V_"out" = 1/n T(V)$, where $n$ is the number of sampled lighting samples
+ Apply the SVGF algorithm to $V_"out"$ for denoising to obtain $V_"out"^'$
+ Input $V_"out"^'$ into the BRDF model to obtain the denoised lighting

==== Algorithm Implementation

+ Actually encode $V_i$ as $V_i = (I*arrow(d), I)$, where $I$ is the sampled lighting intensity, $arrow(d)$ is the sampled incident lighting direction
+ Directly mix $V_i$ during weighted mixing
+ During reconstruction, according to the expression of the $T$ operator, the mixing operation in step 2 is equivalent to the weighted sum of the first three dimensions of $V_i$ and the weighted sum of the fourth dimension, and can correctly construct the result
+ Only apply the $T$ operator during decoding, and in other cases, treat $V_i$ similarly to normal lighting processing

The key code is as follows:
```glsl
struct T
{
    mediump vec4 v_I; // (I * arrow(d), I)
    mediump vec2 CoCg; // (Co, Cg)
};

vec3 project_T_irradiance(T L, vec3 N)
{
    float Y = L.v_I.w;
    float T = Y - L.CoCg.y * 0.5;
    float G = L.CoCg.y + T;
    float B = T - L.CoCg.x * 0.5;
    float R = B + L.CoCg.x;

    vec3 irradiance = vec3(R,G,B) * (max(dot(L.v_I.xyz, N),0) + (Y - length(L.v_I.xyz))) / (Y+1e-3);
    return max(irradiance, vec3(0.0));
}

T irradiance_to_T(vec3 irradiance, vec3 dir)
{
    T result;
    float Co = irradiance.r - irradiance.b;
    float t = irradiance.b + Co * 0.5;
    float Cg = irradiance.g - t;
    float Y = max(t + Cg * 0.5, 0.0);

    result.CoCg = vec2(Co, Cg);
    result.v_I = vec4(dir * Y,Y);
}

T mix_T(T a, T b, float s)
{
    T result;
    result.v_I = mix(a.v_I, b.v_I, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}

T init_T()
{
    T result;
    result.v_I = vec4(0);
    result.CoCg = vec2(0);
    return result;
}

T scaleT(T A, float x) {
    T tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.v_I = A.v_I * x;
    return tmp;
}

void accumulate_T(inout T accum, T b, float scale)
{
    accum.v_I += b.v_I * scale;
    accum.CoCg += b.CoCg * scale;
}
```

==== Lighting Reconstruction

Assuming the denoised lighting $V_"out"^' = (arrow(D),E)$ has been obtained, where $arrow(D)$ is the directional lighting vector and $E$ is the ambient lighting component, for a given BRDF model, the reconstructed lighting can be obtained by the following method:

$ I_o = f_"brdf" (arrow(e),arrow(D),arrow(N)) + integral_omega f_"brdf" (arrow(e),E arrow(omega),arrow(N)) dif omega $

where $I_o$ is the reconstructed lighting intensity, $f_"brdf"$ is the BRDF function, $arrow(e)$ is the viewing direction, $arrow(N)$ is the normal vector, $arrow(omega)$ is the incident lighting direction, and $dif omega$ is the differential solid angle.

*The following images show the directional components*

#image("imgs/2024-12-03_22.53.47.png")

#image("imgs/2024-12-03_22.57.05.png")

#image("imgs/2024-12-03_22.57.46.png")

#image("imgs/2024-12-03_23.02.51.png")

*The following images show the directional components and their diffuse lighting reconstruction results*

#image("imgs/2024-12-03_23.09.46.png")
#image("imgs/2024-12-03_23.10.01.png")
#image("imgs/2024-12-03_23.10.14.png")

It is clear that the directional components of the denoised lighting roughly point to the direction with significant lighting contribution. Therefore, we can reuse this directional component, input it into the sampler for importance sampling, and obtain more accurate lighting reconstruction results.

It is expected that future path tracers will reuse the lighting directional components from the previous frame for importance sampling to obtain more accurate lighting reconstruction results, rather than simply using the BRDF's pdf for aimless sampling.

=== Variance Estimation (Possibly Useful?)

Consider the following optimization problem (temporarily ignoring the $(N+1)$-dimensional component):

$ E(arrow(V)^2) = min  1/n sum_(i=1)^n |arrow(V)_i|^2 $
$ s.t. 1/n sum_(i=1)^n arrow(V)_i = arrow(mu) $
$ s.t. 1/n sum_(i=1)^n |arrow(V)_i| = I $

where $arrow(mu)$ is the expectation of $arrow(V)$, and $I$ is the expectation of $|arrow(V)|$

We need to solve this problem to obtain $E(arrow(V)^2)$

Considering symmetry, let $arrow(V)_i = 1/n arrow(mu) + arrow(e)_i$, satisfying $arrow(mu) dot arrow(e)_i = 0$
and $sum arrow(e)_j = 0$, $|arrow(e)_i| = |arrow(e)_j|$ 

Obviously, the first constraint is satisfied, and the second constraint becomes

$ 1/n sum_(i=1)^n |1/n arrow(mu) + arrow(e)_i| = I $

That is

$ 1/n sum_(i=1)^n sqrt((1/n arrow(mu) + arrow(e)_i)^2) = I $

That is

$ 1/n sum_(i=1)^n sqrt(1/n^2 |arrow(mu)|^2 + 2/n arrow(mu) dot arrow(e)_i + |arrow(e)_i|^2) = I $

Considering $arrow(mu) dot arrow(e)_i = 0$, then

$ 1/n sum_(i=1)^n sqrt(1/n^2 |arrow(mu)|^2 + |arrow(e)_i|^2) = I $

Since $|arrow(e)_i| = |arrow(e)_j|$, then

$ 1/n sum_(i=1)^n sqrt(1/n^2 |arrow(mu)|^2 + |arrow(e)_i|^2) = sqrt(1/n^2 |arrow(mu)|^2 + |arrow(e)_i|^2) $

Thus $ arrow(V_i)^2 = I^2 $

That is $ E(arrow(V)^2) = I^2 $

Considering the original problem, we have $ E(V^2) = 1/n omega(T(V))^2 $

Then the variance estimation is

$ D(T(V)) = 1/n omega(T(V))^2 - 1/n^2 Lambda (T(V))^2 $

Simplified as $D(T(V)) = sigma^2(V)$

=== Properties of Variance

$ sigma^2(lambda V) = lambda^2 sigma^2(V) $
