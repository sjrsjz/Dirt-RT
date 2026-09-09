"""Numerical and source-contract audit for primary specular qLi guiding."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from shader_compile import expand


ROOT = Path(__file__).resolve().parents[1]


def ggx_lambda(no_x: float, alpha: float) -> float:
    no_x = max(abs(no_x), 1.0e-6)
    return 0.5 * (math.sqrt(1.0 + alpha * alpha
        * (1.0 / (no_x * no_x) - 1.0)) - 1.0)


def audit_cancelled_response(cases: int = 100_000) -> float:
    rng = random.Random(0x514C49)
    worst = 0.0
    for _ in range(cases):
        no_v = 10.0 ** rng.uniform(-5.0, 0.0)
        no_l = 10.0 ** rng.uniform(-5.0, 0.0)
        alpha = 10.0 ** rng.uniform(-4.0, 0.0)
        no_h = 10.0 ** rng.uniform(-5.0, 0.0)
        f = rng.random()
        a2 = alpha * alpha
        b = 1.0 + (a2 - 1.0) * no_h * no_h
        d = a2 / (math.pi * b * b)
        lv = ggx_lambda(no_v, alpha)
        ll = ggx_lambda(no_l, alpha)
        g1 = 1.0 / (1.0 + lv)
        g2 = 1.0 / (1.0 + lv + ll)
        q = d * g1 / (4.0 * no_v)
        f_no_l = f * d * g2 / (4.0 * no_v)
        direct = f_no_l / q
        cancelled = f * (1.0 + lv) / (1.0 + lv + ll)
        worst = max(worst, abs(direct - cancelled)
            / max(abs(direct), 1.0e-30))
    return worst


def audit_mixture_measure(cases: int = 10_000) -> float:
    rng = random.Random(0x504D4958)
    worst = 0.0
    for _ in range(cases):
        count = 8
        q = [rng.random() + 1.0e-4 for _ in range(count)]
        g = [rng.random() + 1.0e-4 for _ in range(count)]
        sq, sg = sum(q), sum(g)
        q = [x / sq for x in q]
        g = [x / sg for x in g]
        beta = rng.random() * 0.75
        p = [(1.0 - beta) * q[i] + beta * g[i]
             for i in range(count)]
        li = [rng.random() * 100.0 for _ in range(count)]
        response = [rng.random() + 1.0e-3 for _ in range(count)]

        recovered_q_li = sum(p[i]
            * ((response[i] * q[i] * li[i] / p[i]) / response[i])
            for i in range(count))
        exact_q_li = sum(q[i] * li[i] for i in range(count))
        worst = max(worst, abs(recovered_q_li - exact_q_li)
            / max(abs(exact_q_li), 1.0e-30))
    return worst


def audit_sources() -> None:
    read = lambda p: (ROOT / p).read_text(encoding="utf-8")
    buffer_source = read("shaders/lib/buffers/specular_buffer.glsl")
    transport = expand(ROOT / "shaders/lib/rt/raytrace/transport.glsl")
    bounces = read("shaders/lib/rt/raytrace/bounces.glsl")
    trace = read("shaders/lib/rt/raytrace/path_trace.glsl")
    output = read("shaders/lib/rt/raytrace/gbuffer_io.glsl")
    primary = read("shaders/lib/rt/raytrace/primary_pass.glsl")
    rgen = read("shaders/lib/rt/raytrace_rgen.glsl")
    ray4 = read("shaders/ray4.rgen")
    camera_state = read("shaders/lib/rt/camera_state.glsl")
    temporal = read("shaders/post/denoiser/reflection/temporal.glsl")

    assert "addr(SPEC_N_HISTMETA, xy)" in buffer_source
    assert "computeSpecularMaxEntGuide" in transport
    assert "readMaxEntSpecularPreparedSurfaceDenoised" in transport
    assert "(1.0 - guide.prob) * vndfPdf" in transport
    assert "pdfNDF, qLiResponse" in bounces
    assert "qLiResponse * (pdfNDF / sampledStrategyPdf)" in bounces
    assert trace.count("specularGuideMixturePdf(") == 1
    assert "pdfNDF * misWeight" in trace
    assert "(f/p)*Li" in output and "(q/p)*Li" in output
    assert "reflectionFirstBsdfWeight" not in trace
    assert "fb.motionValid, rtViewProjection, currentViewProjection" in primary
    assert "rtPrevViewProjection =" not in rgen
    assert "publishRtCameraState(currentCamera);" in ray4
    assert "rtPrevViewProjection = rtViewProjection" in camera_state
    assert "false, true, false" in temporal
    assert "readMaxEntSpecularPreparedSurfaceDenoised(" in temporal


def main() -> None:
    audit_sources()
    response_error = audit_cancelled_response()
    mixture_error = audit_mixture_measure()
    assert response_error < 2.0e-12
    assert mixture_error < 2.0e-12
    print(json.dumps({
        "cancelled_response_cases": 100_000,
        "cancelled_response_max_relative_error": response_error,
        "mixture_measure_cases": 10_000,
        "mixture_measure_max_relative_error": mixture_error,
        "guide_probability_cap": 0.75,
        "history_source": "ray0 shared surface reprojection: N3 history -> N4 scratch",
        "scope": "CPU algebra and source contract; GPU behavior requires in-game validation",
    }, indent=2))


if __name__ == "__main__":
    main()
