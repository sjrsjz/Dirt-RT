float dot2(vec2 v) {
    return dot(v, v);
}

float sdSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C)
{
    vec2 a = B - A;
    vec2 b = A - 2.0 * B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;
    float kk = 1.0 / dot(b, b);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float res = 0.0;
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p3;
    if (h >= 0.0)
    {
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) / 2.0;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = dot2(d + (c + b * t) * t);
    }
    else
    {
        float z = sqrt(-p);
        float v = acos(q / (p * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
        res = min(dot2(d + (c + b * t.x) * t.x),
                dot2(d + (c + b * t.y) * t.y));
        // the third root cannot be the closest
        // res = min(res,dot2(d+(c+b*t.z)*t.z));
    }
    return sqrt(res);
}

// a — 5 primitives
float sdf_sga_a(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.76190476, 0.98095238), vec2(-0.45714286, 0.98095238)));
    d = min(d, sdBezier(p, vec2(-0.45714286, 0.98095238), vec2(-0.19047619, 1.00000000), vec2(-0.15238095, 0.67619048)));
    d = min(d, sdSegment(p, vec2(-0.15238095, 0.67619048), vec2(-0.15238095, -0.54285714)));
    d = min(d, sdBezier(p, vec2(-0.15238095, -0.54285714), vec2(-0.09523810, -1.00000000), vec2(0.30476190, -1.00000000)));
    d = min(d, sdBezier(p, vec2(0.30476190, -1.00000000), vec2(0.70476190, -1.00000000), vec2(0.76190476, -0.54285714)));
    return d;
}

// b — 5 primitives
float sdf_sga_b(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.74285714, 0.91428571), vec2(0.17142857, 0.91428571)));
    d = min(d, sdBezier(p, vec2(0.17142857, 0.91428571), vec2(0.74285714, 0.80000000), vec2(0.40000000, 0.45714286)));
    d = min(d, sdSegment(p, vec2(0.40000000, 0.45714286), vec2(-0.05714286, 0.00000000)));
    d = min(d, sdBezier(p, vec2(-0.05714286, 0.00000000), vec2(-0.15238095, -0.09523810), vec2(-0.13333333, -0.30476190)));
    d = min(d, sdSegment(p, vec2(-0.13333333, -0.30476190), vec2(-0.13333333, -0.91428571)));
    return d;
}

// c — 5 primitives
float sdf_sga_c(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.30476190, -0.30476190), vec2(-0.30476190, -0.15238095)));
    d = min(d, sdBezier(p, vec2(-0.30476190, -0.15238095), vec2(-0.26666667, 0.03809524), vec2(0.00000000, 0.00000000)));
    d = min(d, sdBezier(p, vec2(0.00000000, 0.00000000), vec2(0.26666667, -0.03809524), vec2(0.30476190, 0.15238095)));
    d = min(d, sdSegment(p, vec2(0.30476190, 0.15238095), vec2(0.30476190, 0.91428571)));
    d = min(d, length(p - vec2(-0.30476190, -0.91428571)));
    return d;
}

// d — 2 primitives
float sdf_sga_d(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.76190476, -0.51428571), vec2(0.76190476, -0.51428571)));
    d = min(d, sdSegment(p, vec2(-0.76190476, 0.09523810), vec2(0.72380952, 0.51428571)));
    return d;
}

// e — 4 primitives
float sdf_sga_e(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.60000000, -0.92380952), vec2(-0.60000000, 0.60000000)));
    d = min(d, sdBezier(p, vec2(-0.60000000, 0.60000000), vec2(-0.61904762, 0.92380952), vec2(-0.29523810, 0.90476190)));
    d = min(d, sdSegment(p, vec2(-0.29523810, 0.90476190), vec2(0.61904762, 0.90476190)));
    d = min(d, length(p - vec2(0.54285714, -0.80952381)));
    return d;
}

// f — 4 primitives
float sdf_sga_f(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.76190476, -0.30476190), vec2(0.76190476, -0.30476190)));
    d = min(d, length(p - vec2(-0.76190476, 0.30476190)));
    d = min(d, length(p - vec2(0.00000000, 0.30476190)));
    d = min(d, length(p - vec2(0.76190476, 0.30476190)));
    return d;
}

// g — 2 primitives
float sdf_sga_g(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(0.38095238, -0.91428571), vec2(0.38095238, 0.91428571)));
    d = min(d, sdSegment(p, vec2(-0.38095238, 0.00000000), vec2(0.38095238, 0.00000000)));
    return d;
}

// h — 4 primitives
float sdf_sga_h(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.76190476, -0.91428571), vec2(0.76190476, -0.91428571)));
    d = min(d, sdSegment(p, vec2(-0.76190476, -0.30476190), vec2(0.76190476, -0.30476190)));
    d = min(d, sdSegment(p, vec2(0.76190476, -0.30476190), vec2(-0.76190476, -0.30476190)));
    d = min(d, sdSegment(p, vec2(0.00000000, -0.30476190), vec2(0.00000000, 0.91428571)));
    return d;
}

// i — 2 primitives
float sdf_sga_i(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(0.00000000, -0.91428571), vec2(0.00000000, -0.30476190)));
    d = min(d, sdSegment(p, vec2(0.00000000, 0.30476190), vec2(0.00000000, 0.91428571)));
    return d;
}

// j — 3 primitives
float sdf_sga_j(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(0.00000000, -0.91428571), vec2(0.00000000, -0.68571429)));
    d = min(d, sdSegment(p, vec2(0.00000000, -0.15238095), vec2(0.00000000, 0.15238095)));
    d = min(d, sdSegment(p, vec2(0.00000000, 0.68571429), vec2(0.00000000, 0.91428571)));
    return d;
}

// k — 3 primitives
float sdf_sga_k(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(0.00000000, -0.91428571), vec2(0.00000000, 0.91428571)));
    d = min(d, length(p - vec2(-0.76190476, 0.00000000)));
    d = min(d, length(p - vec2(0.76190476, 0.00000000)));
    return d;
}

// l — 3 primitives
float sdf_sga_l(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.38095238, -0.91428571), vec2(-0.38095238, 0.91428571)));
    d = min(d, length(p - vec2(0.38095238, -0.38095238)));
    d = min(d, length(p - vec2(0.38095238, 0.38095238)));
    return d;
}

// m — 4 primitives
float sdf_sga_m(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.60952381, 0.91428571), vec2(0.30476190, 0.91428571)));
    d = min(d, sdBezier(p, vec2(0.30476190, 0.91428571), vec2(0.57142857, 0.91428571), vec2(0.60952381, 0.68571429)));
    d = min(d, sdSegment(p, vec2(0.60952381, 0.68571429), vec2(0.60952381, -0.91428571)));
    d = min(d, length(p - vec2(-0.53333333, -0.80000000)));
    return d;
}

// n — 3 primitives
float sdf_sga_n(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.53333333, -0.95238095), vec2(-0.53333333, -0.34285714)));
    d = min(d, sdBezier(p, vec2(-0.53333333, 0.87619048), vec2(-0.34285714, 0.95238095), vec2(0.07619048, 0.57142857)));
    d = min(d, sdBezier(p, vec2(0.07619048, 0.57142857), vec2(0.47619048, 0.20952381), vec2(0.53333333, -0.95238095)));
    return d;
}

// o — 4 primitives
float sdf_sga_o(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.53333333, -0.91428571), vec2(0.30476190, -0.91428571)));
    d = min(d, sdBezier(p, vec2(0.30476190, -0.91428571), vec2(0.53333333, -0.85714286), vec2(0.53333333, -0.68571429)));
    d = min(d, sdBezier(p, vec2(0.53333333, -0.68571429), vec2(0.51428571, 0.09523810), vec2(0.15238095, 0.53333333)));
    d = min(d, sdBezier(p, vec2(0.15238095, 0.53333333), vec2(-0.13333333, 0.89523810), vec2(-0.53333333, 0.91428571)));
    return d;
}

// p — 4 primitives
float sdf_sga_p(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.38095238, -0.91428571), vec2(-0.38095238, 0.30476190)));
    d = min(d, sdSegment(p, vec2(0.38095238, -0.30476190), vec2(0.38095238, 0.91428571)));
    d = min(d, length(p - vec2(0.38095238, -0.91428571)));
    d = min(d, length(p - vec2(-0.38095238, 0.91428571)));
    return d;
}

// q — 6 primitives
float sdf_sga_q(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.60952381, -0.30476190), vec2(0.30476190, -0.30476190)));
    d = min(d, sdBezier(p, vec2(0.30476190, -0.30476190), vec2(0.57142857, -0.30476190), vec2(0.60952381, -0.07619048)));
    d = min(d, sdSegment(p, vec2(0.60952381, -0.07619048), vec2(0.60952381, 0.68571429)));
    d = min(d, sdBezier(p, vec2(0.60952381, 0.68571429), vec2(0.57142857, 0.91428571), vec2(0.30476190, 0.91428571)));
    d = min(d, sdSegment(p, vec2(0.30476190, 0.91428571), vec2(-0.60952381, 0.91428571)));
    d = min(d, length(p - vec2(0.00000000, -0.91428571)));
    return d;
}

// r — 4 primitives
float sdf_sga_r(vec2 p) {
    float d = 1e9;
    d = min(d, length(p - vec2(-0.45714286, -0.45714286)));
    d = min(d, length(p - vec2(0.45714286, 0.45714286)));
    d = min(d, length(p - vec2(-0.45714286, 0.45714286)));
    d = min(d, length(p - vec2(0.45714286, -0.45714286)));
    return d;
}

// s — 5 primitives
float sdf_sga_s(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.30476190, -0.91428571), vec2(-0.30476190, -0.22857143)));
    d = min(d, sdBezier(p, vec2(-0.30476190, -0.22857143), vec2(-0.30476190, 0.00000000), vec2(-0.07619048, 0.00000000)));
    d = min(d, sdSegment(p, vec2(-0.07619048, 0.00000000), vec2(0.07619048, 0.00000000)));
    d = min(d, sdBezier(p, vec2(0.07619048, 0.00000000), vec2(0.30476190, 0.00000000), vec2(0.30476190, 0.22857143)));
    d = min(d, sdSegment(p, vec2(0.30476190, 0.22857143), vec2(0.30476190, 0.91428571)));
    return d;
}

// t — 4 primitives
float sdf_sga_t(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.45714286, -0.91428571), vec2(0.15238095, -0.91428571)));
    d = min(d, sdBezier(p, vec2(0.15238095, -0.91428571), vec2(0.41904762, -0.87619048), vec2(0.45714286, -0.60952381)));
    d = min(d, sdSegment(p, vec2(0.45714286, -0.60952381), vec2(0.45714286, 0.30476190)));
    d = min(d, length(p - vec2(0.45714286, 0.91428571)));
    return d;
}

// u — 3 primitives
float sdf_sga_u(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.91428571, 0.30476190), vec2(0.91428571, 0.30476190)));
    d = min(d, length(p - vec2(-0.41904762, -0.30476190)));
    d = min(d, length(p - vec2(0.41904762, -0.30476190)));
    return d;
}

// v — 3 primitives
float sdf_sga_v(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.76190476, 0.30476190), vec2(0.76190476, 0.30476190)));
    d = min(d, sdSegment(p, vec2(0.00000000, -0.91428571), vec2(0.00000000, 0.30476190)));
    d = min(d, sdSegment(p, vec2(-0.76190476, 0.91428571), vec2(0.76190476, 0.91428571)));
    return d;
}

// w — 3 primitives
float sdf_sga_w(vec2 p) {
    float d = 1e9;
    d = min(d, length(p - vec2(0.00000000, -0.45714286)));
    d = min(d, length(p - vec2(-0.45714286, 0.45714286)));
    d = min(d, length(p - vec2(0.45714286, 0.45714286)));
    return d;
}

// x — 2 primitives
float sdf_sga_x(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.45714286, 0.91428571), vec2(0.45714286, -0.91428571)));
    d = min(d, length(p - vec2(-0.45714286, -0.80000000)));
    return d;
}

// y — 2 primitives
float sdf_sga_y(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.38095238, -0.91428571), vec2(-0.38095238, 0.91428571)));
    d = min(d, sdSegment(p, vec2(0.38095238, -0.91428571), vec2(0.38095238, 0.91428571)));
    return d;
}

// z — 5 primitives
float sdf_sga_z(vec2 p) {
    float d = 1e9;
    d = min(d, sdSegment(p, vec2(-0.53333333, 0.91428571), vec2(-0.53333333, -0.60952381)));
    d = min(d, sdBezier(p, vec2(-0.53333333, -0.60952381), vec2(-0.53333333, -0.87619048), vec2(-0.30476190, -0.91428571)));
    d = min(d, sdSegment(p, vec2(-0.30476190, -0.91428571), vec2(0.30476190, -0.91428571)));
    d = min(d, sdBezier(p, vec2(0.30476190, -0.91428571), vec2(0.53333333, -0.87619048), vec2(0.53333333, -0.60952381)));
    d = min(d, sdSegment(p, vec2(0.53333333, -0.60952381), vec2(0.53333333, 0.91428571)));
    return d;
}

float sdf_sga(int idx, vec2 p) {
    switch (idx) {
        case 0:
        return sdf_sga_a(p);
        case 1:
        return sdf_sga_b(p);
        case 2:
        return sdf_sga_c(p);
        case 3:
        return sdf_sga_d(p);
        case 4:
        return sdf_sga_e(p);
        case 5:
        return sdf_sga_f(p);
        case 6:
        return sdf_sga_g(p);
        case 7:
        return sdf_sga_h(p);
        case 8:
        return sdf_sga_i(p);
        case 9:
        return sdf_sga_j(p);
        case 10:
        return sdf_sga_k(p);
        case 11:
        return sdf_sga_l(p);
        case 12:
        return sdf_sga_m(p);
        case 13:
        return sdf_sga_n(p);
        case 14:
        return sdf_sga_o(p);
        case 15:
        return sdf_sga_p(p);
        case 16:
        return sdf_sga_q(p);
        case 17:
        return sdf_sga_r(p);
        case 18:
        return sdf_sga_s(p);
        case 19:
        return sdf_sga_t(p);
        case 20:
        return sdf_sga_u(p);
        case 21:
        return sdf_sga_v(p);
        case 22:
        return sdf_sga_w(p);
        case 23:
        return sdf_sga_x(p);
        case 24:
        return sdf_sga_y(p);
        case 25:
        return sdf_sga_z(p);
        default:
        return 1e9;
    }
}
