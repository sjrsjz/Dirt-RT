#ifndef OCT_ENCODE_GLSL
#define OCT_ENCODE_GLSL

// 辅助函数：安全的非零符号函数
float sign_not_zero(float v) {
    return (v >= 0.0 ? 1.0 : -1.0);
}
vec2 sign_not_zero(vec2 v) {
    return vec2(sign_not_zero(v.x), sign_not_zero(v.y));
}

// 八面体编码：vec3 → float
float encodeNormal(vec3 n) {
    n = normalize(n);
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * sign_not_zero(p);
    }
    p = p * 0.5 + 0.5;
    uint packed_ = packUnorm2x16(p);
    return uintBitsToFloat(packed_);
}

// 八面体解码：float → vec3
vec3 decodeNormal(float f) {
    uint packed_ = floatBitsToUint(f);
    vec2 p = unpackUnorm2x16(packed_);
    p = p * 2.0 - 1.0;
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * sign_not_zero(n.xy);
    }
    return normalize(n);
}

#endif // OCT_ENCODE_GLSL
