#ifndef QUATERNIONS_GLSL
#define QUATERNIONS_GLSL

// This "library" only handles normalized quaternions
// This allows for some optimizations

// axis must be normalized
vec4 quatAxisAngle(vec3 axis, float angle) {
    return vec4(sin(angle / 2.0) * axis, cos(angle / 2.0));
}

vec4 quatMultiply(vec4 a, vec4 b) {
    return vec4(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), 
        a.w * b.w - dot(a.xyz, b.xyz)
    );
}

vec4 quatInverse(vec4 q) {
    return vec4(-q.xyz, q.w);
}

vec3 quatRotate(vec3 pos, vec4 q) {
    vec4 qInv = quatInverse(q);
    // Originally q * (pos, 0) * q^1
    // But we don't need to calculate the output w as it's always 0
    vec4 q1 = quatMultiply(q, vec4(pos, 0));
    return q1.w * qInv.xyz + q1.xyz * qInv.w + cross(q1.xyz, qInv.xyz);
}

vec3 quatRotate(vec3 pos, vec3 axis, float angle) {
    vec4 q = quatAxisAngle(axis, angle);
    return quatRotate(pos, q);
}

vec4 getRotationToZAxis(vec3 vec) {
	if (vec.z < -0.99999) return vec4(1.0, 0.0, 0.0, 0.0);

	return normalize(vec4(vec.y, -vec.x, 0.0, 1.0 + vec.z));
}

#endif // QUATERNIONS_GLSL