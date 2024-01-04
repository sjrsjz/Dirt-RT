#ifndef RAND_GLSL
#define RAND_GLSL

uint state;

// https://github.com/riccardoscalco/glsl-pcg-prng
uint rand() {
	state = state * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	state = (word >> 22u) ^ word;
    return state;
}

float randFloat() {
    return float(rand() & uvec3(0x7fffffffU)) / float(0x7fffffff);
}

vec2 randVec2() {
    return vec2(randFloat(), randFloat());
}

void initRNG(uvec2 pixel, uint frame) {
    state = frame;
    state = (pixel.x + pixel.y * 14352113u) ^ rand();
    rand();
}

#endif // RAND_GLSL