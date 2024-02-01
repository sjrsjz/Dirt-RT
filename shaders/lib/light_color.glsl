#ifndef LIGHT_COLOR_GLSL
#define LIGHT_COLOR_GLSL
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/common.glsl"
float S_R = 0.05;
float cosD_S = 1 / sqrt(1 + S_R * S_R);
vec3 b_P = vec3(300000); //atmosphere thickness
float b_k = 0.25; //mix

vec3 Mie = vec3(0.2);

vec3 Rayleigh = 4e10 * pow(vec3(1. / 700, 1. / 520, 1. / 450), vec3(4));

vec3 b_k0 = mix(Rayleigh, Mie, b_k);

vec3 b_Q = b_k0 / (b_P * b_P); //absorption
vec3 b_g0 = mix(Rayleigh * 0.01, vec3(0.9), b_k); //single scatter

void setSkyVars() {
    S_R = 0.05;
    cosD_S = 1 / sqrt(1 + S_R * S_R);

    switch (world_type_global) {
        case World_THE_END:
        S_R = 0.25;
        cosD_S = 1 / sqrt(1 + S_R * S_R);
        Mie = vec3(0.01);
        Rayleigh = 1e9 * pow(vec3(1. / 700, 1. / 520, 1. / 450), vec3(4));
        b_P = vec3(4096);
        b_k = 0.75;
        break;
        case World_THE_NETHER:
        S_R = 0.05;
        cosD_S = 1 / sqrt(1 + S_R * S_R);
        Mie = vec3(0.2);
        Rayleigh = 4e11 * pow(vec3(1. / 700, 1. / 520, 1. / 450), vec3(4));
        b_P = vec3(600000);
        b_k = 0.5;
        break;
        default:
        S_R = 0.05;
        cosD_S = 1 / sqrt(1 + S_R * S_R);
        Mie = vec3(0.2);
        Rayleigh = 4e10 * pow(vec3(1. / 700, 1. / 520, 1. / 450), vec3(4));
        b_P = vec3(300000);
        b_k = 0.25 + rainStrength_global * 0.75;
        break;
    }

    b_k0 = mix(Rayleigh, Mie, b_k);
    b_Q = b_k0 / (b_P * b_P); //absorption
    b_g0 = mix(Rayleigh * 0.01, vec3(0.9), b_k); //single scatter
}
vec3 getSkyColor(vec3 b_Sun, vec3 b_Moon, in vec3 pos, in vec3 n, in vec3 lightDir) {
    vec3 n0 = n;
    n.y = max(n.y, 1e-5);
    vec3 lightDir1 = -lightDir;
    vec3 g = 3. / (8. * PI) * (1. + pow(dot(n, lightDir), 2.)) * (1. - b_g0 * b_g0) / (2. + b_g0 * b_g0) / pow(1. + b_g0 * b_g0 - 2. * b_g0 * dot(lightDir, n), vec3(1.5));
    vec3 t = b_Q * 0.5 * (b_P - pos.y) * (b_P - pos.y);
    vec3 c = b_Sun * g * (exp(-t / n.y) - exp(-t / lightDir.y)) / (n.y - lightDir.y) * max(lightDir.y, 0.);

    //g=3./(8.*PI)*(1.+pow(dot(n,lightDir1),2.))*(1.-b_g0*b_g0)/(2.+b_g0*b_g0)/pow(1.+b_g0*b_g0-2.*b_g0*dot(lightDir1,n),vec3(1.5));
    //t=b_Q*0.5*(b_P-pos.y)*(b_P-pos.y);
    //c+=b_Moon*g*(exp(-t/n.y)-exp(-t/lightDir1.y))/(n.y-lightDir1.y)*max(lightDir1.y,0.);

    c += exp(-t / n.y) * b_Sun * exp(-sqrt(abs(min(dot(n0, lightDir) - cosD_S, 0)) * 6000));
    c += exp(-t / n.y) * b_Moon * exp(-sqrt(abs(min(dot(n0, lightDir1) - cosD_S, 0)) * 15000));
    return abs(c);
}
vec3 getFogColor(vec3 b_Sun, vec3 b_Moon, in vec3 pos, in vec3 n, in vec3 lightDir, float s, vec3 col) {
    vec3 n0 = n;
    if (n.y > 0) s = min((b_P.x - pos.y) / n.y, s);
    vec3 g = 3. / (8. * PI) * (1. + pow(dot(n, lightDir), 2.)) * (1. - b_g0 * b_g0) / (2. + b_g0 * b_g0) / pow(1. + b_g0 * b_g0 - 2. * b_g0 * dot(lightDir, n), vec3(1.5));
    vec3 t = b_Q * 0.5 * (b_P - pos.y) * (b_P - pos.y);
    vec3 s1 = exp(b_Q * s * (0.5 * s * n.y - (b_P - pos.y)) * (1 - n.y / lightDir.y));
    vec3 c = 0 * b_Sun * g * exp(-t / lightDir.y) * (1 - s1) / (-n.y + lightDir.y) * max(lightDir.y, 0.);
    c += exp(b_Q * 0.5 * n.y * s * s - b_Q * (b_P - pos.y) * s) * col;
    return c;
}

vec4 fbm3D2(in vec3 x, int n)
{
    const float scale = 4.;

    float a = 0.0;
    float b = 0.5;
    float f = 1.0;
    vec3 d = vec3(0.0);
    for (int i = 0; i < n; i++)
    {
        vec4 n = noised(f * x * scale+time_global * 0.0025 * scale);
        a += b * n.x; // accumulate values
        d += b * n.yzw * f * scale; // accumulate derivatives
        b *= 0.6; // amplitude decrease
        f *= 2; // frequency increase
    }

    return vec4(a, d);
}
float cloud_density(vec3 p) {
    float density = 0.01 + smoothstep(3000., 4000., p.y) * smoothstep(4000., 6000., p.y) * 0.15;
    density+=rainStrength_global * 0.25;
    return clamp(fbm3D2(vec3(0.00000125,0.00000325,0.00000125) * p, 5).x - 1 + density, 0, 1) / density;
}
vec3 getClouds(vec3 b_Sun, vec3 b_Moon, vec3 pos, vec3 n, vec3 lightDir, float Far) {
    if(world_type_global!=0) 
        return getSkyColor(b_Sun, b_Moon, pos, n, lightDir);
    const int step1 = 5;
    const int step2 = 5;

    vec3 c;
    if (lightDir.y > 5e-4) {
        float L = n.y < 1e-2 ? Far : min((b_P.x - pos.y) / n.y, Far);
        vec3 pos1 = pos + n * L;
        c = getSkyColor(b_Sun, b_Moon, pos1, n, lightDir).xyz;
        float s0 = 0;
            vec3 b_k1 = mix(Rayleigh, Mie, 1) * 250 / b_P / b_P;

        for (int i = 0; i < step1; i++) {
            float s = (n.y < 5e-2 ? L / step1 : (b_P.x - pos.y) / (n.y * step1)) * (rand(pos1) + 0.5);
            s0 += s;
            if (s0 > L) break;
            pos1 -= n * s;
            float d = cloud_density(pos1);

            vec3 b_Q1 = mix(b_Q, b_k1, d); //absorption
            vec3 b_g1 = mix(b_g0, vec3(0), d);
            vec3 t = b_Q1 * 0.5 * (2 * (b_P - pos1.y) * s - s * s * n.y);
            vec3 g = 3. / (8. * PI) * (1. + pow(dot(n, lightDir), 2.)) * (1. - b_g1 * b_g1) / (2. + b_g1 * b_g1) / pow(1. + b_g1 * b_g1 - 2. * b_g1 * dot(lightDir, n), vec3(1.5));
            float s2 = (b_P.x - pos1.y) / (lightDir.y * step2);
            vec3 c1 = b_Sun;
            vec3 c0 = vec3(1);
            for (int j = 0; j < step2; j++) {
                vec3 pos2 = pos1 + lightDir * (step2 - j) * s2 * (0.5 + rand(pos1));
                float d = cloud_density(pos2);
                vec3 b_Q1 = mix(b_Q, b_k1, d); //absorption
                vec3 t = b_Q1 * 0.5 * (2 * (b_P - pos2.y) * s2 - s2 * s2 * lightDir.y);
                c0 += max(t, 0);
            }
            c += exp(-c0) * c1 * g * b_Q1 * (b_P - pos1.y - s * n.y) * s;
            c *= exp(-max(t, 0));
        }
    } else {
        c = getSkyColor(b_Sun, b_Moon, pos, n, lightDir).xyz;
    }

    return c;
}

layout(std140, set = 3, binding = 5) buffer SkyBuffer {
    mat3x3 data[];
}skyBuffer;

const uint SkyW=512;
const uint SkyH=512;
uint getSkyBufferIdx(ivec2 uv){
    if(uv.y>int(SkyH)-1){
        uv.y=2*int(SkyH)-uv.y-1;
        uv.x=-uv.x;
    }
    if(uv.y<0){
        uv.y=-uv.y;
        uv.x=-uv.x;
    }
    if(uv.x>int(SkyW)-1){
        uv.x-=int(SkyW);
    }
    if(uv.x<0){
        uv.x+=int(SkyW);
    }

    return uint(uv.y*SkyW+uv.x);
}
void GenSky(vec3 b_Sun,vec3 b_Moon,vec3 lightDir,vec3 pos,ivec2 uv){

    float A=2*PI/SkyW*uv.x;
    float B=(clamp(float(uv.y)/(SkyH-1)*2-1,-1,1))*PI/2;
    vec3 n=vec3(cos(B)*cos(A),sin(B),cos(B)*sin(A));
    rand_i=sin(frame_id)*50+0.4;
    vec3 c=getClouds(b_Sun,b_Moon,pos,n,lightDir,1000000);

    skyBuffer.data[getSkyBufferIdx(uv)][0]=mix(c,skyBuffer.data[getSkyBufferIdx(uv)][0],0.975);
    if(any(isnan(skyBuffer.data[getSkyBufferIdx(uv)][0]))) skyBuffer.data[getSkyBufferIdx(uv)][0]=vec3(0);
}
const int blurR=8;
const int blurSize=2*blurR+1;

void BlurSkyX(ivec2 uv){
    vec3 c=vec3(0);
    float w=0;
    for(int i=-blurR;i<=blurR;i++){
        float w0=exp(-0.25*i*i);
        c+=w0*skyBuffer.data[getSkyBufferIdx(uv+ivec2(i,0))][0];
        w+=w0;
    }
    skyBuffer.data[getSkyBufferIdx(uv)][1]=c/w;
}
void BlurSkyY(ivec2 uv){
    vec3 c=vec3(0);
    float w=0;
    for(int i=-blurR;i<=blurR;i++){
        float w0=exp(-0.25*i*i);
        c+=w0*skyBuffer.data[getSkyBufferIdx(uv+ivec2(0,i))][1];
        w+=w0;
    }
    skyBuffer.data[getSkyBufferIdx(uv)][2]=c/w;
}

vec3 SampleSky(vec3 n){
    float A=atan(n.z,n.x)/(2*PI)*(SkyW);
    float B=(asin(n.y)+PI/2)/PI*(SkyH-1);
    vec2 p=vec2(fract(A),fract(B));
    ivec2 p1=ivec2(A,B);
    vec3 c0=skyBuffer.data[getSkyBufferIdx(p1)][2];
    vec3 c1=skyBuffer.data[getSkyBufferIdx(p1+ivec2(1,0))][2];
    vec3 c2=skyBuffer.data[getSkyBufferIdx(p1+ivec2(0,1))][2];
    vec3 c3=skyBuffer.data[getSkyBufferIdx(p1+ivec2(1,1))][2];
    return mix(mix(c0,c1,p.x),mix(c2,c3,p.x),p.y);
}

#endif // LIGHT_COLOR_GLSL
