#ifndef BLOOM_GLSL
#define BLOOM_GLSL
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"

ivec2 bloomOrigin(int l, ivec2 sz) {
    return sz - (sz >> l);
}
ivec2 bloomSize(int l, ivec2 sz) {
    return sz >> (l + 1);
}

// 查找像素所在 LOD 区域。
// 返回 L in [0, 8]，区域边界 [rO, rM]；无效区域返回 L = -1。
void bloomFindLOD(
    ivec2 c,
    ivec2 sz,
    out int L,
    out ivec2 rO,
    out ivec2 rM
) {
    L = -1;
    rO = ivec2(0);
    rM = ivec2(0);

    // 防止 size - c == 0 时传给 findMSB(0u)。
    if (any(lessThan(c, ivec2(0))) ||
        any(greaterThanEqual(c, sz)) ||
        any(lessThanEqual(sz, ivec2(0)))) {
        return;
    }

    ivec2 d = sz - c;

    int Lx = findMSB(uint(sz.x)) - findMSB(uint(d.x));
    int Ly = findMSB(uint(sz.y)) - findMSB(uint(d.y));

    // 修正 findMSB 忽略的尾数部分。
    if (Lx > 0 && uint(d.x) > (uint(sz.x) >> uint(Lx))) {
        --Lx;
    }

    if (Ly > 0 && uint(d.y) > (uint(sz.y) >> uint(Ly))) {
        --Ly;
    }

    // 两个轴不在同一层，或者超过 Bloom Atlas 的最大层数。
    if (Lx != Ly || Lx < 0 || Lx > 8) {
        return;
    }

    ivec2 o = (Lx == 0) ? ivec2(0) : sz - (sz >> Lx);
    ivec2 s = sz >> (Lx + 1);

    if (all(greaterThanEqual(c, o)) &&
        all(lessThan(c, o + s))) {
        L = Lx;
        rO = o;
        rM = o + s - ivec2(1);
    }
}


int bloomKernelR(int diff) {
    return diff <= 0 ? 1 : 1 << (diff - 1);
}

#define BLOOM_SAMPLE(result,img,srcL,dstL,dp,as) do{ \
    ivec2 _srcSize=bloomSize(srcL,as); \
    ivec2 _dstSize=bloomSize(dstL,as); \
    ivec2 _sO=bloomOrigin(srcL,as),_sM=_sO+_srcSize-1; \
    int _diff=(dstL)-(srcL);int _absDiff=_diff>=0?_diff:-_diff; \
    int _R=bloomKernelR(_absDiff); \
    vec2 _dstUV=vec2(dp)/max(vec2(_dstSize)-1.0,vec2(1e-6)); \
    vec2 _scf=vec2(_sO)+_dstUV*max(vec2(_srcSize)-1.0,vec2(0.0)); \
    ivec2 _sc=ivec2(floor(_scf));vec2 _frac=_scf-vec2(_sc); \
    float _S_f=float(1<<_absDiff);float _alpha=LOG2_E/(_S_f*_S_f); \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        vec2 _d=vec2(_dx,_dy)-_frac; \
        float _gw=exp2(-dot(_d,_d)*_alpha); \
        _w+=_gw; \
        ivec2 _sc2=_sc+ivec2(_dx,_dy); \
        if(_sc2.x>=_sO.x&&_sc2.x<=_sM.x&&_sc2.y>=_sO.y&&_sc2.y<=_sM.y) \
            _s+=imageLoad(img,_sc2).rgb*_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)

#define BLOOM_SAMPLE_TEX(result,tex,srcL,dstL,dp,tsz,as) do{ \
    int _srcEq=(srcL)<0?-1:(srcL); \
    ivec2 _srcSize=_srcEq<0?tsz:bloomSize(_srcEq,as); \
    ivec2 _dstSize=bloomSize(dstL,as); \
    ivec2 _sO=_srcEq<0?ivec2(0):bloomOrigin(_srcEq,as); \
    ivec2 _sM=_sO+_srcSize-1; \
    int _diff=(dstL)-_srcEq;int _absDiff=_diff>=0?_diff:-_diff; \
    int _R=bloomKernelR(_absDiff); \
    vec2 _dstUV=vec2(dp)/max(vec2(_dstSize)-1.0,vec2(1e-6)); \
    vec2 _scf=vec2(_sO)+_dstUV*max(vec2(_srcSize)-1.0,vec2(0.0)); \
    ivec2 _sc=ivec2(floor(_scf));vec2 _frac=_scf-vec2(_sc); \
    float _S_f=float(1<<_absDiff);float _alpha=LOG2_E/(_S_f*_S_f); \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        vec2 _d=vec2(_dx,_dy)-_frac; \
        float _gw=exp2(-dot(_d,_d)*_alpha); \
        _w+=_gw; \
        ivec2 _sc2=_sc+ivec2(_dx,_dy); \
        if(_sc2.x>=_sO.x&&_sc2.x<=_sM.x&&_sc2.y>=_sO.y&&_sc2.y<=_sM.y) \
            _s+=texelFetch(tex,_sc2,0).rgb*_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)
#endif
