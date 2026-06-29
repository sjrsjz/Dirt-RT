#ifndef BLOOM_GLSL
#define BLOOM_GLSL
#include "/lib/settings.glsl"

ivec2 bloomOrigin(int l, ivec2 sz) {
    return sz - (sz >> l);
}
ivec2 bloomSize(int l, ivec2 sz) {
    return sz >> (l + 1);
}

// 查找像素所在LOD区域, 返回 L∈[0,8], 区域边界 [rO,rM]; 不在任何区域返回 L=-1
void bloomFindLOD(ivec2 c, ivec2 sz, out int L, out ivec2 rO, out ivec2 rM) {
    L = -1;
    rO = ivec2(0);
    rM = ivec2(0);

    // 1. 物理对数估算 (利用 GPU 硬件级单周期 SFU log2 指令)
    // 使用 max(1e-7, ...) 防止越界造成 log2(0) 产生 NaN
    float tx = float(c.x) / float(sz.x);
    float ty = float(c.y) / float(sz.y);
    int Lx = int(floor(-log2(max(1e-7, 1.0 - tx))));
    int Ly = int(floor(-log2(max(1e-7, 1.0 - ty))));

    // 2. 几何拒绝：如果 X 和 Y 的级数不相等，或者超出 Mip 范围，
    // 说明 100% 处于黑色无效区域，瞬间退出！(拯救 75% 的像素算力)
    if (Lx != Ly || Lx < 0 || Lx > 8) {
        return;
    }

    // 3. 整数位移精确验证 (消除浮点数在高层级下的舍入误差和空洞)
    ivec2 o = (Lx == 0) ? ivec2(0) : sz - (sz >> Lx);
    ivec2 s = sz >> (Lx + 1);

    // 只要通过最终的 AABB 验证，则当前估算级数 100% 物理正确
    if (c.x >= o.x && c.x < o.x + s.x && c.y >= o.y && c.y < o.y + s.y) {
        L = Lx;
        rO = o;
        rM = o + s - ivec2(1, 1);
    }
}

int bloomKernelR(int diff) {
    return diff <= 0 ? 1 : 1 << (diff - 1);
}

#define BLOOM_SAMPLE(result,img,srcL,dstL,dp,as) do{ \
    ivec2 _sO=bloomOrigin(srcL,as),_sM=_sO+bloomSize(srcL,as)-1; \
    int _diff=(dstL)-(srcL);ivec2 _sc;int _R;int _absDiff; \
    if(_diff>=0){int _S=1<<_diff;_sc=_sO+(dp)*_S;_R=bloomKernelR(_diff);_absDiff=_diff;} \
    else{int _S=1<<(-_diff);_sc=_sO+(dp)/_S;_R=bloomKernelR(-_diff);_absDiff=-_diff;} \
    float _S_f=float(1<<_absDiff);float _alpha=1.0/(_S_f*_S_f); \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        float _gw=exp(-float(_dx*_dx+_dy*_dy)*_alpha); \
        _s+=imageLoad(img,clamp(_sc+ivec2(_dx,_dy),_sO,_sM)).rgb*_gw;_w+=_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)

#define BLOOM_SAMPLE_TEX(result,tex,srcL,dstL,dp,tsz) do{ \
    int _srcEq=(srcL)<0?-1:(srcL); \
    ivec2 _sO=_srcEq<0?ivec2(0):bloomOrigin(_srcEq,tsz); \
    ivec2 _sM=_srcEq<0?tsz-1:_sO+bloomSize(_srcEq,tsz)-1; \
    int _diff=(dstL)-_srcEq;ivec2 _sc;int _R;int _absDiff; \
    if(_diff>=0){int _S=1<<_diff;_sc=_sO+(dp)*_S;_R=bloomKernelR(_diff);_absDiff=_diff;} \
    else{int _S=1<<(-_diff);_sc=_sO+(dp)/_S;_R=bloomKernelR(-_diff);_absDiff=-_diff;} \
    float _S_f=float(1<<_absDiff);float _alpha=1.0/(_S_f*_S_f); \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        float _gw=exp(-float(_dx*_dx+_dy*_dy)*_alpha); \
        _s+=texelFetch(tex,clamp(_sc+ivec2(_dx,_dy),_sO,_sM),0).rgb*_gw;_w+=_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)
#endif
