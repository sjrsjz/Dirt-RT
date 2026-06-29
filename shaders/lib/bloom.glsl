#ifndef BLOOM_GLSL
#define BLOOM_GLSL
#include "/lib/settings.glsl"

ivec2 bloomOrigin(int l,ivec2 sz){return sz-(sz>>l);}
ivec2 bloomSize(int l,ivec2 sz){return sz>>(l+1);}

// 查找像素所在LOD区域, 返回 L∈[0,8], 区域边界 [rO,rM]; 不在任何区域返回 L=-1
void bloomFindLOD(ivec2 c,ivec2 sz,out int L,out ivec2 rO,out ivec2 rM){
    L=-1;
    for(int i=0;i<=8;i++){
        ivec2 o=(i==0)?ivec2(0):sz-(sz>>i), s=sz>>(i+1);
        if(c.x>=o.x&&c.x<o.x+s.x&&c.y>=o.y&&c.y<o.y+s.y){L=i;rO=o;rM=o+s-ivec2(1,1);return;}
    }
}


int bloomKernelR(int diff){return diff<=0?1:1<<(diff-1);}

#define BLOOM_SAMPLE(result,img,srcL,dstL,dp,as) do{ \
    ivec2 _sO=bloomOrigin(srcL,as),_sM=_sO+bloomSize(srcL,as)-1; \
    int _diff=(dstL)-(srcL);ivec2 _sc;int _R; \
    if(_diff>=0){int _S=1<<_diff;_sc=_sO+(dp)*_S+(_S>>1);_R=bloomKernelR(_diff);} \
    else{int _S=1<<(-_diff);_sc=_sO+(dp)/_S;_R=bloomKernelR(-_diff);} \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        float _gw=exp(-float(_dx*_dx+_dy*_dy)*.25); \
        _s+=imageLoad(img,clamp(_sc+ivec2(_dx,_dy),_sO,_sM)).rgb*_gw;_w+=_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)

#define BLOOM_SAMPLE_TEX(result,tex,srcL,dstL,dp,tsz) do{ \
    int _srcEq=(srcL)<0?-1:(srcL); \
    ivec2 _sO=_srcEq<0?ivec2(0):bloomOrigin(_srcEq,tsz); \
    ivec2 _sM=_srcEq<0?tsz-1:_sO+bloomSize(_srcEq,tsz)-1; \
    int _diff=(dstL)-_srcEq;ivec2 _sc;int _R; \
    if(_diff>=0){int _S=1<<_diff;_sc=_sO+(dp)*_S+(_S>>1);_R=bloomKernelR(_diff);} \
    else{int _S=1<<(-_diff);_sc=_sO+(dp)/_S;_R=bloomKernelR(-_diff);} \
    vec3 _s=vec3(0);float _w=0.; \
    for(int _dy=-_R;_dy<=_R;_dy++)for(int _dx=-_R;_dx<=_R;_dx++){ \
        float _gw=exp(-float(_dx*_dx+_dy*_dy)*.25); \
        _s+=texelFetch(tex,clamp(_sc+ivec2(_dx,_dy),_sO,_sM),0).rgb*_gw;_w+=_gw; \
    } \
    (result)=_s/max(_w,1e-5); \
}while(false)
#endif
