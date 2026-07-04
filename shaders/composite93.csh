#version 430 compatibility
// 级联下采样 L3 -> L4+L5+L6
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f) uniform image2D bloomAtlas;
#include "/lib/post_processing/bloom.glsl"
void main(){
    ivec2 localPix=ivec2(gl_GlobalInvocationID.xy);
    ivec2 atlasSize=imageSize(bloomAtlas);
    vec3 color;
    for(int dstLevel=4;dstLevel<=6;dstLevel++){
        ivec2 dstSize=bloomSize(dstLevel,atlasSize);
        if(localPix.x>=dstSize.x||localPix.y>=dstSize.y)continue;
        BLOOM_SAMPLE(color,bloomAtlas,3,dstLevel,localPix,atlasSize);
        imageStore(bloomAtlas,bloomOrigin(dstLevel,atlasSize)+localPix,vec4(color,1.0));
    }
}
