#version 430 compatibility
// 下采样 colortex0 -> bloomAtlas L0
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f) uniform writeonly image2D bloomAtlas;
uniform sampler2D colortex0;
#include "/lib/bloom.glsl"
void main(){
    ivec2 localPix=ivec2(gl_GlobalInvocationID.xy);
    ivec2 atlasSize=imageSize(bloomAtlas);
    ivec2 dstSize=bloomSize(0,atlasSize);
    if(localPix.x>=dstSize.x||localPix.y>=dstSize.y)return;
    vec3 color;BLOOM_SAMPLE_TEX(color,colortex0,-1,0,localPix,textureSize(colortex0,0),atlasSize);
    imageStore(bloomAtlas,localPix,vec4(color,1.0));
}
