#ifndef DIRT_RT_LIB_LIGHTING_SPECULAR_CDF_RESPONSE_GLSL
#define DIRT_RT_LIB_LIGHTING_SPECULAR_CDF_RESPONSE_GLSL

// Two-component response reconstruction for proposal-weighted incident moments.
#include "/lib/lighting/specular_cdf/coefficients.glsl"
// Backend supplies vecN, QLI_CDF_ARRAY, QLI_CDF_ATAN2, QLI_CDF_SAMPLE, mix,
// generated weights, and GGX_QLI_CDF_WIDTH.
// Existing physical delta events remain in the application wrapper.
float ggxQLiCdfSoft(float x) { return x/(1.0+abs(x)); }
vec2 ggxQLiCdfHemi(float q,float s2) {
    float d=sqrt(s2+q*q),t=q/d;
    float one=q>=0.0 ? 1.0+t : s2/max(d*(d-q),1e-30);
    return vec2(one*one*(2.0-t)*0.25,d);
}
float ggxQLiCdfEdge(float a,float d,float beta) {
    if(a<0.0 && d<(-0.25*a)) {
        float u=d/(-a),u2=u*u;
        float poly=1.0/3.0+u2*(-2.0/5.0+u2*(3.0/7.0+u2*(-4.0/9.0+u2*(5.0/11.0-u2*6.0/13.0))));
        return 4.0*poly/(-a*a*a);
    }
    return max(0.0,2.0*(beta/(d*d*d)+a/(d*d*(d*d+a*a))));
}

void ggxQLiCdfParameters(float k,float alpha,vec3 axis,vec3 view,vec3 ns,vec3 ng,
    out vec2 shape0,out vec2 shape1,out float weight0,out float mass) {
    k=clamp(k,0.0,1.0-1.1920928955078125e-7);
    float a=clamp(alpha,1e-4,1.0),v=max(dot(ns,view),1e-6);
    float s2=(1.0-k)*(1.0+k),s=sqrt(s2);
    vec3 kv=k*axis;
    float q1=dot(kv,ns),q2=dot(kv,ng),qv=dot(kv,view);
    float g=clamp(dot(ns,ng),-1.0,1.0),sn=length(cross(ns,ng));
    vec2 hn=ggxQLiCdfHemi(q1,s2),hg=ggxQLiCdfHemi(q2,s2);
    float hmin=min(hn.x,hg.x),dn=hn.y,dg=hg.y;
    float h;vec3 mean;
    if(sn<1e-7) {
        h=g>0.0 ? hn.x : 0.0;
        mean=kv+(dn/(2.0-q1/dn)-q1)*ns;
    } else {
        float a1=(q2-g*q1)/sn,a2=(q1-g*q2)/sn;
        float b1=QLI_CDF_ATAN2(dn,-a1),b2=QLI_CDF_ATAN2(dg,-a2);
        float j1=ggxQLiCdfEdge(a1,dn,b1),j2=ggxQLiCdfEdge(a2,dg,b2);
        float area=2.0*(3.141592653589793-QLI_CDF_ATAN2(sn,g)+q1/dn*b1+q2/dg*b2);
        float raw=area/(4.0*3.141592653589793)+s2*(q1*j1+q2*j2)/(8.0*3.141592653589793);
        h=clamp(raw,0.0,hmin);
        mean=(kv*h+s2*s2/(8.0*3.141592653589793)*(ns*j1+ng*j2))/max(h,1e-12);
        bool useN=hn.x<=hg.x;
        float q=useN?q1:q2,d=useN?dn:dg;
        vec3 normal=useN?ns:ng;
        vec3 fallback=kv+(d/(2.0-q/d)-q)*normal;
        float blend=hmin/(hmin+1e-4);
        mean=mix(fallback,mean,blend);
        h=mix(0.5*hmin,h,blend);
    }
    mean/=max(length(mean),1.0);
    float mn=clamp(dot(mean,ns),1e-8,1.0),mv=clamp(dot(mean,view),-1.0,1.0);
    float mg=clamp(dot(mean,ng),-1.0,1.0),vg=dot(view,ng);
    float spread=s*sqrt(max(0.0,1.0-mv*mv))+(1.0-k);
    float gate=min(1.0,(1.0-k)+s/(s+abs(q1))+s/(s+abs(q2)));
    float f[21]=float[21](k,s,q1*0.5+0.5,q2*0.5+0.5,qv*0.5+0.5,v,g*0.5+0.5,vg*0.5+0.5,a,
        q1/dn*0.5+0.5,q2/dg*0.5+0.5,h,mn,mv*0.5+0.5,mg*0.5+0.5,
        a/(a+s+abs(q1)),v/(v+a),v/(v+s+abs(q1)),gate,log2(s)/24.0+1.0,log2(a)/16.0+1.0);
    float l0[GGX_QLI_CDF_WIDTH],l1[GGX_QLI_CDF_WIDTH],raw[6];
    f[0]=2.0*f[0]-1.0;
    f[1]=2.0*f[1]-1.0;
    f[2]=2.0*f[2]-1.0;
    f[3]=2.0*f[3]-1.0;
    f[4]=2.0*f[4]-1.0;
    f[5]=2.0*f[5]-1.0;
    f[6]=2.0*f[6]-1.0;
    f[7]=2.0*f[7]-1.0;
    f[8]=2.0*f[8]-1.0;
    f[9]=2.0*f[9]-1.0;
    f[10]=2.0*f[10]-1.0;
    f[11]=2.0*f[11]-1.0;
    f[12]=2.0*f[12]-1.0;
    f[13]=2.0*f[13]-1.0;
    f[14]=2.0*f[14]-1.0;
    f[15]=2.0*f[15]-1.0;
    f[16]=2.0*f[16]-1.0;
    f[17]=2.0*f[17]-1.0;
    f[18]=2.0*f[18]-1.0;
    f[19]=2.0*f[19]-1.0;
    f[20]=2.0*f[20]-1.0;
    l0[0]=ggxQLiCdfSoft(ggxQLiCdfB0[0]+ggxQLiCdfW0[0]*f[0]+ggxQLiCdfW0[1]*f[1]+ggxQLiCdfW0[2]*f[2]+ggxQLiCdfW0[3]*f[3]+ggxQLiCdfW0[4]*f[4]+ggxQLiCdfW0[5]*f[5]+ggxQLiCdfW0[6]*f[6]+ggxQLiCdfW0[7]*f[7]+ggxQLiCdfW0[8]*f[8]+ggxQLiCdfW0[9]*f[9]+ggxQLiCdfW0[10]*f[10]+ggxQLiCdfW0[11]*f[11]+ggxQLiCdfW0[12]*f[12]+ggxQLiCdfW0[13]*f[13]+ggxQLiCdfW0[14]*f[14]+ggxQLiCdfW0[15]*f[15]+ggxQLiCdfW0[16]*f[16]+ggxQLiCdfW0[17]*f[17]+ggxQLiCdfW0[18]*f[18]+ggxQLiCdfW0[19]*f[19]+ggxQLiCdfW0[20]*f[20]);
    l0[1]=ggxQLiCdfSoft(ggxQLiCdfB0[1]+ggxQLiCdfW0[21]*f[0]+ggxQLiCdfW0[22]*f[1]+ggxQLiCdfW0[23]*f[2]+ggxQLiCdfW0[24]*f[3]+ggxQLiCdfW0[25]*f[4]+ggxQLiCdfW0[26]*f[5]+ggxQLiCdfW0[27]*f[6]+ggxQLiCdfW0[28]*f[7]+ggxQLiCdfW0[29]*f[8]+ggxQLiCdfW0[30]*f[9]+ggxQLiCdfW0[31]*f[10]+ggxQLiCdfW0[32]*f[11]+ggxQLiCdfW0[33]*f[12]+ggxQLiCdfW0[34]*f[13]+ggxQLiCdfW0[35]*f[14]+ggxQLiCdfW0[36]*f[15]+ggxQLiCdfW0[37]*f[16]+ggxQLiCdfW0[38]*f[17]+ggxQLiCdfW0[39]*f[18]+ggxQLiCdfW0[40]*f[19]+ggxQLiCdfW0[41]*f[20]);
    l0[2]=ggxQLiCdfSoft(ggxQLiCdfB0[2]+ggxQLiCdfW0[42]*f[0]+ggxQLiCdfW0[43]*f[1]+ggxQLiCdfW0[44]*f[2]+ggxQLiCdfW0[45]*f[3]+ggxQLiCdfW0[46]*f[4]+ggxQLiCdfW0[47]*f[5]+ggxQLiCdfW0[48]*f[6]+ggxQLiCdfW0[49]*f[7]+ggxQLiCdfW0[50]*f[8]+ggxQLiCdfW0[51]*f[9]+ggxQLiCdfW0[52]*f[10]+ggxQLiCdfW0[53]*f[11]+ggxQLiCdfW0[54]*f[12]+ggxQLiCdfW0[55]*f[13]+ggxQLiCdfW0[56]*f[14]+ggxQLiCdfW0[57]*f[15]+ggxQLiCdfW0[58]*f[16]+ggxQLiCdfW0[59]*f[17]+ggxQLiCdfW0[60]*f[18]+ggxQLiCdfW0[61]*f[19]+ggxQLiCdfW0[62]*f[20]);
    l0[3]=ggxQLiCdfSoft(ggxQLiCdfB0[3]+ggxQLiCdfW0[63]*f[0]+ggxQLiCdfW0[64]*f[1]+ggxQLiCdfW0[65]*f[2]+ggxQLiCdfW0[66]*f[3]+ggxQLiCdfW0[67]*f[4]+ggxQLiCdfW0[68]*f[5]+ggxQLiCdfW0[69]*f[6]+ggxQLiCdfW0[70]*f[7]+ggxQLiCdfW0[71]*f[8]+ggxQLiCdfW0[72]*f[9]+ggxQLiCdfW0[73]*f[10]+ggxQLiCdfW0[74]*f[11]+ggxQLiCdfW0[75]*f[12]+ggxQLiCdfW0[76]*f[13]+ggxQLiCdfW0[77]*f[14]+ggxQLiCdfW0[78]*f[15]+ggxQLiCdfW0[79]*f[16]+ggxQLiCdfW0[80]*f[17]+ggxQLiCdfW0[81]*f[18]+ggxQLiCdfW0[82]*f[19]+ggxQLiCdfW0[83]*f[20]);
    l0[4]=ggxQLiCdfSoft(ggxQLiCdfB0[4]+ggxQLiCdfW0[84]*f[0]+ggxQLiCdfW0[85]*f[1]+ggxQLiCdfW0[86]*f[2]+ggxQLiCdfW0[87]*f[3]+ggxQLiCdfW0[88]*f[4]+ggxQLiCdfW0[89]*f[5]+ggxQLiCdfW0[90]*f[6]+ggxQLiCdfW0[91]*f[7]+ggxQLiCdfW0[92]*f[8]+ggxQLiCdfW0[93]*f[9]+ggxQLiCdfW0[94]*f[10]+ggxQLiCdfW0[95]*f[11]+ggxQLiCdfW0[96]*f[12]+ggxQLiCdfW0[97]*f[13]+ggxQLiCdfW0[98]*f[14]+ggxQLiCdfW0[99]*f[15]+ggxQLiCdfW0[100]*f[16]+ggxQLiCdfW0[101]*f[17]+ggxQLiCdfW0[102]*f[18]+ggxQLiCdfW0[103]*f[19]+ggxQLiCdfW0[104]*f[20]);
    l0[5]=ggxQLiCdfSoft(ggxQLiCdfB0[5]+ggxQLiCdfW0[105]*f[0]+ggxQLiCdfW0[106]*f[1]+ggxQLiCdfW0[107]*f[2]+ggxQLiCdfW0[108]*f[3]+ggxQLiCdfW0[109]*f[4]+ggxQLiCdfW0[110]*f[5]+ggxQLiCdfW0[111]*f[6]+ggxQLiCdfW0[112]*f[7]+ggxQLiCdfW0[113]*f[8]+ggxQLiCdfW0[114]*f[9]+ggxQLiCdfW0[115]*f[10]+ggxQLiCdfW0[116]*f[11]+ggxQLiCdfW0[117]*f[12]+ggxQLiCdfW0[118]*f[13]+ggxQLiCdfW0[119]*f[14]+ggxQLiCdfW0[120]*f[15]+ggxQLiCdfW0[121]*f[16]+ggxQLiCdfW0[122]*f[17]+ggxQLiCdfW0[123]*f[18]+ggxQLiCdfW0[124]*f[19]+ggxQLiCdfW0[125]*f[20]);
    l0[6]=ggxQLiCdfSoft(ggxQLiCdfB0[6]+ggxQLiCdfW0[126]*f[0]+ggxQLiCdfW0[127]*f[1]+ggxQLiCdfW0[128]*f[2]+ggxQLiCdfW0[129]*f[3]+ggxQLiCdfW0[130]*f[4]+ggxQLiCdfW0[131]*f[5]+ggxQLiCdfW0[132]*f[6]+ggxQLiCdfW0[133]*f[7]+ggxQLiCdfW0[134]*f[8]+ggxQLiCdfW0[135]*f[9]+ggxQLiCdfW0[136]*f[10]+ggxQLiCdfW0[137]*f[11]+ggxQLiCdfW0[138]*f[12]+ggxQLiCdfW0[139]*f[13]+ggxQLiCdfW0[140]*f[14]+ggxQLiCdfW0[141]*f[15]+ggxQLiCdfW0[142]*f[16]+ggxQLiCdfW0[143]*f[17]+ggxQLiCdfW0[144]*f[18]+ggxQLiCdfW0[145]*f[19]+ggxQLiCdfW0[146]*f[20]);
    l0[7]=ggxQLiCdfSoft(ggxQLiCdfB0[7]+ggxQLiCdfW0[147]*f[0]+ggxQLiCdfW0[148]*f[1]+ggxQLiCdfW0[149]*f[2]+ggxQLiCdfW0[150]*f[3]+ggxQLiCdfW0[151]*f[4]+ggxQLiCdfW0[152]*f[5]+ggxQLiCdfW0[153]*f[6]+ggxQLiCdfW0[154]*f[7]+ggxQLiCdfW0[155]*f[8]+ggxQLiCdfW0[156]*f[9]+ggxQLiCdfW0[157]*f[10]+ggxQLiCdfW0[158]*f[11]+ggxQLiCdfW0[159]*f[12]+ggxQLiCdfW0[160]*f[13]+ggxQLiCdfW0[161]*f[14]+ggxQLiCdfW0[162]*f[15]+ggxQLiCdfW0[163]*f[16]+ggxQLiCdfW0[164]*f[17]+ggxQLiCdfW0[165]*f[18]+ggxQLiCdfW0[166]*f[19]+ggxQLiCdfW0[167]*f[20]);
    l0[8]=ggxQLiCdfSoft(ggxQLiCdfB0[8]+ggxQLiCdfW0[168]*f[0]+ggxQLiCdfW0[169]*f[1]+ggxQLiCdfW0[170]*f[2]+ggxQLiCdfW0[171]*f[3]+ggxQLiCdfW0[172]*f[4]+ggxQLiCdfW0[173]*f[5]+ggxQLiCdfW0[174]*f[6]+ggxQLiCdfW0[175]*f[7]+ggxQLiCdfW0[176]*f[8]+ggxQLiCdfW0[177]*f[9]+ggxQLiCdfW0[178]*f[10]+ggxQLiCdfW0[179]*f[11]+ggxQLiCdfW0[180]*f[12]+ggxQLiCdfW0[181]*f[13]+ggxQLiCdfW0[182]*f[14]+ggxQLiCdfW0[183]*f[15]+ggxQLiCdfW0[184]*f[16]+ggxQLiCdfW0[185]*f[17]+ggxQLiCdfW0[186]*f[18]+ggxQLiCdfW0[187]*f[19]+ggxQLiCdfW0[188]*f[20]);
    l0[9]=ggxQLiCdfSoft(ggxQLiCdfB0[9]+ggxQLiCdfW0[189]*f[0]+ggxQLiCdfW0[190]*f[1]+ggxQLiCdfW0[191]*f[2]+ggxQLiCdfW0[192]*f[3]+ggxQLiCdfW0[193]*f[4]+ggxQLiCdfW0[194]*f[5]+ggxQLiCdfW0[195]*f[6]+ggxQLiCdfW0[196]*f[7]+ggxQLiCdfW0[197]*f[8]+ggxQLiCdfW0[198]*f[9]+ggxQLiCdfW0[199]*f[10]+ggxQLiCdfW0[200]*f[11]+ggxQLiCdfW0[201]*f[12]+ggxQLiCdfW0[202]*f[13]+ggxQLiCdfW0[203]*f[14]+ggxQLiCdfW0[204]*f[15]+ggxQLiCdfW0[205]*f[16]+ggxQLiCdfW0[206]*f[17]+ggxQLiCdfW0[207]*f[18]+ggxQLiCdfW0[208]*f[19]+ggxQLiCdfW0[209]*f[20]);
    l0[10]=ggxQLiCdfSoft(ggxQLiCdfB0[10]+ggxQLiCdfW0[210]*f[0]+ggxQLiCdfW0[211]*f[1]+ggxQLiCdfW0[212]*f[2]+ggxQLiCdfW0[213]*f[3]+ggxQLiCdfW0[214]*f[4]+ggxQLiCdfW0[215]*f[5]+ggxQLiCdfW0[216]*f[6]+ggxQLiCdfW0[217]*f[7]+ggxQLiCdfW0[218]*f[8]+ggxQLiCdfW0[219]*f[9]+ggxQLiCdfW0[220]*f[10]+ggxQLiCdfW0[221]*f[11]+ggxQLiCdfW0[222]*f[12]+ggxQLiCdfW0[223]*f[13]+ggxQLiCdfW0[224]*f[14]+ggxQLiCdfW0[225]*f[15]+ggxQLiCdfW0[226]*f[16]+ggxQLiCdfW0[227]*f[17]+ggxQLiCdfW0[228]*f[18]+ggxQLiCdfW0[229]*f[19]+ggxQLiCdfW0[230]*f[20]);
    l0[11]=ggxQLiCdfSoft(ggxQLiCdfB0[11]+ggxQLiCdfW0[231]*f[0]+ggxQLiCdfW0[232]*f[1]+ggxQLiCdfW0[233]*f[2]+ggxQLiCdfW0[234]*f[3]+ggxQLiCdfW0[235]*f[4]+ggxQLiCdfW0[236]*f[5]+ggxQLiCdfW0[237]*f[6]+ggxQLiCdfW0[238]*f[7]+ggxQLiCdfW0[239]*f[8]+ggxQLiCdfW0[240]*f[9]+ggxQLiCdfW0[241]*f[10]+ggxQLiCdfW0[242]*f[11]+ggxQLiCdfW0[243]*f[12]+ggxQLiCdfW0[244]*f[13]+ggxQLiCdfW0[245]*f[14]+ggxQLiCdfW0[246]*f[15]+ggxQLiCdfW0[247]*f[16]+ggxQLiCdfW0[248]*f[17]+ggxQLiCdfW0[249]*f[18]+ggxQLiCdfW0[250]*f[19]+ggxQLiCdfW0[251]*f[20]);
    l0[12]=ggxQLiCdfSoft(ggxQLiCdfB0[12]+ggxQLiCdfW0[252]*f[0]+ggxQLiCdfW0[253]*f[1]+ggxQLiCdfW0[254]*f[2]+ggxQLiCdfW0[255]*f[3]+ggxQLiCdfW0[256]*f[4]+ggxQLiCdfW0[257]*f[5]+ggxQLiCdfW0[258]*f[6]+ggxQLiCdfW0[259]*f[7]+ggxQLiCdfW0[260]*f[8]+ggxQLiCdfW0[261]*f[9]+ggxQLiCdfW0[262]*f[10]+ggxQLiCdfW0[263]*f[11]+ggxQLiCdfW0[264]*f[12]+ggxQLiCdfW0[265]*f[13]+ggxQLiCdfW0[266]*f[14]+ggxQLiCdfW0[267]*f[15]+ggxQLiCdfW0[268]*f[16]+ggxQLiCdfW0[269]*f[17]+ggxQLiCdfW0[270]*f[18]+ggxQLiCdfW0[271]*f[19]+ggxQLiCdfW0[272]*f[20]);
    l0[13]=ggxQLiCdfSoft(ggxQLiCdfB0[13]+ggxQLiCdfW0[273]*f[0]+ggxQLiCdfW0[274]*f[1]+ggxQLiCdfW0[275]*f[2]+ggxQLiCdfW0[276]*f[3]+ggxQLiCdfW0[277]*f[4]+ggxQLiCdfW0[278]*f[5]+ggxQLiCdfW0[279]*f[6]+ggxQLiCdfW0[280]*f[7]+ggxQLiCdfW0[281]*f[8]+ggxQLiCdfW0[282]*f[9]+ggxQLiCdfW0[283]*f[10]+ggxQLiCdfW0[284]*f[11]+ggxQLiCdfW0[285]*f[12]+ggxQLiCdfW0[286]*f[13]+ggxQLiCdfW0[287]*f[14]+ggxQLiCdfW0[288]*f[15]+ggxQLiCdfW0[289]*f[16]+ggxQLiCdfW0[290]*f[17]+ggxQLiCdfW0[291]*f[18]+ggxQLiCdfW0[292]*f[19]+ggxQLiCdfW0[293]*f[20]);
    l0[14]=ggxQLiCdfSoft(ggxQLiCdfB0[14]+ggxQLiCdfW0[294]*f[0]+ggxQLiCdfW0[295]*f[1]+ggxQLiCdfW0[296]*f[2]+ggxQLiCdfW0[297]*f[3]+ggxQLiCdfW0[298]*f[4]+ggxQLiCdfW0[299]*f[5]+ggxQLiCdfW0[300]*f[6]+ggxQLiCdfW0[301]*f[7]+ggxQLiCdfW0[302]*f[8]+ggxQLiCdfW0[303]*f[9]+ggxQLiCdfW0[304]*f[10]+ggxQLiCdfW0[305]*f[11]+ggxQLiCdfW0[306]*f[12]+ggxQLiCdfW0[307]*f[13]+ggxQLiCdfW0[308]*f[14]+ggxQLiCdfW0[309]*f[15]+ggxQLiCdfW0[310]*f[16]+ggxQLiCdfW0[311]*f[17]+ggxQLiCdfW0[312]*f[18]+ggxQLiCdfW0[313]*f[19]+ggxQLiCdfW0[314]*f[20]);
    l0[15]=ggxQLiCdfSoft(ggxQLiCdfB0[15]+ggxQLiCdfW0[315]*f[0]+ggxQLiCdfW0[316]*f[1]+ggxQLiCdfW0[317]*f[2]+ggxQLiCdfW0[318]*f[3]+ggxQLiCdfW0[319]*f[4]+ggxQLiCdfW0[320]*f[5]+ggxQLiCdfW0[321]*f[6]+ggxQLiCdfW0[322]*f[7]+ggxQLiCdfW0[323]*f[8]+ggxQLiCdfW0[324]*f[9]+ggxQLiCdfW0[325]*f[10]+ggxQLiCdfW0[326]*f[11]+ggxQLiCdfW0[327]*f[12]+ggxQLiCdfW0[328]*f[13]+ggxQLiCdfW0[329]*f[14]+ggxQLiCdfW0[330]*f[15]+ggxQLiCdfW0[331]*f[16]+ggxQLiCdfW0[332]*f[17]+ggxQLiCdfW0[333]*f[18]+ggxQLiCdfW0[334]*f[19]+ggxQLiCdfW0[335]*f[20]);
    l1[0]=ggxQLiCdfSoft(ggxQLiCdfB1[0]+ggxQLiCdfW1[0]*l0[0]+ggxQLiCdfW1[1]*l0[1]+ggxQLiCdfW1[2]*l0[2]+ggxQLiCdfW1[3]*l0[3]+ggxQLiCdfW1[4]*l0[4]+ggxQLiCdfW1[5]*l0[5]+ggxQLiCdfW1[6]*l0[6]+ggxQLiCdfW1[7]*l0[7]+ggxQLiCdfW1[8]*l0[8]+ggxQLiCdfW1[9]*l0[9]+ggxQLiCdfW1[10]*l0[10]+ggxQLiCdfW1[11]*l0[11]+ggxQLiCdfW1[12]*l0[12]+ggxQLiCdfW1[13]*l0[13]+ggxQLiCdfW1[14]*l0[14]+ggxQLiCdfW1[15]*l0[15]);
    l1[1]=ggxQLiCdfSoft(ggxQLiCdfB1[1]+ggxQLiCdfW1[16]*l0[0]+ggxQLiCdfW1[17]*l0[1]+ggxQLiCdfW1[18]*l0[2]+ggxQLiCdfW1[19]*l0[3]+ggxQLiCdfW1[20]*l0[4]+ggxQLiCdfW1[21]*l0[5]+ggxQLiCdfW1[22]*l0[6]+ggxQLiCdfW1[23]*l0[7]+ggxQLiCdfW1[24]*l0[8]+ggxQLiCdfW1[25]*l0[9]+ggxQLiCdfW1[26]*l0[10]+ggxQLiCdfW1[27]*l0[11]+ggxQLiCdfW1[28]*l0[12]+ggxQLiCdfW1[29]*l0[13]+ggxQLiCdfW1[30]*l0[14]+ggxQLiCdfW1[31]*l0[15]);
    l1[2]=ggxQLiCdfSoft(ggxQLiCdfB1[2]+ggxQLiCdfW1[32]*l0[0]+ggxQLiCdfW1[33]*l0[1]+ggxQLiCdfW1[34]*l0[2]+ggxQLiCdfW1[35]*l0[3]+ggxQLiCdfW1[36]*l0[4]+ggxQLiCdfW1[37]*l0[5]+ggxQLiCdfW1[38]*l0[6]+ggxQLiCdfW1[39]*l0[7]+ggxQLiCdfW1[40]*l0[8]+ggxQLiCdfW1[41]*l0[9]+ggxQLiCdfW1[42]*l0[10]+ggxQLiCdfW1[43]*l0[11]+ggxQLiCdfW1[44]*l0[12]+ggxQLiCdfW1[45]*l0[13]+ggxQLiCdfW1[46]*l0[14]+ggxQLiCdfW1[47]*l0[15]);
    l1[3]=ggxQLiCdfSoft(ggxQLiCdfB1[3]+ggxQLiCdfW1[48]*l0[0]+ggxQLiCdfW1[49]*l0[1]+ggxQLiCdfW1[50]*l0[2]+ggxQLiCdfW1[51]*l0[3]+ggxQLiCdfW1[52]*l0[4]+ggxQLiCdfW1[53]*l0[5]+ggxQLiCdfW1[54]*l0[6]+ggxQLiCdfW1[55]*l0[7]+ggxQLiCdfW1[56]*l0[8]+ggxQLiCdfW1[57]*l0[9]+ggxQLiCdfW1[58]*l0[10]+ggxQLiCdfW1[59]*l0[11]+ggxQLiCdfW1[60]*l0[12]+ggxQLiCdfW1[61]*l0[13]+ggxQLiCdfW1[62]*l0[14]+ggxQLiCdfW1[63]*l0[15]);
    l1[4]=ggxQLiCdfSoft(ggxQLiCdfB1[4]+ggxQLiCdfW1[64]*l0[0]+ggxQLiCdfW1[65]*l0[1]+ggxQLiCdfW1[66]*l0[2]+ggxQLiCdfW1[67]*l0[3]+ggxQLiCdfW1[68]*l0[4]+ggxQLiCdfW1[69]*l0[5]+ggxQLiCdfW1[70]*l0[6]+ggxQLiCdfW1[71]*l0[7]+ggxQLiCdfW1[72]*l0[8]+ggxQLiCdfW1[73]*l0[9]+ggxQLiCdfW1[74]*l0[10]+ggxQLiCdfW1[75]*l0[11]+ggxQLiCdfW1[76]*l0[12]+ggxQLiCdfW1[77]*l0[13]+ggxQLiCdfW1[78]*l0[14]+ggxQLiCdfW1[79]*l0[15]);
    l1[5]=ggxQLiCdfSoft(ggxQLiCdfB1[5]+ggxQLiCdfW1[80]*l0[0]+ggxQLiCdfW1[81]*l0[1]+ggxQLiCdfW1[82]*l0[2]+ggxQLiCdfW1[83]*l0[3]+ggxQLiCdfW1[84]*l0[4]+ggxQLiCdfW1[85]*l0[5]+ggxQLiCdfW1[86]*l0[6]+ggxQLiCdfW1[87]*l0[7]+ggxQLiCdfW1[88]*l0[8]+ggxQLiCdfW1[89]*l0[9]+ggxQLiCdfW1[90]*l0[10]+ggxQLiCdfW1[91]*l0[11]+ggxQLiCdfW1[92]*l0[12]+ggxQLiCdfW1[93]*l0[13]+ggxQLiCdfW1[94]*l0[14]+ggxQLiCdfW1[95]*l0[15]);
    l1[6]=ggxQLiCdfSoft(ggxQLiCdfB1[6]+ggxQLiCdfW1[96]*l0[0]+ggxQLiCdfW1[97]*l0[1]+ggxQLiCdfW1[98]*l0[2]+ggxQLiCdfW1[99]*l0[3]+ggxQLiCdfW1[100]*l0[4]+ggxQLiCdfW1[101]*l0[5]+ggxQLiCdfW1[102]*l0[6]+ggxQLiCdfW1[103]*l0[7]+ggxQLiCdfW1[104]*l0[8]+ggxQLiCdfW1[105]*l0[9]+ggxQLiCdfW1[106]*l0[10]+ggxQLiCdfW1[107]*l0[11]+ggxQLiCdfW1[108]*l0[12]+ggxQLiCdfW1[109]*l0[13]+ggxQLiCdfW1[110]*l0[14]+ggxQLiCdfW1[111]*l0[15]);
    l1[7]=ggxQLiCdfSoft(ggxQLiCdfB1[7]+ggxQLiCdfW1[112]*l0[0]+ggxQLiCdfW1[113]*l0[1]+ggxQLiCdfW1[114]*l0[2]+ggxQLiCdfW1[115]*l0[3]+ggxQLiCdfW1[116]*l0[4]+ggxQLiCdfW1[117]*l0[5]+ggxQLiCdfW1[118]*l0[6]+ggxQLiCdfW1[119]*l0[7]+ggxQLiCdfW1[120]*l0[8]+ggxQLiCdfW1[121]*l0[9]+ggxQLiCdfW1[122]*l0[10]+ggxQLiCdfW1[123]*l0[11]+ggxQLiCdfW1[124]*l0[12]+ggxQLiCdfW1[125]*l0[13]+ggxQLiCdfW1[126]*l0[14]+ggxQLiCdfW1[127]*l0[15]);
    l1[8]=ggxQLiCdfSoft(ggxQLiCdfB1[8]+ggxQLiCdfW1[128]*l0[0]+ggxQLiCdfW1[129]*l0[1]+ggxQLiCdfW1[130]*l0[2]+ggxQLiCdfW1[131]*l0[3]+ggxQLiCdfW1[132]*l0[4]+ggxQLiCdfW1[133]*l0[5]+ggxQLiCdfW1[134]*l0[6]+ggxQLiCdfW1[135]*l0[7]+ggxQLiCdfW1[136]*l0[8]+ggxQLiCdfW1[137]*l0[9]+ggxQLiCdfW1[138]*l0[10]+ggxQLiCdfW1[139]*l0[11]+ggxQLiCdfW1[140]*l0[12]+ggxQLiCdfW1[141]*l0[13]+ggxQLiCdfW1[142]*l0[14]+ggxQLiCdfW1[143]*l0[15]);
    l1[9]=ggxQLiCdfSoft(ggxQLiCdfB1[9]+ggxQLiCdfW1[144]*l0[0]+ggxQLiCdfW1[145]*l0[1]+ggxQLiCdfW1[146]*l0[2]+ggxQLiCdfW1[147]*l0[3]+ggxQLiCdfW1[148]*l0[4]+ggxQLiCdfW1[149]*l0[5]+ggxQLiCdfW1[150]*l0[6]+ggxQLiCdfW1[151]*l0[7]+ggxQLiCdfW1[152]*l0[8]+ggxQLiCdfW1[153]*l0[9]+ggxQLiCdfW1[154]*l0[10]+ggxQLiCdfW1[155]*l0[11]+ggxQLiCdfW1[156]*l0[12]+ggxQLiCdfW1[157]*l0[13]+ggxQLiCdfW1[158]*l0[14]+ggxQLiCdfW1[159]*l0[15]);
    l1[10]=ggxQLiCdfSoft(ggxQLiCdfB1[10]+ggxQLiCdfW1[160]*l0[0]+ggxQLiCdfW1[161]*l0[1]+ggxQLiCdfW1[162]*l0[2]+ggxQLiCdfW1[163]*l0[3]+ggxQLiCdfW1[164]*l0[4]+ggxQLiCdfW1[165]*l0[5]+ggxQLiCdfW1[166]*l0[6]+ggxQLiCdfW1[167]*l0[7]+ggxQLiCdfW1[168]*l0[8]+ggxQLiCdfW1[169]*l0[9]+ggxQLiCdfW1[170]*l0[10]+ggxQLiCdfW1[171]*l0[11]+ggxQLiCdfW1[172]*l0[12]+ggxQLiCdfW1[173]*l0[13]+ggxQLiCdfW1[174]*l0[14]+ggxQLiCdfW1[175]*l0[15]);
    l1[11]=ggxQLiCdfSoft(ggxQLiCdfB1[11]+ggxQLiCdfW1[176]*l0[0]+ggxQLiCdfW1[177]*l0[1]+ggxQLiCdfW1[178]*l0[2]+ggxQLiCdfW1[179]*l0[3]+ggxQLiCdfW1[180]*l0[4]+ggxQLiCdfW1[181]*l0[5]+ggxQLiCdfW1[182]*l0[6]+ggxQLiCdfW1[183]*l0[7]+ggxQLiCdfW1[184]*l0[8]+ggxQLiCdfW1[185]*l0[9]+ggxQLiCdfW1[186]*l0[10]+ggxQLiCdfW1[187]*l0[11]+ggxQLiCdfW1[188]*l0[12]+ggxQLiCdfW1[189]*l0[13]+ggxQLiCdfW1[190]*l0[14]+ggxQLiCdfW1[191]*l0[15]);
    l1[12]=ggxQLiCdfSoft(ggxQLiCdfB1[12]+ggxQLiCdfW1[192]*l0[0]+ggxQLiCdfW1[193]*l0[1]+ggxQLiCdfW1[194]*l0[2]+ggxQLiCdfW1[195]*l0[3]+ggxQLiCdfW1[196]*l0[4]+ggxQLiCdfW1[197]*l0[5]+ggxQLiCdfW1[198]*l0[6]+ggxQLiCdfW1[199]*l0[7]+ggxQLiCdfW1[200]*l0[8]+ggxQLiCdfW1[201]*l0[9]+ggxQLiCdfW1[202]*l0[10]+ggxQLiCdfW1[203]*l0[11]+ggxQLiCdfW1[204]*l0[12]+ggxQLiCdfW1[205]*l0[13]+ggxQLiCdfW1[206]*l0[14]+ggxQLiCdfW1[207]*l0[15]);
    l1[13]=ggxQLiCdfSoft(ggxQLiCdfB1[13]+ggxQLiCdfW1[208]*l0[0]+ggxQLiCdfW1[209]*l0[1]+ggxQLiCdfW1[210]*l0[2]+ggxQLiCdfW1[211]*l0[3]+ggxQLiCdfW1[212]*l0[4]+ggxQLiCdfW1[213]*l0[5]+ggxQLiCdfW1[214]*l0[6]+ggxQLiCdfW1[215]*l0[7]+ggxQLiCdfW1[216]*l0[8]+ggxQLiCdfW1[217]*l0[9]+ggxQLiCdfW1[218]*l0[10]+ggxQLiCdfW1[219]*l0[11]+ggxQLiCdfW1[220]*l0[12]+ggxQLiCdfW1[221]*l0[13]+ggxQLiCdfW1[222]*l0[14]+ggxQLiCdfW1[223]*l0[15]);
    l1[14]=ggxQLiCdfSoft(ggxQLiCdfB1[14]+ggxQLiCdfW1[224]*l0[0]+ggxQLiCdfW1[225]*l0[1]+ggxQLiCdfW1[226]*l0[2]+ggxQLiCdfW1[227]*l0[3]+ggxQLiCdfW1[228]*l0[4]+ggxQLiCdfW1[229]*l0[5]+ggxQLiCdfW1[230]*l0[6]+ggxQLiCdfW1[231]*l0[7]+ggxQLiCdfW1[232]*l0[8]+ggxQLiCdfW1[233]*l0[9]+ggxQLiCdfW1[234]*l0[10]+ggxQLiCdfW1[235]*l0[11]+ggxQLiCdfW1[236]*l0[12]+ggxQLiCdfW1[237]*l0[13]+ggxQLiCdfW1[238]*l0[14]+ggxQLiCdfW1[239]*l0[15]);
    l1[15]=ggxQLiCdfSoft(ggxQLiCdfB1[15]+ggxQLiCdfW1[240]*l0[0]+ggxQLiCdfW1[241]*l0[1]+ggxQLiCdfW1[242]*l0[2]+ggxQLiCdfW1[243]*l0[3]+ggxQLiCdfW1[244]*l0[4]+ggxQLiCdfW1[245]*l0[5]+ggxQLiCdfW1[246]*l0[6]+ggxQLiCdfW1[247]*l0[7]+ggxQLiCdfW1[248]*l0[8]+ggxQLiCdfW1[249]*l0[9]+ggxQLiCdfW1[250]*l0[10]+ggxQLiCdfW1[251]*l0[11]+ggxQLiCdfW1[252]*l0[12]+ggxQLiCdfW1[253]*l0[13]+ggxQLiCdfW1[254]*l0[14]+ggxQLiCdfW1[255]*l0[15]);
    raw[0]=ggxQLiCdfSoft(ggxQLiCdfB2[0]+ggxQLiCdfW2[0]*l1[0]+ggxQLiCdfW2[1]*l1[1]+ggxQLiCdfW2[2]*l1[2]+ggxQLiCdfW2[3]*l1[3]+ggxQLiCdfW2[4]*l1[4]+ggxQLiCdfW2[5]*l1[5]+ggxQLiCdfW2[6]*l1[6]+ggxQLiCdfW2[7]*l1[7]+ggxQLiCdfW2[8]*l1[8]+ggxQLiCdfW2[9]*l1[9]+ggxQLiCdfW2[10]*l1[10]+ggxQLiCdfW2[11]*l1[11]+ggxQLiCdfW2[12]*l1[12]+ggxQLiCdfW2[13]*l1[13]+ggxQLiCdfW2[14]*l1[14]+ggxQLiCdfW2[15]*l1[15]);
    raw[1]=ggxQLiCdfSoft(ggxQLiCdfB2[1]+ggxQLiCdfW2[16]*l1[0]+ggxQLiCdfW2[17]*l1[1]+ggxQLiCdfW2[18]*l1[2]+ggxQLiCdfW2[19]*l1[3]+ggxQLiCdfW2[20]*l1[4]+ggxQLiCdfW2[21]*l1[5]+ggxQLiCdfW2[22]*l1[6]+ggxQLiCdfW2[23]*l1[7]+ggxQLiCdfW2[24]*l1[8]+ggxQLiCdfW2[25]*l1[9]+ggxQLiCdfW2[26]*l1[10]+ggxQLiCdfW2[27]*l1[11]+ggxQLiCdfW2[28]*l1[12]+ggxQLiCdfW2[29]*l1[13]+ggxQLiCdfW2[30]*l1[14]+ggxQLiCdfW2[31]*l1[15]);
    raw[2]=ggxQLiCdfSoft(ggxQLiCdfB2[2]+ggxQLiCdfW2[32]*l1[0]+ggxQLiCdfW2[33]*l1[1]+ggxQLiCdfW2[34]*l1[2]+ggxQLiCdfW2[35]*l1[3]+ggxQLiCdfW2[36]*l1[4]+ggxQLiCdfW2[37]*l1[5]+ggxQLiCdfW2[38]*l1[6]+ggxQLiCdfW2[39]*l1[7]+ggxQLiCdfW2[40]*l1[8]+ggxQLiCdfW2[41]*l1[9]+ggxQLiCdfW2[42]*l1[10]+ggxQLiCdfW2[43]*l1[11]+ggxQLiCdfW2[44]*l1[12]+ggxQLiCdfW2[45]*l1[13]+ggxQLiCdfW2[46]*l1[14]+ggxQLiCdfW2[47]*l1[15]);
    raw[3]=ggxQLiCdfSoft(ggxQLiCdfB2[3]+ggxQLiCdfW2[48]*l1[0]+ggxQLiCdfW2[49]*l1[1]+ggxQLiCdfW2[50]*l1[2]+ggxQLiCdfW2[51]*l1[3]+ggxQLiCdfW2[52]*l1[4]+ggxQLiCdfW2[53]*l1[5]+ggxQLiCdfW2[54]*l1[6]+ggxQLiCdfW2[55]*l1[7]+ggxQLiCdfW2[56]*l1[8]+ggxQLiCdfW2[57]*l1[9]+ggxQLiCdfW2[58]*l1[10]+ggxQLiCdfW2[59]*l1[11]+ggxQLiCdfW2[60]*l1[12]+ggxQLiCdfW2[61]*l1[13]+ggxQLiCdfW2[62]*l1[14]+ggxQLiCdfW2[63]*l1[15]);
    raw[4]=ggxQLiCdfSoft(ggxQLiCdfB2[4]+ggxQLiCdfW2[64]*l1[0]+ggxQLiCdfW2[65]*l1[1]+ggxQLiCdfW2[66]*l1[2]+ggxQLiCdfW2[67]*l1[3]+ggxQLiCdfW2[68]*l1[4]+ggxQLiCdfW2[69]*l1[5]+ggxQLiCdfW2[70]*l1[6]+ggxQLiCdfW2[71]*l1[7]+ggxQLiCdfW2[72]*l1[8]+ggxQLiCdfW2[73]*l1[9]+ggxQLiCdfW2[74]*l1[10]+ggxQLiCdfW2[75]*l1[11]+ggxQLiCdfW2[76]*l1[12]+ggxQLiCdfW2[77]*l1[13]+ggxQLiCdfW2[78]*l1[14]+ggxQLiCdfW2[79]*l1[15]);
    raw[5]=ggxQLiCdfSoft(ggxQLiCdfB2[5]+ggxQLiCdfW2[80]*l1[0]+ggxQLiCdfW2[81]*l1[1]+ggxQLiCdfW2[82]*l1[2]+ggxQLiCdfW2[83]*l1[3]+ggxQLiCdfW2[84]*l1[4]+ggxQLiCdfW2[85]*l1[5]+ggxQLiCdfW2[86]*l1[6]+ggxQLiCdfW2[87]*l1[7]+ggxQLiCdfW2[88]*l1[8]+ggxQLiCdfW2[89]*l1[9]+ggxQLiCdfW2[90]*l1[10]+ggxQLiCdfW2[91]*l1[11]+ggxQLiCdfW2[92]*l1[12]+ggxQLiCdfW2[93]*l1[13]+ggxQLiCdfW2[94]*l1[14]+ggxQLiCdfW2[95]*l1[15]);
    float base=0.5*(1.0+mv),lw=log2(spread)-2.0;
    shape0=vec2(clamp(base+spread*raw[1],1e-8,1.0-1e-7),clamp(lw+4.0*raw[2],-24.0,2.0));
    shape1=vec2(clamp(base+spread*raw[3],1e-8,1.0-1e-7),clamp(lw+4.0*raw[4],-24.0,2.0));
    weight0=0.5+0.5*raw[0];
    float mi=clamp(mn*exp2(4.0*gate*raw[5]),1e-8,1.0);
    float ri=sqrt(a*a+(1.0-a*a)*mi*mi),ro=sqrt(a*a+(1.0-a*a)*v*v);
    mass=h*mi*(ro+v)/max(v*ri+mi*ro,1e-25);
}

vec2 ggxQLiCdfLookup(vec2 shape,float eta) {
    bool inside=eta>1.0;
    float n=max(eta,1.0/max(eta,1e-8));
    n=clamp(n,1.0+1.1920928955078125e-7,1.7);
    float threshold=1.0-1.0/(n*n),m=shape.x;
    float u=log2(1.0+m*16777216.0)/24.0;
    if(inside)u=m<=threshold ? 0.5*(1.0-sqrt(max(0.0,1.0-m/threshold)))
        : 0.5*(1.0+sqrt(max(0.0,(m-threshold)/(1.0-threshold))));
    vec3 p=clamp(vec3(u,(shape.y+24.0)/26.0,(log2(n-1.0)+23.0)/22.48542682717024),0.0,1.0);
    vec4 value=QLI_CDF_SAMPLE((0.5+32.0*p)/33.0);
    vec2 result=inside?value.ga:value.rb;
    if(eta==1.0)result.x=0.0;
    return result;
}

// Returns (dielectric, Schlick-edge, unit-Fresnel mass).
vec3 ggxQLiCdfResponse(float k,float alpha,vec3 axis,vec3 view,vec3 ns,vec3 ng,float eta) {
    if(dot(ns,view)<=1e-6)return vec3(0.0,0.0,0.0);
    vec2 a,b;float weight0,mass;
    ggxQLiCdfParameters(k,alpha,axis,view,ns,ng,a,b,weight0,mass);
    vec2 value=mix(ggxQLiCdfLookup(b,eta),ggxQLiCdfLookup(a,eta),weight0);
    return vec3(mass*value,mass);
}

#endif // DIRT_RT_LIB_LIGHTING_SPECULAR_CDF_RESPONSE_GLSL
