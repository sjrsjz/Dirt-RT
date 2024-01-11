#ifndef LIGHT_COLOR_GLSL
#define LIGHT_COLOR_GLSL
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
float S_R=0.05;
float cosD_S=1/sqrt(1+S_R*S_R);
vec3 b_P=vec3(300000);//atmosphere thickness
float b_k=0.25;//mix

 
vec3 Mie=vec3(0.2);

vec3 Rayleigh=4e10*pow(vec3(1./700,1./520,1./450),vec3(4));
  
vec3 b_k0=mix(Rayleigh,Mie,b_k);

vec3 b_Q=b_k0/(b_P*b_P);//absorption
vec3 b_g0=mix(Rayleigh*0.01,vec3(0.9),b_k);//single scatter

void setSkyVars(){
    S_R=0.05;
    cosD_S=1/sqrt(1+S_R*S_R);

    switch(world_type_global){
        case World_THE_END:
            S_R=0.25;
            cosD_S=1/sqrt(1+S_R*S_R);
            Mie=vec3(0.01);
            Rayleigh=1e9*pow(vec3(1./700,1./520,1./450),vec3(4));
            b_P=vec3(4096);
            b_k=0.75;
            break;
        case World_THE_NETHER:
            S_R=0.05;
            cosD_S=1/sqrt(1+S_R*S_R);
            Mie=vec3(0.2);
            Rayleigh=4e11*pow(vec3(1./700,1./520,1./450),vec3(4));
            b_P=vec3(600000);
            b_k=0.5;
            break;
        default:
            S_R=0.05;
            cosD_S=1/sqrt(1+S_R*S_R);
            Mie=vec3(0.2);
            Rayleigh=4e10*pow(vec3(1./700,1./520,1./450),vec3(4));
            b_P=vec3(300000);
            b_k = 0.25 + rainStrength_global * 0.75;
            break;
    }

    b_k0=mix(Rayleigh,Mie,b_k);
    b_Q=b_k0/(b_P*b_P);//absorption
    b_g0=mix(Rayleigh*0.01,vec3(0.9),b_k);//single scatter
}
vec3 getSkyColor(vec3 b_Sun,vec3 b_Moon,in vec3 pos,in vec3 n,in vec3 lightDir ) {
    vec3 n0=n;
    n.y=max(n.y,1e-5);
    vec3 lightDir1=-lightDir;
    vec3 g=3./(8.*PI)*(1.+pow(dot(n,lightDir),2.))*(1.-b_g0*b_g0)/(2.+b_g0*b_g0)/pow(1.+b_g0*b_g0-2.*b_g0*dot(lightDir,n),vec3(1.5));
    vec3 t=b_Q*0.5*(b_P-pos.y)*(b_P-pos.y);
    vec3 c=b_Sun*g*(exp(-t/n.y)-exp(-t/lightDir.y))/(n.y-lightDir.y)*max(lightDir.y,0.);

    //g=3./(8.*PI)*(1.+pow(dot(n,lightDir1),2.))*(1.-b_g0*b_g0)/(2.+b_g0*b_g0)/pow(1.+b_g0*b_g0-2.*b_g0*dot(lightDir1,n),vec3(1.5));
    //t=b_Q*0.5*(b_P-pos.y)*(b_P-pos.y);
    //c+=b_Moon*g*(exp(-t/n.y)-exp(-t/lightDir1.y))/(n.y-lightDir1.y)*max(lightDir1.y,0.);

    c+=exp(-t/n.y)*b_Sun*exp(-sqrt(abs(min(dot(n0,lightDir)-cosD_S,0))*6000));
    c+=exp(-t/n.y)*b_Moon*exp(-sqrt(abs(min(dot(n0,lightDir1)-cosD_S,0))*15000));
    return abs(c);
}
vec3 getFogColor(vec3 b_Sun,vec3 b_Moon,in vec3 pos, in vec3 n,in vec3 lightDir,float s,vec3 col ) {
    vec3 n0=n;
    if(n.y>0) s=min((b_P.x-pos.y)/n.y,s);
    vec3 g=3./(8.*PI)*(1.+pow(dot(n,lightDir),2.))*(1.-b_g0*b_g0)/(2.+b_g0*b_g0)/pow(1.+b_g0*b_g0-2.*b_g0*dot(lightDir,n),vec3(1.5));
    vec3 t=b_Q*0.5*(b_P-pos.y)*(b_P-pos.y);
    vec3 s1=exp(b_Q*s*(0.5*s*n.y-(b_P-pos.y))*(1-n.y/lightDir.y));
    vec3 c=0*b_Sun*g*exp(-t/lightDir.y)*(1-s1)/(-n.y+lightDir.y)*max(lightDir.y,0.);
    c+=exp(b_Q*0.5*n.y*s*s-b_Q*(b_P-pos.y)*s)*col;
	 return c;
}



#endif // LIGHT_COLOR_GLSL