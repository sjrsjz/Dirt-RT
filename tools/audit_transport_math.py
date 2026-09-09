"""Independent transport references, with optional execution of production GLSL.

Run with --gpu for an OpenGL compute audit (numpy + moderngl). This checks the
actual arithmetic on the selected GPU; it does not execute the Vulkan RT host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shader_compile import ROOT, expand


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smith_lambda(c, alpha):
    return (np.sqrt(1 + alpha**2 * (1/c**2 - 1)) - 1) / 2


def dielectric(c, eta):
    transmitted = np.sqrt(np.maximum(1 - eta**2 * (1-c*c), 0))
    a = (eta*c-transmitted) / np.maximum(eta*c+transmitted, 1e-30)
    b = (eta*transmitted-c) / np.maximum(eta*transmitted+c, 1e-30)
    return np.where(eta == 1, 0, (a*a+b*b)/2)


def check_close(name, actual, expected, rtol=2e-4, atol=2e-6):
    assert np.all(np.isfinite(actual)), name
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    return float(np.max(np.abs(actual-expected)))


def cpu_audit():
    rng = np.random.default_rng(0x46525054)
    v, l = 10**rng.uniform(-5, 0, (2, 100_000))
    alpha = 10**rng.uniform(-4, 0, 100_000)
    lv, ll = smith_lambda(v, alpha), smith_lambda(l, alpha)
    sv = np.sqrt(alpha**2 + (1-alpha**2)*v*v)
    sl = np.sqrt(alpha**2 + (1-alpha**2)*l*l)
    ratio = l*(v+sv)/(l*sv+v*sl)
    error = check_close('Smith ratio', ratio, (1+lv)/(1+lv+ll), 3e-14, 3e-14)
    # Independent Jacobian form of the radiance-mode rough BTDF / VNDF PDF.
    eta = rng.uniform(.65, 1.6, 100_000)
    vh, lh = rng.uniform(.01, 1, (2, 100_000))
    d = 10**rng.uniform(-7, 7, 100_000)
    denominator = (-lh + vh*eta)**2
    ft_no_l = .7*d/(1+lv+ll)*lh*vh/(v*denominator)*eta**2
    pdf = d/(1+lv)*vh/v*lh/denominator
    transmission_error = check_close('BTDF cancellation', ft_no_l/pdf,
                                     .7*eta**2*ratio, 3e-14, 3e-14)
    a = np.float32(1e-4)
    old_denom = np.float32(1) + (a*a-np.float32(1))*np.float32(1)
    old_peak = float(a*a/np.maximum(np.float32(np.pi)*old_denom**2, 1e-20))
    reference_peak = 1/(np.pi*float(a)**2)
    return {'cases': 100_000, 'smith_max_absolute_error': error,
            'transmission_max_absolute_error': transmission_error,
            'old_fp32_peak_over_reference': old_peak/reference_peak}


def shader_source():
    # Use production modules verbatim. Only the material type/albedo adapter
    # is extracted from scene.glsl to avoid requiring RT resources in OpenGL.
    scene = (ROOT/'shaders/lib/rt/raytrace/scene.glsl').read_text(encoding='utf-8')
    material = scene[scene.index('struct material {'):scene.index('Material evaluateMaterial(')]
    adapters = scene[scene.index('int transportBlockFromMaterial('):scene.index('// The compact shadow payload')]
    source = '#version 430\n#define MC_GL_NV_gpu_shader5 1\n'
    source += expand(ROOT/'shaders/lib/common.glsl') + '\n'
    source += '#undef EON_ENABLED\n#define EON_ENABLED 0\n'
    source += expand(ROOT/'shaders/lib/lighting/maxent.glsl') + '\n'
    source += material + adapters + '\nconst float cosD_S = 0.999;\n'
    for module in ('types', 'bsdf', 'lobe_selection'):
        source += expand(ROOT/f'shaders/lib/rt/raytrace/{module}.glsl') + '\n'
    return source + '''
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Inputs { vec4 data[]; } inputs;
layout(std430,binding=1) writeonly buffer Outputs { vec4 data[]; } outputs;
void main() {
    uint i=gl_GlobalInvocationID.x;
    vec4 a=inputs.data[5u*i], b=inputs.data[5u*i+1u];
    vec4 c=inputs.data[5u*i+2u], d=inputs.data[5u*i+3u];
    vec4 e=inputs.data[5u*i+4u];
    vec3 n=a.xyz, wo=b.xyz, wi=c.xyz;
    vec3 tangent,bitangent;
    orthonormalBasis(n,tangent,bitangent);
    vec3 draw=sample_maxent_guiding(n,a.w,d.xy);
    outputs.data[8u*i]=vec4(draw,maxent_guiding_pdf(draw,n,a.w));
    outputs.data[8u*i+1u]=vec4(tangent,GGX_D(d.z,b.w));
    outputs.data[8u*i+2u]=vec4(bitangent,GGX_G2(dot(n,wo),dot(n,wi),b.w));
    vec3 f,response;float pdf;
    evaluateSpecularBRDF(wo,wi,n,vec3(.04),1.0,0.0,c.w,b.w,f,pdf,response);
    outputs.data[8u*i+3u]=vec4(f,pdf);
    outputs.data[8u*i+4u]=vec4(response,
        maxent_cosine_response(a.w,d.w));
    vec3 transmitted=refract(-wo,n,c.w), weight=vec3(0.0);
    if(dot(transmitted,transmitted)>0.0)
        sampleTransmissionWeight(wo,transmitted,n,n,vec3(1),c.w,b.w,weight);
    outputs.data[8u*i+5u]=vec4(weight,evaluateDisneyDiffuseFactor(wo,wi,n,b.w));
    material surf=newMaterial(vec3(e.x),vec3(e.y),vec2(1,e.z),vec4(b.w,0,0,0),vec3(0));
    LobeProbs lobes=computeLobeProbs(surf,-wo,n,c.w);
    outputs.data[8u*i+6u]=vec4(lobes.P_spec,lobes.P_refr,lobes.P_diff,fresnel(wo,n,c.w));
    setFrame(uint(e.w));wseed3=uvec3(i+1u,i+2u,i+3u);
    outputs.data[8u*i+7u]=vec4(getRandom(),getRandom(),weyl2());
}
'''


def gpu_payload_audit(ctx):
    source='#version 430\n'+expand(ROOT/'shaders/lib/rt/payload_pack.glsl')
    source+='''
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Input {float data[];} inputData;
layout(std430,binding=1) writeonly buffer Output {uvec2 data[];} outputData;
void main(){
    uint i=gl_GlobalInvocationID.x,flags=i&15u;
    uint words[PAYLOAD_SLOTS];
    payload_packFlags(words,inputData.data[i],(flags&1u)!=0u,
        (flags&2u)!=0u,(flags&4u)!=0u,(flags&8u)!=0u);
    bool inside,handedness,nee,ignoreTransmissive;
    float distance=payload_unpackFlags(words,inside,handedness,nee,ignoreTransmissive);
    uint decoded=uint(inside)|(uint(handedness)<<1u)|(uint(nee)<<2u)|(uint(ignoreTransmissive)<<3u);
    outputData.data[i]=uvec2(floatBitsToUint(distance),decoded);
}
'''
    rng=np.random.default_rng(0x5041594c)
    distances=rng.uniform(0,2048,65536).astype('f4')
    distances[:16]=0
    program=ctx.compute_shader(source)
    inputs=ctx.buffer(distances.tobytes());inputs.bind_to_storage_buffer(0)
    outputs=ctx.buffer(reserve=65536*8);outputs.bind_to_storage_buffer(1)
    program.run(65536//64);ctx.memory_barrier()
    data=np.frombuffer(outputs.read(),dtype='u4').reshape(-1,2)
    np.testing.assert_array_equal(data[:,0].copy().view('f4'),distances.astype('f2').astype('f4'))
    np.testing.assert_array_equal(data[:,1],np.arange(65536,dtype='u4')&15)
    return {'cases':65536,'all_flag_masks':16,'distance_roundtrip':'exact FP16 conversion'}


def gpu_vndf_audit(ctx):
    # Compare observed acceptance and throughput to independent solid-angle
    # quadrature. VNDF reflection has a null-event mass below the hemisphere;
    # renormalizing those rejections would fail this test.
    source = '#version 430\n#define MC_GL_NV_gpu_shader5 1\n'
    source += expand(ROOT/'shaders/lib/common.glsl')
    source += '''
layout(local_size_x=64) in;
uniform float alpha,viewCosine;
layout(std430,binding=0) readonly buffer Xi {vec2 data[];} xi;
layout(std430,binding=1) writeonly buffer Result {vec4 data[];} result;
void main(){
    uint i=gl_GlobalInvocationID.x;
    vec3 n=vec3(0,0,1),wo=vec3(sqrt(1-viewCosine*viewCosine),0,viewCosine);
    vec3 h=GGXVNDFNormal(n,wo,alpha,xi.data[i]);
    vec3 wi=reflect(-wo,h);
    float weight=wi.z>0.0?GGX_G2(viewCosine,wi.z,alpha):0.0;
    result.data[i]=vec4(wi.z>0.0?1.0:0.0,weight,dot(h,h),dot(wo,h));
}
'''
    program=ctx.compute_shader(source)
    count=262144;rng=np.random.default_rng(0x564e4446)
    inputs=ctx.buffer(rng.random((count,2),dtype='f4').tobytes())
    outputs=ctx.buffer(reserve=count*16)
    inputs.bind_to_storage_buffer(0);outputs.bind_to_storage_buffer(1)
    x,w=np.polynomial.legendre.leggauss(256)
    cosine=(x+1)/2;weights=w/2
    phi=(np.arange(512)+.5)*(2*np.pi/512)
    wi=np.stack(np.broadcast_arrays(np.sqrt(1-cosine[:,None]**2)*np.cos(phi),
        np.sqrt(1-cosine[:,None]**2)*np.sin(phi),cosine[:,None]),axis=-1)
    reports=[]
    for alpha,view in ((1.,1.),(.5,.3),(.2,.8)):
        program['alpha']=alpha;program['viewCosine']=view
        program.run(count//64);ctx.memory_barrier()
        data=np.frombuffer(outputs.read(),dtype='f4').reshape(count,4)
        check_close('VNDF half-vector length',data[:,2],1.,atol=8e-7)
        assert np.all(data[:,3]>0), 'visible-normal sampler returned hidden facets'
        wo=np.array([np.sqrt(1-view*view),0,view]);h=normalize(wi+wo)
        nh=h[...,2];D=alpha**2/(np.pi*((1-nh*nh)+alpha**2*nh*nh)**2)
        lv=smith_lambda(view,alpha);ll=smith_lambda(cosine,alpha)
        q=D/(1+lv)/(4*view)
        f=D/(1+lv+ll[:,None])/(4*view)
        expected=np.array([np.sum(a*weights[:,None])*2*np.pi/512 for a in (q,f)])
        measured=data[:,:2].mean(axis=0,dtype='f8')
        se=data[:,:2].std(axis=0,dtype='f8')/np.sqrt(count)
        assert np.all(abs(measured-expected)<6*se+2e-5)
        reports.append({'alpha':alpha,'view_cosine':view,
            'observed_acceptance_and_energy':measured.tolist(),
            'quadrature_acceptance_and_energy':expected.tolist()})
    return reports


def gpu_audit():
    import moderngl
    ctx = moderngl.create_standalone_context(require=430)
    rng = np.random.default_rng(0x46525054)
    count = 65536
    n = normalize(rng.normal(size=(count, 3)))
    n[:6] = [[0,0,1], [0,0,-1], [0,1,0], [0,-1,0], [1,0,0], [-1,0,0]]
    tangent = normalize(np.cross(n, np.roll(n, 1, axis=1)+[.13,.27,.49]))
    bitangent = np.cross(n,tangent)
    def hemisphere():
        mu = rng.uniform(.002, 1, count)
        phi = rng.uniform(0, 2*np.pi, count)
        return n*mu[:,None] + np.sqrt(1-mu**2)[:,None]*(
            tangent*np.cos(phi)[:,None]+bitangent*np.sin(phi)[:,None])
    k = rng.choice([0.,1e-8,1e-6,.001,.009999,.01,.5,.9,.99,1-1e-6],count)
    alpha = 10**rng.uniform(-4,0,count)
    eta = rng.choice([1.,1.5,1/1.5,1.31,1/1.31],count)
    xi = rng.random((count,2))
    xi[:6,0]=[0,0,.5,.5,1-2**-24,1-2**-24]
    no_h = rng.uniform(0,1,count); no_h[:256]=1
    values = np.zeros((count,5,4),dtype='f4')
    values[:,0,:3]=n;values[:,0,3]=k
    values[:,1,:3]=hemisphere();values[:,1,3]=alpha
    values[:,2,:3]=hemisphere();values[:,2,3]=eta
    values[:,3,:2]=xi;values[:,3,2]=no_h
    values[:,3,3]=rng.uniform(-1,1,count)
    values[:,4,0]=.04;values[:,4,1]=1;values[:,4,2]=1
    values[:,4,3]=2**25
    folder=ROOT/'temp/transport_math';folder.mkdir(parents=True,exist_ok=True)
    source=shader_source();(folder/'audit.comp').write_text(source,encoding='utf-8')
    program=ctx.compute_shader(source)
    input_buffer=ctx.buffer(values.tobytes()); input_buffer.bind_to_storage_buffer(0)
    output_buffer=ctx.buffer(reserve=count*8*16); output_buffer.bind_to_storage_buffer(1)
    program.run(count//64);ctx.memory_barrier()
    out=np.frombuffer(output_buffer.read(),dtype='f4').reshape(count,8,4).astype('f8')
    assert np.isfinite(out).all()
    v=values.astype('f8');n=v[:,0,:3];k=v[:,0,3];wo=v[:,1,:3];alpha=v[:,1,3]
    wi=v[:,2,:3];eta=v[:,2,3];xi=v[:,3,:2];no_h=v[:,3,2]
    errors={}
    frame=np.stack([out[:,1,:3],out[:,2,:3],n],axis=-1)
    errors['frame']=check_close('orthonormal frame',frame.transpose(0,2,1)@frame,
                               np.broadcast_to(np.eye(3),(count,3,3)),atol=5e-7)
    errors['draw_length']=check_close('unit sample',np.linalg.norm(out[:,0,:3],axis=1),1.,atol=6e-7)
    # Long-double inverse CDF reference, rationalized at small k only to keep
    # reference precision; the shader never executes this CPU expression.
    kl=k.astype(np.longdouble);x=xi[:,0].astype(np.longdouble)
    t=1/np.sqrt((1-x)/(1+kl)**2+x/(1-kl)**2)
    mu=np.divide(1-t,kl,out=2*x-1,where=kl!=0).astype('f8')
    errors['inverse_cdf']=check_close('CDF inversion',np.sum(n*out[:,0,:3],axis=1),mu,atol=2e-5)
    ref_d=alpha**2/(np.pi*((1-no_h**2)+alpha**2*no_h**2)**2)
    errors['ggx_D']=check_close('GGX distribution',out[:,1,3],ref_d,rtol=2e-6)
    nv=np.sum(n*wo,axis=1);nl=np.sum(n*wi,axis=1)
    lv=smith_lambda(nv,alpha);ll=smith_lambda(nl,alpha)
    ratio=(1+lv)/(1+lv+ll)
    errors['masking']=check_close('GGX masking',out[:,2,3],ratio,rtol=5e-5)
    h=normalize(wo+wi);nh=np.sum(n*h,axis=1);vh=np.sum(wo*h,axis=1)
    D=alpha**2/(np.pi*((1-nh**2)+alpha**2*nh**2)**2)
    F=.04+.96*(1-np.abs(vh))**5
    response=F*ratio
    errors['reflection_response']=check_close('reflection response',out[:,4,0],response,rtol=6e-5)
    # Direction normalization/dots are FP32 on the GPU. Bound amplification
    # by the narrow-lobe denominator; the scalar D test above isolates its
    # arithmetic at identical inputs, including NoH=1 and alpha near 1e-4.
    conditioning = 16*np.finfo('f4').eps / ((1-nh**2)+alpha**2*nh**2)
    for name, actual, expected in (
            ('reflection_pdf',out[:,3,3],D/(1+lv)/(4*nv)),
            ('reflection_f',out[:,3,0],F*D/(1+lv+ll)/(4*nv))):
        relative = np.abs(actual-expected)/np.maximum(expected,1e-20)
        assert np.all(relative < conditioning+1e-4), name
        errors[name+'_relative']=float(relative.max())
    p=1-k*k;q=k*v[:,3,3];root=np.sqrt(p+q*q)
    cosine=(root+q)**2/(4*root)
    errors['cosine']=check_close('cosine response',out[:,4,3],cosine)
    Fd90=.5+2*np.sqrt(alpha)*np.sum(wi*h,axis=1)**2
    disney=(1+(Fd90-1)*(1-nl)**5)*(1+(Fd90-1)*(1-nv)**5)
    errors['disney']=check_close('Disney factor',out[:,5,3],disney)
    fn=dielectric(nv,eta)
    errors['fresnel']=check_close('dielectric Fresnel',out[:,6,3],fn,rtol=3e-4,atol=1e-5)
    cos_t=np.sqrt(np.maximum(1-eta**2*(1-nv**2),0))
    ratio_t=(1+lv)/(1+lv+smith_lambda(np.maximum(cos_t,1e-20),alpha))
    expected_t=np.where(eta==1,1,(1-fn)*eta**2*ratio_t)
    # Snell's square root is ill-conditioned at the critical angle. Keep an
    # absolute radiance-weight budget there instead of a relative-only test.
    errors['transmission']=check_close('sampled transmission',out[:,5,0],expected_t,rtol=5e-4,atol=1e-4)
    tir=fn>=1-1e-8
    assert np.all(out[tir,6,1]>0), 'rough TIR must retain a transmission proposal'
    check_close('lobe sum',out[:,6,:3].sum(axis=1),1.)
    assert np.all((out[:,7]>=0)&(out[:,7]<1))
    unique=len(np.unique(out[:,7,0]))
    assert unique>count*.98, 'late-frame RNG lost its fractional bits'
    # Sampler moment check uses independent random xi across each kappa stratum.
    moment_errors={}
    measured=np.sum(out[:,0,:3]*n,axis=1)
    for concentration in np.unique(k):
        subset=measured[k==concentration]
        delta=float(abs(np.mean(subset)-concentration))
        assert delta<6*np.std(subset)/np.sqrt(len(subset))+2e-6
        moment_errors[str(concentration)]=delta
    vndf=gpu_vndf_audit(ctx)
    payload=gpu_payload_audit(ctx)
    return {'gpu':ctx.info['GL_RENDERER'],'driver':ctx.info['GL_VERSION'],
            'cases':count,'max_absolute_errors':errors,
            'sample_moment_errors':moment_errors,'rough_tir_cases':int(tir.sum()),
            'late_frame_unique_rng_values':unique,
            'vndf_quadrature':vndf,
            'payload':payload,
            'scope':'production GLSL compute kernels; no Vulkan RT traversal or frame timing'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    result={'cpu':cpu_audit()}
    if args.gpu: result['gpu']=gpu_audit()
    folder=ROOT/'temp/transport_math';folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
