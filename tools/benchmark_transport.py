"""Compare isolated production kernels with a git revision on the local GPU.

These compute timings measure math + SSBO traffic, never full-frame RT speed.
Example: python -B tools/benchmark_transport.py --baseline-ref <revision>
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess

import moderngl
import numpy as np

from shader_compile import ROOT, expand


def historical_expander(revision):
    @lru_cache(None)
    def read(path):
        return subprocess.check_output(['git','show',f'{revision}:{path}'],
            cwd=ROOT).decode('utf-8-sig')
    @lru_cache(None)
    def old_expand(path):
        lines=[]
        for line in read(path).splitlines():
            match=re.match(r'\s*#include\s+"([^"]+)"',line)
            if match:
                name=match[1]
                child='shaders/'+name.lstrip('/') if name.startswith('/') else str(Path(path).parent/name).replace('\\','/')
                lines.append(old_expand(child))
            else:lines.append(line)
        return '\n'.join(lines)
    return read,old_expand


def build_source(kernel, read, expand_source, baseline):
    source='#version 430\n#define MC_GL_NV_gpu_shader5 1\n'
    source+=expand_source('shaders/lib/common.glsl')+'\n'
    source+='#undef EON_ENABLED\n#define EON_ENABLED 0\n'
    source+=expand_source('shaders/lib/lighting/maxent.glsl')+'\n'
    source+='const float cosD_S=0.999;\n'
    if baseline:
        transport=read('shaders/lib/rt/raytrace/transport.glsl')
        # All functions before lobe selection are independent of scene I/O.
        source+=transport[:transport.index('// ===========================================================================\n// BSDF Lobe Probabilities')]
        source+='\n#endif\n'
    else:
        for module in ('types','bsdf'):
            source+=expand_source(f'shaders/lib/rt/raytrace/{module}.glsl')+'\n'
    operations={
        'reflection': '''vec3 f,response;float pdf;
            evaluateSpecularBRDF(wo,wi,n,vec3(.04),1.0,0.0,eta,alpha,f,pdf%s);
            %s
            result=vec4(response+f,pdf);''' % (
                '' if baseline else ',response',
                'response=evaluateSpecularQLiResponse(wo,wi,n,vec3(.04),1.0,0.0,eta,alpha);' if baseline else ''),
        'transmission': ('''vec3 f;float pdf;
            evaluateTransmissionBSDF(wo,wi,n,n,vec3(1),eta,alpha,f,pdf);
            result=vec4(pdf>1e-8?f/pdf:vec3(0),0);''' if baseline else '''
            vec3 weight;sampleTransmissionWeight(wo,wi,n,n,vec3(1),eta,alpha,weight);
            result=vec4(weight,0);'''),
        'guiding': '''vec3 direction=sample_maxent_guiding(n,kappa,xi);
            result=vec4(direction,maxent_guiding_pdf(direction,n,kappa));''',
    }
    return source+'''
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer InputData {vec4 data[];} inputData;
layout(std430,binding=1) writeonly buffer OutputData {vec4 data[];} outputData;
void main(){
    uint i=gl_GlobalInvocationID.x;
    vec4 a=inputData.data[i*4u],b=inputData.data[i*4u+1u];
    vec4 c=inputData.data[i*4u+2u],d=inputData.data[i*4u+3u];
    vec3 n=a.xyz,wo=b.xyz,wi=c.xyz;float kappa=a.w,alpha=b.w,eta=c.w;
    vec2 xi=d.xy;vec4 result;
'''+operations[kernel]+'''
    outputData.data[i]=result;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref',required=True)
    args=parser.parse_args()
    revision=subprocess.check_output(['git','rev-parse',args.baseline_ref],cwd=ROOT,text=True).strip()
    old_read,old_expand=historical_expander(revision)
    read=lambda p:(ROOT/p).read_text(encoding='utf-8')
    current_expand=lambda p:expand(ROOT/p)
    folder=ROOT/'temp/transport_benchmark';folder.mkdir(parents=True,exist_ok=True)
    ctx=moderngl.create_standalone_context(require=430)
    count=262144;repeats=8;trials=21
    rng=np.random.default_rng(0x42534446)
    values=np.zeros((count,4,4),dtype='f4')
    values[:,0,2]=1;values[:,0,3]=rng.uniform(0,.99,count)
    for idx in (1,2):
        direction=rng.normal(size=(count,3));direction[:,2]=abs(direction[:,2])+.01
        values[:,idx,:3]=direction/np.linalg.norm(direction,axis=1,keepdims=True)
    values[:,1,3]=10**rng.uniform(-4,0,count)
    values[:,2,3]=1/1.5
    values[:,3,:2]=rng.random((count,2))
    input_buffer=ctx.buffer(values.tobytes());input_buffer.bind_to_storage_buffer(0)
    output_buffer=ctx.buffer(reserve=count*16);output_buffer.bind_to_storage_buffer(1)
    report={'baseline_revision':revision,'gpu':ctx.info['GL_RENDERER'],
            'driver':ctx.info['GL_VERSION'],'invocations':count,
            'repeats':repeats,'trials':trials,'kernels':{},
            'scope':'isolated compute math and SSBO traffic; not in-game frame time'}
    for kernel in ('reflection','transmission','guiding'):
        programs=[]
        for label,is_old,reader,expander in (
                ('baseline',True,old_read,old_expand),('current',False,read,current_expand)):
            source=build_source(kernel,reader,expander,is_old)
            (folder/f'{kernel}_{label}.comp').write_text(source,encoding='utf-8')
            programs.append(ctx.compute_shader(source))
        data=values.copy()
        if kernel=='transmission':
            wo=data[:,1,:3].astype('f8');eta=data[:,2,3].astype('f8');mu=wo[:,2]
            data[:,2,:3]=-eta[:,None]*wo
            data[:,2,2]+=eta*mu-np.sqrt(1-eta**2*(1-mu**2))
        input_buffer.write(data.tobytes())
        for program in programs:
            for _ in range(4):program.run(count//64)
        ctx.finish()
        times=[[],[]]
        for trial in range(trials):
            for which in ((0,1) if trial%2==0 else (1,0)):
                with ctx.query(time=True) as query:
                    for _ in range(repeats):
                        programs[which].run(count//64)
                        ctx.memory_barrier()
                ctx.finish();times[which].append(query.elapsed/1e6/repeats)
        medians=[float(np.median(t)) for t in times]
        report['kernels'][kernel]={'baseline_ms':medians[0],'current_ms':medians[1],
            'current_over_baseline':medians[1]/medians[0],'samples_ms':times}
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({**report,'kernels':{k:{n:v for n,v in r.items() if n!='samples_ms'}
                     for k,r in report['kernels'].items()}},indent=2))


if __name__=='__main__':main()
