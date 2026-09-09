"""Check the production weighted-sigma cap on GPU against an original snapshot.

The float64 oracle operates on the original weights and endpoint sigmas, not
the rounded GPU accumulators. This complements the high-precision cap proof.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import moderngl
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'temp/bench_sigma'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir',type=Path,required=True)
parser.add_argument('--candidate-dir',type=Path,default=ROOT/'shaders')
args=parser.parse_args()
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(190609)
parts=[]
def add(name,sigma,weight,rho):
 n=len(sigma)
 data=np.zeros((n,10,4),dtype='f4')
 data[:,:9,0]=sigma;data[:,:9,1]=weight
 data[:,9,0]=rho
 parts.append((name,data))
n=12000
f=rng.uniform(0,1,(n,2)).astype('f4')
w=np.zeros((n,9),dtype='f4')
w[:,:4]=np.stack(((1-f[:,0])*(1-f[:,1]),f[:,0]*(1-f[:,1]),(1-f[:,0])*f[:,1],f[:,0]*f[:,1]),axis=-1)
add('bilinear_max',np.full_like(w,65504),w,1)
t=rng.uniform(0,1,n).astype('f4');w=np.zeros((n,9),dtype='f4');w[:,0]=1-t;w[:,1]=t
add('temporal_max',np.full_like(w,65504),w,0)
w=np.exp(rng.uniform(-22,0,(n,9))).astype('f4');w[:,0]=1
s=np.full_like(w,65504);s[:,0]=-2
add('unknown_center_max',s,w,.14262104)
w=np.exp(rng.uniform(-18,0,(n,9))).astype('f4');w[:,0]=1
s=np.exp(rng.uniform(-16,11,(n,9))).astype('f2').astype('f4');s[rng.uniform(size=s.shape)<.2]=-2
add('mixed_finite',s,w,.28524208)
s=np.full((200,9),-2,dtype='f4');w=np.ones_like(s)
add('all_unknown',s,w,.14262104)
w=np.zeros((200,9),dtype='f4');w[:,0]=1;w[:,1]=np.exp2(np.linspace(-149,-100,200)).astype('f4')
s=np.full_like(w,-2);s[:,1]=65504
add('tiny_donor',s,w,.14262104)
w=np.ones((6,9),dtype='f4');s=np.full_like(w,np.nan)
s[1]=np.inf;s[2]=-np.inf;s[3]=-1;s[4]=65505;s[5]=0
add('invalid_and_zero',s,w,.14262104)
data=np.concatenate([p[1] for p in parts])
strict=np.array([0,65504.0**2,np.nextafter(np.float32(65504.0**2),np.float32(np.inf)),np.inf,np.nan,-1],dtype='f4')
data[:,9,1]=np.resize(strict,len(data))
ctx=moderngl.create_standalone_context(require=430)
source_buffer=ctx.buffer(data.tobytes());source_buffer.bind_to_storage_buffer(0)
outputs=[]
trees=[args.baseline_dir.resolve(),args.candidate_dir.resolve()]
hashes=[]
for tree in trees:
 lib=(tree/'lib/math/denoiser_uncertainty.glsl').read_text(encoding='utf8')
 source='''#version 430 core
layout(local_size_x=128) in;
layout(std430,binding=0) readonly buffer InputData { vec4 samples[]; };
layout(std430,binding=1) writeonly buffer OutputData { vec2 results[]; };
'''+lib+'''
void main() {
 uint p=gl_GlobalInvocationID.x;
 if(p>=uint(results.length()))return;
 DenoiserEstimatorVarianceAccumulator a=denoiserBeginEstimatorVariance();
 for(uint i=0u;i<9u;++i) {
  vec2 v=samples[p*10u+i].xy;
  denoiserAccumulateEstimatorVariance(a,v.x,v.y);
 }
 results[p]=vec2(denoiserResolveEstimatorSigma(a,samples[p*10u+9u].x),
   denoiserVarianceToSigma(samples[p*10u+9u].y));
}
'''
 shader=ctx.compute_shader(source);output=ctx.buffer(reserve=len(data)*8);output.bind_to_storage_buffer(1)
 shader.run((len(data)+127)//128);ctx.memory_barrier()
 outputs.append(np.frombuffer(output.read(),dtype='f4').reshape(-1,2).copy())
 hashes.append(hashlib.sha256(lib.encode()).hexdigest());output.release();shader.release()
assert np.array_equal(outputs[0][:,1],outputs[1][:,1]),'strict conversion changed'
old,new=outputs[0][:,0],outputs[1][:,0]
assert np.all(np.isfinite(new))
known=old>=0
assert np.array_equal(old[known].view('u4'),new[known].view('u4')),'ordinary known output changed'
rows=[];offset=0
for name,p in parts:
 n=len(p);s=p[:,:9,0].astype('f8');w=p[:,:9,1].astype('f8');rho=p[:,9,0].astype('f8')
 k=(s>=0)&(s<=65504);valid_w=np.isfinite(w)&(w>0)
 w=np.where(valid_w,w,0);sw=np.where(k,s,0)*w
 K=np.sum(np.where(k,w,0),axis=1);W=w.sum(axis=1)
 with np.errstate(divide='ignore',invalid='ignore'):
  d=sw.sum(axis=1)/K
  q=np.sum(np.where(k,0,w*w),axis=1)
  variance=(1-rho)*(np.sum(sw*sw,axis=1)+q*d*d)/(W*W)+rho*d*d
  oracle=np.sqrt(variance)
 before=old[offset:offset+n];after=new[offset:offset+n]
 fixed=(before<0)&(after>=0)
 assert np.all(np.isfinite(oracle[fixed])) and np.all(oracle[fixed]<=65504*(1+1e-12))
 err=np.abs(after[fixed]-oracle[fixed])/np.maximum(oracle[fixed],1e-30)
 assert np.all(err<1e-5),(name,err.max())
 rows.append(dict(case=name,count=n,baseline_unknown=int(np.sum(before<0)),candidate_unknown=int(np.sum(after<0)),
  corrected_false_unknown=int(fixed.sum()),max_corrected_relative_sigma_error=float(err.max(initial=0))))
 offset+=n
assert sum(r['corrected_false_unknown'] for r in rows)>0
report=dict(gpu=ctx.info['GL_RENDERER'],driver=ctx.info['GL_VERSION'],source_sha256=hashes,rows=rows,
 ordinary_known_outputs='bitwise equal',strict_variance_ingress='unchanged; NaN/Inf/negative/above-cap remain unknown',
 note='Float64 oracle plus separately recorded 80-digit CPU proof. Existing tiny-donor FTZ limitations remain.')
(OUT/'cap_gpu_check.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
source_buffer.release();ctx.release()
