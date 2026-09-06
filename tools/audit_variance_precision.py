"""Precision isolation of the raw-moment -> Prepared/debug chain.

Runs synthetic flat-plane, frozen-weight histories; no shader or game changes.
FP16 is IEEE round-to-nearest emulation. FTZ is a separate stress assumption,
not a claim about the active Minecraft driver. Full A-Trous/response/geometry
are not simulated: freezing those decisions isolates numerical error.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 906202613
MODES = ('fp64', 'fp32', 'fp32_capped', 'fp16', 'fp16_ftz')
AXIS = np.array([.36, .48, .8])
TANGENT = np.array([.8, -.6, 0.])
BITANGENT = np.cross(AXIS, TANGENT)


def dtype(mode):
    return np.float64 if mode == 'fp64' else np.float32


def pack(value, mode):
    value = np.asarray(value, dtype=dtype(mode))
    if mode == 'fp32_capped':
        value = np.clip(value, np.float32(-65504), np.float32(65504)).astype(np.float32)
    if mode.startswith('fp16'):
        value = np.clip(value, -65504, 65504).astype(np.float16).astype(np.float32)
        if mode.endswith('ftz'):
            value = np.where(np.abs(value) < 2**-14, 0, value).astype(np.float32)
    return value


def variance(m, rms, n):
    w = np.maximum(m[..., 3], 0)
    # Shader returns before evaluating the ratio at w=0.
    safe_w = np.where(w > 0, w, np.asarray(1., dtype=w.dtype))
    rms = np.maximum(rms, w)
    k2 = np.clip(np.sum(m[..., :3]**2, axis=-1)/(safe_w*safe_w), 0, 1)
    radial = (rms-w)*(rms+w)
    result = (radial + 3*(1-k2)/(3+k2)*rms*rms)/(4*safe_w)
    result = np.where(w > 0, result, np.where(rms > 0, 65504.**2, 0))
    return np.where(n > 1, result/np.maximum(1-1/np.maximum(n, 1), 1e-30), 0)


def describe(v):
    x = np.asarray(v, dtype=np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
        p99=float(np.quantile(x, .99)), maximum=float(x.max()),
        zero_fraction=float(np.mean(x == 0)), finite=bool(np.all(np.isfinite(x))))


def compare(v, ref):
    mask = ref > max(float(np.max(ref))*1e-12, 1e-30)
    rel = np.abs(v[mask]-ref[mask])/ref[mask]
    return dict(relative_p50=float(np.median(rel)), relative_p99=float(np.quantile(rel, .99)),
        relative_max=float(rel.max()),
        collapsed_fraction=float(np.mean(v[mask] == 0))) if np.any(mask) else {}


def spatial(m, r, n):
    # 7x7 sigma=1 Gaussian, exact same sum(w^2/N) as variance_prepare.
    wsum = 0.; ms = np.zeros_like(m); s2 = np.zeros_like(r); q = np.zeros_like(n)
    for y in range(-3, 4):
        for x in range(-3, 4):
            a = np.asarray(np.exp(-.5*(x*x+y*y)), dtype=m.dtype)
            shifted = np.roll(m, (y, x), (0, 1))
            rr = np.roll(r, (y, x), (0, 1))
            nn = np.roll(n, (y, x), (0, 1))
            ms += a*shifted; s2 += a*rr*rr; q += a*a/nn
            wsum = np.asarray(wsum+a, dtype=m.dtype)
    return variance(ms/wsum, np.sqrt(s2/wsum), wsum*wsum/q)


def reproject(m, r, n, fractional):
    if not fractional:
        return m.copy(), r.copy(), n.copy()
    ms=np.zeros_like(m); s2=np.zeros_like(r); invn=np.zeros_like(n)
    for a, shift in zip((.4,.3,.2,.1), ((0,0),(0,1),(1,0),(1,1))):
        a=np.asarray(a,dtype=m.dtype)
        ms += a*np.roll(m,shift,(0,1))
        rr=np.roll(r,shift,(0,1)); nn=np.roll(n,shift,(0,1))
        s2 += a*rr*rr; invn += a/np.sqrt(nn)
    return ms, np.sqrt(s2), 1/(invn*invn)


def sweep():
    rows=[]
    for level in (1e-5, .01, 1., 256., 16384.):
        for cv in (.001,.01,.1,1.):
            for k in (0.,.7,.98,.9999,1.):
                m=np.r_[level*k*AXIS,level][None,:]
                rms=np.array([level*np.sqrt(1+cv*cv)]); n=np.array([32.])
                ref=variance(m,rms,n)
                values={}
                for mode in MODES:
                    v=variance(pack(m,mode),pack(rms,mode),pack(n,mode))
                    values[mode]=dict(value=float(v[0]), **compare(v,ref))
                rows.append(dict(level=level,cv=cv,kappa=k,values=values))
    return rows


def history_case(level, cv, k, fractional=False, frames=512):
    rng=np.random.default_rng(SEED)
    side=48; shape=(side,side)
    initial=np.broadcast_to(np.r_[level*k*AXIS,level],(*shape,4))
    initial_r=np.full(shape,level*np.sqrt(1+cv*cv))
    states={mode:(pack(initial,mode),pack(initial_r,mode),pack(np.full(shape,32.),mode)) for mode in MODES}
    records=[]
    log_sigma=np.sqrt(np.log1p(cv*cv))
    for frame in range(1,frames+1):
        energy=rng.lognormal(-.5*log_sigma**2,log_sigma,shape)*level
        phi=rng.uniform(0,2*np.pi,shape)
        direction=k*AXIS+np.sqrt(1-k*k)*(np.cos(phi)[...,None]*TANGENT+np.sin(phi)[...,None]*BITANGENT)
        raw=np.concatenate((energy[...,None]*direction,energy[...,None]),axis=-1)
        evaluation={}
        for mode,(hm,hr,hn) in states.items():
            cm=pack(raw,mode); cr=pack(energy,mode)
            pm,pr,pn=reproject(hm,hr,hn,fractional)
            # Proposal uses the un-staged FP32 reprojection, alpha=.01.
            a=np.asarray(.01,dtype=dtype(mode)); b=np.asarray(1,dtype=dtype(mode))-a
            m=pack(b*pm+a*cm,mode)
            r=pack(np.sqrt(b*pr*pr+a*cr*cr),mode)
            n=pack(1/(b*b/pn+a*a),mode)
            # Resolve reads the separately packed raw-reprojection scratch.
            sm,sr,sn=pack(pm,mode),pack(pr,mode),pack(pn,mode)
            states[mode]=(pack(b*sm+a*cm,mode),pack(np.sqrt(b*sr*sr+a*cr*cr),mode),pack(1/(b*b/sn+a*a),mode))
            if frame in (1,32,128,512):
                # Shared tile re-packs the same already-half values.
                tiled_m,tiled_r,tiled_n=pack(m,mode),pack(r,mode),pack(n,mode)
                vt=variance(m,r,n); vs=spatial(tiled_m,tiled_r,tiled_n)
                # Prepared blend sweep; this checks both force limits on IDENTICAL input.
                prepared={}
                for start in (1.,32.):
                    trust=np.clip((n-start)/1.,0,1); trust=trust*trust*(3-2*trust)
                    pv=(1-trust)*vs+trust*vt
                    stored_sigma=pack(np.sqrt(pv),mode)
                    color=np.clip(np.log2(1+stored_sigma*stored_sigma)/16,0,1)
                    prepared[str(start)]=dict(variance=describe(stored_sigma*stored_sigma),display=describe(color),trust=describe(trust))
                evaluation[mode]=dict(temporal=vt,spatial=vs,neff=describe(n),prepared=prepared,
                    boundaries=dict(rms_equal_mean_fraction=float(np.mean(r==m[...,3])),
                        rms_below_mean_fraction=float(np.mean(r<m[...,3])),
                        direction_outside_cone_fraction=float(np.mean(np.sum(m[...,:3]**2,axis=-1)>m[...,3]**2))))
        if evaluation:
            ref=evaluation['fp64']; row=dict(frame=frame,modes={})
            for mode,entry in evaluation.items():
                row['modes'][mode]=dict(neff=entry['neff'], prepared=entry['prepared'],
                    boundaries=entry['boundaries'],
                    temporal=describe(entry['temporal']),spatial=describe(entry['spatial']),
                    temporal_error=compare(entry['temporal'],ref['temporal']),spatial_error=compare(entry['spatial'],ref['spatial']))
            records.append(row)
    return dict(level=level,cv=cv,kappa=k,fractional_reprojection=fractional,
        fixed_alpha=.01,initial_neff=32,side=side,frames=frames,records=records)


def lane_test():
    rng=np.random.default_rng(SEED)
    n=rng.uniform(1,65504,10000).astype(np.float16)
    r=np.exp(rng.uniform(-10,10,10000)).astype(np.float16)
    nb=n.view(np.uint16).astype(np.uint32);rb=r.view(np.uint16).astype(np.uint32)
    diffuse=nb|(rb<<16); reflection=rb|(nb<<16)
    assert np.all((diffuse&65535)==nb) and np.all((diffuse>>16)==rb)
    assert np.all((reflection&65535)==rb) and np.all((reflection>>16)==nb)
    # N=32 rounding cannot turn 1-1/N into a small denominator.
    before=np.nextafter(np.float16(32),np.float16(0))
    after=np.nextafter(np.float16(32),np.float16(100))
    return dict(pairs=10000,layout_roundtrip=True,neff_around_32=[float(before),32.,float(after)],
        correction_around_32=[1/(1-1/float(x)) for x in (before,32,after)])


def main():
    # Keep the input distributions identical across arithmetic/storage variants.
    cases=[(256.,1.,.7,False),(256.,.01,1.,False),(256.,.1,.9999,False),
        (256.,.01,1.,True),(1e-5,.1,.98,False),(16384.,1.,.7,False)]
    result=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,
        scope=__doc__,lanes=lane_test(),sweep=sweep(),histories=[history_case(*c) for c in cases])
    assert all(c['records'][-1]['modes'][m]['temporal']['finite']
        and c['records'][-1]['modes'][m]['spatial']['finite'] for c in result['histories'] for m in MODES)
    paths=[Path(__file__), ROOT/'shaders/lib/lighting/denoiser/variance_prepare.glsl',
        ROOT/'shaders/lib/math/statistics.glsl',ROOT/'shaders/lib/common/pack_half.glsl',
        ROOT/'shaders/lib/buffers/diffuse_buffer.glsl',ROOT/'shaders/lib/buffers/specular_buffer.glsl',
        ROOT/'shaders/post/denoiser/diffuse/temporal.glsl',ROOT/'shaders/post/denoiser/reflection/temporal.glsl',
        ROOT/'shaders/post/composite_lighting.glsl']
    result['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    dest=ROOT/'doc/calibration/variance_precision.json'
    dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Wrote',dest)
    print('Lanes:',result['lanes'])
    for c in result['histories']:
        print('Case:', {k:v for k,v in c.items() if k!='records'})
        for mode,m in c['records'][-1]['modes'].items():
            print(mode,'Neff',m['neff']['mean'],'T',m['temporal']['mean'],'S',m['spatial']['mean'],
                'zero',m['temporal']['zero_fraction'],'T relative P99',m['temporal_error'].get('relative_p99'))


if __name__=='__main__':
    main()
