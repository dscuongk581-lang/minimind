"""Apply the two-constraint projection to real, rounded parameter updates."""
import torch
from vlanix_tiny.safe_region import project_two

@torch.no_grad()
def project_core_step(params,before,local,readout):
    if not params or not len(params)==len(before)==len(local)==len(readout) or any(p.shape!=b.shape or p.shape!=a.shape or p.shape!=r.shape for p,b,a,r in zip(params,before,local,readout)):
        raise ValueError('layout')
    def flat(xs): return torch.cat([x.detach().double().reshape(-1) for x in xs])
    base=flat(before); d=flat(params)-base; a=flat(local); r=flat(readout)
    x=project_two(d,a,r)
    raw_a=torch.dot(a,d).item(); raw_r=torch.dot(r,d).item()
    def rounded(v):
        return [(b.double()+t.reshape(b.shape)).to(p.dtype) for p,b,t in zip(params,before,v.split([p.numel() for p in params]))]
    def measured(c):
        y=flat(c)-base
        return torch.dot(a,y).item(),torch.dot(r,y).item(),bool(torch.count_nonzero(y).item())
    c=rounded(x); da,dr,moved=measured(c)
    # Try a small inward correction if storage rounding reintroduces conflict.
    repaired=False
    if da>0 or dr>0:
        normals=[v/v.norm() for v in (a,r) if v.norm()>0]
        inward=sum(normals,torch.zeros_like(d))
        slopes=[torch.dot(v,inward).item() for v in (a,r)]
        if all(s>0 or q<=0 for s,q in zip(slopes,(da,dr))):
            step=max([q/s for s,q in zip(slopes,(da,dr)) if s>0]+[0.])
            for factor in (2.,8.,32.):
                trial=rounded(x-factor*step*inward)
                ta,tr,tm=measured(trial)
                if ta<=0 and tr<=0:
                    c,da,dr,moved=trial,ta,tr,tm; repaired=True; break
    fallback=da>0 or dr>0
    if fallback:
        c=list(before); da=dr=0.; moved=False
    if any(not torch.isfinite(v).all() for v in c): raise ValueError('nonfinite candidate')
    for p,v in zip(params,c): p.copy_(v)
    return dict(raw_local=raw_a,raw_readout=raw_r,after_local=da,after_readout=dr,moved=moved,repaired=repaired,fallback=fallback)
