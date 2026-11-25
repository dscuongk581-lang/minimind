"""Project the actual AdamW displacement onto the old-loss tangent halfspace."""
import math
import torch

@torch.no_grad()
def project_step(params,before,h):
    if not params or not len(params)==len(before)==len(h) or any(p.shape!=b.shape or p.shape!=v.shape for p,b,v in zip(params,before,h)):
        raise ValueError('update layout')
    dot=norm=0.
    for p,b,v in zip(params,before,h):
        d=p.double()-b.double(); w=v.double()
        dot+=(d*w).sum().item(); norm+=w.square().sum().item()
        if not torch.isfinite(d).all(): raise ValueError('nonfinite update')
    if not all(math.isfinite(x) for x in (dot,norm)): raise ValueError('nonfinite update')
    conflict=dot>0 and norm>0
    if conflict:
        coeff=dot/norm
        candidates=[(p.double()-coeff*v.double()).to(p.dtype) for p,v in zip(params,h)]
        if any(not torch.isfinite(c).all() for c in candidates): raise ValueError('nonfinite projection')
        for p,c in zip(params,candidates): p.copy_(c)
    after=sum(((p.double()-b.double())*v.double()).sum().item() for p,b,v in zip(params,before,h))
    return dict(conflict=conflict,dot=dot,projected_dot=after)
