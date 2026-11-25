"""One-sided global gradient projection, before AdamW preconditioning."""
import math
import torch

@torch.no_grad()
def combine(g,h,project):
    if not g or len(g)!=len(h) or any(a.shape!=b.shape for a,b in zip(g,h)):
        raise ValueError('gradient layout')
    dot=sum(torch.sum(a.double()*b.double()).item() for a,b in zip(g,h))
    norm=sum(torch.sum(b.double().square()).item() for b in h)
    gn=sum(torch.sum(a.double().square()).item() for a in g)
    if not all(math.isfinite(v) for v in (dot,norm,gn)): raise ValueError('nonfinite gradient')
    conflict=dot<0 and norm>0
    coeff=dot/norm if project and conflict else 0.
    corrected=[a-coeff*b for a,b in zip(g,h)]
    projected_dot=sum(torch.sum(a.double()*b.double()).item() for a,b in zip(corrected,h))
    return [a*.75+b*.25 for a,b in zip(corrected,h)],dict(conflict=conflict,dot=dot,projected_dot=projected_dot,coefficient=coeff)
