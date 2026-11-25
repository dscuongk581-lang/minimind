"""Project conflicting acquisition gradients before AdamW preconditioning."""
import math
import torch

@torch.no_grad()
def project_acquisition(g,h):
    if not g or len(g)!=len(h) or any(a.shape!=b.shape for a,b in zip(g,h)):
        raise ValueError('gradient layout')
    dot=sum((a.double()*b.double()).sum().item() for a,b in zip(g,h))
    norm=sum(b.double().square().sum().item() for b in h)
    gn=sum(a.double().square().sum().item() for a in g)
    if not all(math.isfinite(v) for v in (dot,norm,gn)):
        raise ValueError('nonfinite gradient')
    conflict=dot<0 and norm>0
    coeff=dot/norm if conflict else 0.
    projected=[a-coeff*b for a,b in zip(g,h)]
    projected_dot=sum((a.double()*b.double()).sum().item() for a,b in zip(projected,h))
    return [a+b for a,b in zip(projected,h)],dict(conflict=conflict,dot=dot,projected_dot=projected_dot,coefficient=coeff)
