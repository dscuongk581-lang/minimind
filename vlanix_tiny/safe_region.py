"""P0.12E geometry probe: two homogeneous first-order protection constraints."""
import torch

@torch.no_grad()
def project_two(delta, local, readout):
    if delta.ndim != 1 or delta.numel() == 0 or any(v.shape != delta.shape or v.device != delta.device for v in (local,readout)):
        raise ValueError('layout')
    if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in (delta,local,readout)):
        raise ValueError('finite float64 vectors required')
    normals=[]
    for v in (local,readout):
        scale=v.abs().max()
        if scale > 0:
            w=v/scale
            normals.append(w/w.norm())
    if not normals: return delta.clone()
    scale=delta.abs().max()
    if scale == 0: return delta.clone()
    d=delta/scale
    tol=64*torch.finfo(d.dtype).eps*max(1.,d.norm().item())
    def feasible(x): return all(torch.dot(n,x).item() <= tol for n in normals)
    if feasible(d): return delta.clone()
    candidates=[torch.zeros_like(d)]
    for n in normals:
        x=d-torch.dot(n,d)*n
        if feasible(x): candidates.append(x)
    if len(normals)==2:
        a,b=normals
        v=b-torch.dot(a,b)*a
        v=v-torch.dot(a,v)*a
        if v.norm() > 64*torch.finfo(d.dtype).eps:
            v=v/v.norm()
            x=d-torch.dot(a,d)*a-torch.dot(v,d)*v
            if feasible(x): candidates.append(x)
    out=min(candidates,key=lambda x:(x-d).norm().item())*scale
    if not torch.isfinite(out).all(): raise ValueError('nonfinite projection')
    return out
