"""P0.11B: tied embedding learning-rate intervention; keep full gradients."""
import math
import torch

def make_optimizer(model,lr):
    e=model.model.embed_tokens.weight
    if e is not model.lm_head.weight or not e.requires_grad or tuple(e.shape)!=(265,256):
        raise ValueError('tied trainable embedding required')
    if not math.isfinite(lr) or lr<=0: raise ValueError('lr')
    core=[p for p in model.parameters() if p is not e]
    return torch.optim.AdamW([dict(params=core,role='core'),dict(params=[e],role='embedding')],lr=lr)

def set_policy(opt,scale):
    if scale not in (0.,.1,1.) or [g['role'] for g in opt.param_groups]!=['core','embedding']:
        raise ValueError('policy')
    opt.param_groups[1]['lr']=opt.param_groups[0]['lr']*scale

@torch.no_grad()
def geometry(before,after):
    if tuple(before.shape)!=(265,256) or before.shape!=after.shape:
        raise ValueError('shape')
    a=before.detach(); b=after.detach(); delta=b-a
    if not torch.isfinite(delta).all(): raise ValueError('nonfinite embedding')
    return dict(relative_l2=(delta.norm()/a.norm().clamp_min(1e-12)).item(),
        max_abs=delta.abs().max().item(),exact=torch.equal(a,b),
        bit_sign_agreement=(a[9:,:8].sign()==b[9:,:8].sign()).float().mean().item())
