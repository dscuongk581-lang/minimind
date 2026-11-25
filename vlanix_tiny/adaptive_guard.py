"""Bounded learning-rate backtracking with transactional CPU rollback."""
import math
from vlanix_tiny.protected_core import guarded_update

SCALES=(1.,.25,.0625,.015625)

def backtrack(model,opt,update,check,scales=SCALES):
    if not scales or any(not math.isfinite(s) or not 0<s<=1 for s in scales) or any(a<=b for a,b in zip(scales,scales[1:])):
        raise ValueError('scales')
    rates=[g['lr'] for g in opt.param_groups]
    if any(not math.isfinite(r) or r<0 for r in rates): raise ValueError('rates')
    try:
        for i,scale in enumerate(scales,1):
            def proposal():
                for g,r in zip(opt.param_groups,rates): g['lr']=r*scale
                return update()
            ok,loss=guarded_update(model,opt,proposal,check)
            if ok: return dict(scale=scale,attempts=i,loss=loss)
        return dict(scale=None,attempts=len(scales),loss=loss)
    finally:
        for g,r in zip(opt.param_groups,rates): g['lr']=r
