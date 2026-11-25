"""Differentiable local-rule margin penalty; no new parameters."""
import math
import torch


def penalty(z,y,margin=1.):
    if not math.isfinite(margin) or margin<0:
        raise ValueError('margin')
    if z.ndim!=2 or not len(z) or z.shape[1]<2 or y.shape!=(len(z),) or y.dtype!=torch.long or not torch.isfinite(z).all() or ((y<0)|(y>=z.shape[1])).any():
        raise ValueError('logits or targets')
    chosen=z.gather(1,y[:,None]).squeeze(1)
    other=z.scatter(1,y[:,None],-torch.inf).max(1).values
    return torch.relu(margin-chosen+other).mean()
