"""P0.11I: weighted token CE with strict stream batch layout."""
import math
import torch
import torch.nn.functional as F


def weighted_loss(z,y,final_weight=1,local_weight=1):
    if z.shape!=(32,26,265) or y.shape!=(32,26) or y.dtype!=torch.long:
        raise ValueError('batch shape')
    if any(not math.isfinite(w) or w<=0 for w in (final_weight,local_weight)):
        raise ValueError('weight')
    state=torch.zeros_like(y,dtype=torch.bool)
    final=state.clone(); local=state.clone()
    state[:24,[3,6,9,12,15,18,21,24]]=True
    final[:24,25]=True; local[24:,4]=True
    if not torch.equal(y!=-100,state|final|local): raise ValueError('target layout')
    if ((y[state|local]<9)|(y[state|local]>12)).any() or ((y[final]<9)|(y[final]>264)).any():
        raise ValueError('target ID')
    if final_weight==local_weight==1:
        return F.cross_entropy(z.reshape(-1,265),y.reshape(-1),ignore_index=-100)
    weights=state.to(z.dtype)+final.to(z.dtype)*final_weight+local.to(z.dtype)*local_weight
    losses=F.cross_entropy(z.reshape(-1,265),y.reshape(-1),ignore_index=-100,reduction='none').reshape_as(y)
    return (weights*losses).sum()/weights.sum()


def decode_states(states):
    if len(states)!=8 or any(type(p) is not int or not 9<=p<=12 for p in states): return None
    return sum(((p-9)&1)<<k for k,p in enumerate(states))
