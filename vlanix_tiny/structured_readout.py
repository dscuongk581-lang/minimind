"""Auxiliary bit supervision restricted to the independent readout task."""
import math
import torch
import torch.nn.functional as F


def bit_masks(z):
    return ((torch.arange(256,device=z.device)[:,None]>>torch.arange(8,device=z.device))&1).bool()


def objective(z,y,weight):
    if z.shape!=(32,10,265) or y.shape!=(32,10) or y.dtype!=torch.long or not math.isfinite(weight) or weight<0:
        raise ValueError('batch or weight')
    mask=torch.zeros_like(y,dtype=torch.bool); mask[:24,9]=True; mask[24:,4]=True
    if not torch.equal(y!=-100,mask) or ((y[:24,9]<9)|(y[:24,9]>264)).any() or ((y[24:,4]<9)|(y[24:,4]>12)).any():
        raise ValueError('target layout')
    ce=F.cross_entropy(z.reshape(-1,265),y.reshape(-1))
    if weight==0: return ce
    v=z[:24,9,9:]; bits=bit_masks(v)
    target=(((y[:24,9,None]-9)>>torch.arange(8,device=z.device))&1).bool()
    correct=torch.logsumexp(v[:,:,None].masked_fill(bits[None,:,:]!=target[:,None,:],-torch.inf),dim=1)
    loss=(torch.logsumexp(v,dim=1,keepdim=True)-correct).mean()
    return ce+.75*weight*loss


@torch.no_grad()
def marginal_bytes(z):
    if z.ndim!=2 or z.shape[1]!=265 or not torch.isfinite(z).all(): raise ValueError('logits')
    v=z[:,9:]; bits=bit_masks(v)
    one=torch.logsumexp(v[:,:,None].masked_fill(~bits[None,:,:],-torch.inf),dim=1)
    zero=torch.logsumexp(v[:,:,None].masked_fill(bits[None,:,:],-torch.inf),dim=1)
    return ((one>zero).long()*(1<<torch.arange(8,device=z.device))).sum(1)
