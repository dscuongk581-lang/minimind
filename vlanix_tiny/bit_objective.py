"""Auxiliary bit marginal NLL, with byte CE retaining control-ID penalty."""
import math
import torch
import torch.nn.functional as F

def bit_loss(z,y):
    if z.ndim!=2 or z.shape[1]!=265 or not len(z) or y.shape!=(len(z),) or y.dtype!=torch.long or ((y<9)|(y>264)).any():
        raise ValueError('byte targets required')
    v=z[:,9:]
    bits=(torch.arange(256,device=z.device)[:,None]>>torch.arange(8,device=z.device))&1
    target=((y[:,None]-9)>>torch.arange(8,device=z.device))&1
    mask=bits[None,:,:]==target[:,None,:]
    correct=torch.logsumexp(v[:,:,None].masked_fill(~mask,-torch.inf),dim=1)
    total=torch.logsumexp(v,dim=1,keepdim=True)
    return (total-correct).mean()

def objective(z,y,weight):
    if not math.isfinite(weight) or weight<0: raise ValueError('bit weight')
    if weight==0.: return F.cross_entropy(z,y)
    return F.cross_entropy(z,y)+weight*bit_loss(z,y)
