"""Matched full-table rehearsal and local classification margins."""
import torch
import torch.nn.functional as F


def rehearsal(qs,sample,full):
    if len(qs)!=32 or len(set(qs))!=32 or len(sample)!=8 or len(set(sample))!=8 or not set(sample)<=set(qs):
        raise ValueError('rehearsal frames')
    return list(qs) if full else list(sample)*4


def objective(z,y):
    if z.shape!=(56,10,265) or y.shape!=(56,10) or y.dtype!=torch.long:
        raise ValueError('batch')
    mask=torch.zeros_like(y,dtype=torch.bool); mask[:24,9]=True; mask[24:,4]=True
    if not torch.equal(y!=-100,mask) or ((y[:24,9]<9)|(y[:24,9]>264)).any() or ((y[24:,4]<9)|(y[24:,4]>12)).any():
        raise ValueError('target layout')
    return .75*F.cross_entropy(z[:24,9],y[:24,9])+.25*F.cross_entropy(z[24:,4],y[24:,4])


def margins(z,y):
    if z.ndim!=2 or y.shape!=(len(z),) or y.dtype!=torch.long or not len(z) or z.shape[1]<2 or not torch.isfinite(z).all() or ((y<0)|(y>=z.shape[1])).any():
        raise ValueError('logits or targets')
    chosen=z.gather(1,y[:,None]).squeeze(1)
    other=z.scatter(1,y[:,None],-torch.inf).max(1).values
    return chosen-other
