"""Explicit AdamW task-boundary state policy; model weights are untouched."""
import torch


def transition_optimizer(opt,mode):
    if not isinstance(opt,torch.optim.AdamW) or mode not in ('retain','reset'):
        raise ValueError('AdamW transition policy')
    before=len(opt.state)
    if mode=='reset':
        opt.state.clear()
    return dict(before_slots=before,after_slots=len(opt.state))
