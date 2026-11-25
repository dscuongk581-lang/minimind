"""CPU-only guarded optimizer proposal and independent readout grammar."""
import copy
import torch


def pack_prompt(states):
    if len(states)!=8 or any(type(p) is not int or not 9<=p<=12 for p in states): return None
    return [1,18]+list(states)


def guarded_update(model,opt,update,check):
    # Callbacks must be synchronous and check must be read-only.
    weights=copy.deepcopy(model.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone(); training=model.training
    def restore():
        model.load_state_dict(weights,strict=True); opt.load_state_dict(moments)
        torch.set_rng_state(rng); model.train(training)
        opt.zero_grad(set_to_none=True)
    try:
        loss=update()
        with torch.no_grad(): accepted=bool(check())
        if not accepted: restore()
        else:
            model.train(training); opt.zero_grad(set_to_none=True)
        return accepted,loss
    except Exception:
        restore(); raise
