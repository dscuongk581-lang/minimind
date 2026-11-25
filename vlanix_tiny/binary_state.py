"""Trusted local CPU checkpoint reference for a growing binary model."""
import os
import tempfile
from pathlib import Path
import torch
from vlanix_tiny.binary_model import make_binary_model, EXPECTED_PARAMS
from vlanix_tiny.organic import add_microblock, count_params, CEILING

SCHEMA='ngc-binary-state-v1'

class State:
    def __init__(self):
        self.model=make_binary_model()
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.001)
        self.topology=[]
        self.steps=0

    def grow(self,layer,width=32):
        self.opt.zero_grad(set_to_none=True)
        add_microblock(self.model,layer,width,self.opt)
        self.topology.append([layer,width])

    def step(self,rows):
        from vlanix_tiny.binary_model import train_step
        result=train_step(self.model,self.opt,rows)
        self.steps+=1
        self.opt.zero_grad(set_to_none=True)
        return result

    def save(self,path,identity,runtime=None):
        if not isinstance(identity,str) or not identity: raise ValueError('identity')
        if any(p.grad is not None for p in self.model.parameters()): raise ValueError('step boundary')
        path=Path(path)
        payload=dict(schema=SCHEMA,identity=identity,topology=self.topology,
            steps=self.steps,model=self.model.state_dict(),optimizer=self.opt.state_dict(),
            rng=torch.get_rng_state(),params=count_params(self.model),runtime=runtime)
        fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:
                torch.save(payload,f)
                f.flush(); os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name): os.unlink(name)

    @classmethod
    def restore(cls,path,identity):
        # Trusted local artifacts only; weights_only is not authentication.
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d.get('schema')!=SCHEMA or d.get('identity')!=identity or not identity:
            raise ValueError('checkpoint identity')
        top=d['topology']
        if not isinstance(top,list) or any(not isinstance(x,list) or len(x)!=2 or
            type(x[0]) is not int or not 0<=x[0]<6 or type(x[1]) is not int or x[1]<=0 for x in top):
            raise ValueError('topology')
        n=EXPECTED_PARAMS+768*sum(x[1] for x in top)
        if n>CEILING or n!=d['params'] or type(d['steps']) is not int or d['steps']<0:
            raise ValueError('checkpoint bounds')
        # Constructor and branch initialization consume RNG; restore only on success.
        original=torch.get_rng_state()
        try:
            state=cls()
            for layer,width in top: state.grow(layer,width)
            state.model.load_state_dict(d['model'],strict=True)
            state.opt.load_state_dict(d['optimizer'])
            state.steps=d['steps']
            state.runtime=d.get('runtime')
            if count_params(state.model)!=n or state.model.model.embed_tokens.weight is not state.model.lm_head.weight:
                raise ValueError('state invariants')
            torch.set_rng_state(d['rng'])
            return state
        except Exception:
            torch.set_rng_state(original)
            raise
