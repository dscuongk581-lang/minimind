"""P0.9G: local, single-writer guarded answer-learning session."""
import hashlib
import json
import math
import tempfile
from pathlib import Path
import torch
from vlanix_tiny.binary_logic import generate,verify,FAMILIES
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_homeostasis import answer_step,reference,drift
from vlanix_tiny.homeostasis import PI

SCHEMA='ngc-guarded-session-v1'

def validate_plan(cal,plan):
    if not cal or not plan or any(not b for b in plan):
        raise ValueError('empty plan')
    rows=list(cal)+[r for b in plan for r in b]
    if any(not verify(r) or r['split']!='train' for r in rows):
        raise ValueError('invalid train/calibration row')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('overlap or duplicate')
    return hashlib.sha256(json.dumps([cal,plan],sort_keys=True,separators=(',',':')).encode()).hexdigest()

def valid_cap(cap):
    if type(cap) not in (int,float) or not math.isfinite(cap) or cap<=0:
        raise ValueError('hard KL cap')

class Session:
    def __init__(self,state,cal,plan,identity,cap=.075):
        valid_cap(cap)
        if not isinstance(identity,str) or not identity: raise ValueError('identity')
        self.plan_digest=validate_plan(cal,plan)
        self.cal,self.plan=json.loads(json.dumps([cal,plan]))
        self.state=state; self.identity=identity; self.cap=cap
        self.pi=PI(); self.cursor=0; self.start_steps=state.steps; self.blocked=False
        self.ref=reference(state.model,self.cal).detach().clone()

    def save(self,path):
        if validate_plan(self.cal,self.plan)!=self.plan_digest:
            raise ValueError('plan mutated')
        self.state.save(path,self.identity,runtime=dict(schema=SCHEMA,plan_digest=self.plan_digest,
            pi=self.pi.snapshot(),reference=self.ref,cap=self.cap,cursor=self.cursor,
            start_steps=self.start_steps,blocked=self.blocked))

    @classmethod
    def restore(cls,path,cal,plan,identity):
        digest=validate_plan(cal,plan)
        rng=torch.get_rng_state()
        try:
            state=State.restore(path,identity)
            d=state.runtime
            if not isinstance(d,dict) or d.get('schema')!=SCHEMA or d.get('plan_digest')!=digest:
                raise ValueError('session identity')
            valid_cap(d['cap'])
            p=PI.restore(d['pi'])
            cursor,start=d['cursor'],d['start_steps']
            if type(cursor) is not int or not 0<=cursor<=len(plan) or type(start) is not int or start<0 or state.steps!=start+cursor or type(d['blocked']) is not bool:
                raise ValueError('cursor')
            ref=d['reference']
            if not isinstance(ref,torch.Tensor) or ref.dtype!=torch.float32 or tuple(ref.shape)!=(len(cal),265) or not torch.isfinite(ref).all():
                raise ValueError('reference')
            if not torch.allclose(ref.logsumexp(-1),torch.zeros(len(cal)),atol=1e-5,rtol=0):
                raise ValueError('reference normalization')
            obj=cls.__new__(cls)
            obj.state=state; obj.cal,obj.plan=json.loads(json.dumps([cal,plan]))
            obj.identity=identity; obj.plan_digest=digest; obj.cap=d['cap']
            obj.pi=p; obj.cursor=cursor; obj.start_steps=start; obj.blocked=d['blocked']; obj.ref=ref
            return obj
        except Exception:
            torch.set_rng_state(rng)
            raise

    def attempt(self):
        if self.blocked: raise RuntimeError('session blocked; explicit new session required')
        if self.cursor>=len(self.plan): raise RuntimeError('plan complete')
        if validate_plan(self.cal,self.plan)!=self.plan_digest: raise ValueError('plan mutated')
        before=drift(self.state.model,self.cal,self.ref)
        if before>self.cap:
            self.blocked=True
            return dict(accepted=False,reason='PRE_CAP',before=before,proposed=before,cursor=self.cursor)
        # Snapshot before changing LR, PI, parameters, optimizer or RNG.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pre.pt'
            self.save(p)
            try:
                scale=self.pi.update(before)
                self.state.opt.param_groups[0]['lr']=.001*scale
                answer_step(self.state,self.plan[self.cursor])
                proposed=drift(self.state.model,self.cal,self.ref)
                if proposed<=self.cap:
                    self.cursor+=1
                    return dict(accepted=True,reason='WITHIN_CAP',before=before,proposed=proposed,cursor=self.cursor,scale=scale)
            except Exception:
                self.__dict__.update(Session.restore(p,self.cal,self.plan,self.identity).__dict__)
                self.blocked=True
                raise
            self.__dict__.update(Session.restore(p,self.cal,self.plan,self.identity).__dict__)
            self.blocked=True
            return dict(accepted=False,reason='POST_CAP_ROLLBACK',before=before,proposed=proposed,cursor=self.cursor,scale=scale)

def run(seed=42):
    torch.manual_seed(seed)
    rows=generate(16,seed)
    a=[r for r in rows if r['split']=='train' and r['family'] in FAMILIES[:3]]
    b=[r for r in rows if r['split']=='train' and r['family'] in FAMILIES[3:]]
    warm=[r for f in FAMILIES[:3] for r in [x for x in a if x['family']==f][:12]]
    cal=[r for f in FAMILIES[:3] for r in [x for x in a if x['family']==f][12:]]
    plan=[b[i:i+8] for i in range(0,len(b),8)]
    s=State()
    for i in range(0,len(warm),12): answer_step(s,warm[i:i+12])
    s.grow(2,32)
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    session=Session(s,cal,plan,identity)
    trace=[]
    while not session.blocked and session.cursor<len(plan):
        trace.append(session.attempt())
    committed=drift(session.state.model,cal,session.ref)
    assert committed<=session.cap
    return dict(seed=seed,accepted=session.cursor,attempted=len(trace),blocked=session.blocked,
        final_committed_kl=committed,cap=session.cap,trace=trace,growth_authorized=False)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-guard-pilot-v1',runs=[run(42),run(43)]),sort_keys=True))
