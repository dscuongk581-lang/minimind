"""P0.11T: task-boundary AdamW state reset versus retention."""
import copy
import hashlib
import json
import torch
import time
import argparse
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target
from vlanix_tiny.local_rule_learning import inputs
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.math_curriculum import generate as corpus,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import batches
from vlanix_tiny.protected_core import pack_prompt
from vlanix_tiny.protected_learning import evaluate
from vlanix_tiny.adaptive_guard import backtrack
from vlanix_tiny.complete_rehearsal import rehearsal,objective,margins
from vlanix_tiny.margin_learning import penalty
from vlanix_tiny.learning_state import transition_optimizer
from vlanix_tiny.readout_balance import states,plan,probes

def batch(rows,local):
    if len(rows)!=24 or len(local)!=32: raise ValueError('batch')
    x=[]; y=[]
    for b,c in rows:
        x.append(pack_prompt(states(b,c)))
        y.append([-100]*9+[b+9])
    for q in local:
        x.append([1]+[v+9 for v in q]+[0]*5)
        y.append([-100]*4+[target(q)+9]+[-100]*5)
    return torch.tensor(x),torch.tensor(y)

@torch.no_grad()
def local_stats(m,qs):
    m.eval()
    z=m(inputs(qs),use_cache=False).logits[:,-1]
    y=torch.tensor([target(q)+9 for q in qs])
    v=margins(z,y)
    return dict(correct=int((z.argmax(-1)==y).sum()),minimum=float(v.min()),mean=float(v.mean()))

def fit_full(m,opt,work,qs,weight):
    start=time.monotonic(); accepted=attempts=0; scales={}
    initial=local_stats(m,qs); trace=[dict(step=0,**initial)]
    assert initial['correct']==32
    min_margin=initial['minimum']
    for i,(rows,sample) in enumerate(work,1):
        x,y=batch(rows,rehearsal(qs,sample,True))
        def update():
            m.train(); opt.zero_grad(set_to_none=True)
            z=m(x,use_cache=False).logits
            loss=objective(z,y)
            if weight: loss=loss+weight*penalty(z[24:,4],y[24:,4],1.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            return loss.item()
        result=backtrack(m,opt,update,lambda:local_correct(m,qs)==32)
        scale=result['scale']; accepted+=int(scale is not None); attempts+=result['attempts']
        key=str(scale) if scale is not None else 'rejected'
        scales[key]=scales.get(key,0)+1
        stats=local_stats(m,qs); assert stats['correct']==32
        min_margin=min(min_margin,stats['minimum'])
        if i%20==0 or i==len(work): trace.append(dict(step=i,**stats,accepted=accepted,attempts=attempts))
    return dict(proposed_steps=len(work),accepted_steps=accepted,rejected_steps=len(work)-accepted,
        optimizer_attempts=attempts,target_tokens=56*len(work),attempt_target_tokens=56*attempts,
        input_tokens=560*attempts,scale_counts=scales,min_local_correct=32,
        min_committed_margin=min_margin,margin_trace=trace,train_seconds=time.monotonic()-start)

@torch.no_grad()
def probe(m,rows):
    m.eval(); correct=invalid=0; per_byte={}
    for i in range(0,len(rows),64):
        part=rows[i:i+64]
        x=torch.tensor([pack_prompt(states(b,c)) for b,c in part])
        out=m(x,use_cache=False).logits[:,-1].argmax(-1).tolist()
        for (b,c),v in zip(part,out):
            hit=int(v==b+9); correct+=hit; invalid+=int(not 9<=v<=264)
            n,k=per_byte.get(b,(0,0)); per_byte[b]=(n+1,k+hit)
    return dict(rows=len(rows),correct=correct,invalid=invalid,
        all_masks_correct_bytes=sum(n==k for n,k in per_byte.values()))

def run(seed):
    torch.manual_seed(seed)
    old=corpus(seed); val=[r for r in old+or_rows(old) if r['split']=='validation']
    train=plan(seed+10000); held=probes(seed+10000)
    assert len(val)==864 and len(train)==3840 and len(held)==4096
    assert not set(train)&set(held)
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    anchor=m.lm_head.weight.detach().clone(); opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    foundation=local_stats(m,qs)
    if foundation['correct']!=32:
        return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_steps=120,
            foundation=foundation,status='foundation_failed',arms={})
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    work=batches(train,qs,seed+11000)
    digest=hashlib.sha256(json.dumps(work,separators=(',',':')).encode()).hexdigest()
    arms={}
    for name in ('retain','reset'):
        weight=.25
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        assert m.lm_head.weight is m.model.embed_tokens.weight
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        transition=transition_optimizer(opt,name)
        assert all(torch.equal(v,weights[k]) for k,v in m.state_dict().items())
        assert local_correct(m,qs)==32
        stats=fit_full(m,opt,work,qs,weight)
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[name]=dict(**stats,optimizer_transition=transition,unique_local_per_batch=32,margin_weight=weight,margin_target=1.,geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
            schedule_sha256=digest,local_correct=local_correct(m,qs),
            canonical_probe=probe(m,[(b,0) for b in range(256)]),
            unseen_carry_probe=probe(m,held),validation=evaluate(m,val,qs))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_steps=120,
        status='completed',foundation=foundation,train_examples=len(train),distinct_train_frames=3840,
        probe_frames=len(held),probe_digest=hashlib.sha256(json.dumps(held).encode()).hexdigest(),
        dataset=audit(old),test_rows_used=0,arms=arms)

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=(42,43,44,45))
    args=p.parse_args()
    print(json.dumps(dict(schema='ngc-learning-state-v1',runs=[run(args.seed)],
        torch_version=torch.__version__,deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='same local foundation and margin objective; retain vs reset AdamW slots; exact postcheck'),sort_keys=True))
