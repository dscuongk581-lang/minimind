"""Matched supervision and rehearsal; causal interleaving vs front-loaded trace."""
import copy
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target
from vlanix_tiny.local_rule_learning import inputs
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.trace_protocol import targets,assess
from vlanix_tiny.math_curriculum import generate as corpus,schedule,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import encode,generate,batches

def fit(m,opt,work,mode,loss_fn=None):
    loss_sum=0.; n=0; start=time.monotonic()
    m.train()
    for rows,local in work:
        if any(r['split']!='train' for r in rows): raise ValueError('train only')
        x=[]; y=[]
        for r in rows:
            q=bytes.fromhex(r['query']); t=targets(q)
            assert t[-1]==r['answer']+9
            a,b=encode(q,t,mode); x.append(a); y.append(b)
        for q in local:
            x.append([1]+[v+9 for v in q]+[0]*21)
            y.append([-100]*4+[target(q)+9]+[-100]*21)
        a,b=torch.tensor(x),torch.tensor(y)
        assert a.shape==b.shape==(32,26)
        count=int((b!=-100).sum()); assert count==224
        opt.zero_grad(set_to_none=True)
        logits=m(a,use_cache=False).logits
        loss=(F.cross_entropy(logits.reshape(-1,265),b.reshape(-1),ignore_index=-100)
            if loss_fn is None else loss_fn(logits,b))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step(); loss_sum+=loss.item()*count; n+=count
    opt.zero_grad(set_to_none=True)
    return dict(steps=len(work),trace_examples=24*len(work),local_examples=8*len(work),
        padded_input_tokens=32*26*len(work),target_tokens=n,
        train_ce=loss_sum/n,train_seconds=time.monotonic()-start)

@torch.no_grad()
def evaluate(m,rows,mode):
    m.eval(); groups={}; start=time.monotonic()
    def predict(p):
        return m(torch.tensor(p),use_cache=False).logits[:,-1].argmax(-1).tolist()
    for i in range(0,len(rows),32):
        part=rows[i:i+32]; qs=[bytes.fromhex(r['query']) for r in part]
        outputs=generate(qs,predict,mode)
        for r,q,out in zip(part,qs,outputs):
            a=assess(q,out)
            g=groups.setdefault(r['tier']+'/'+r['family'],dict(rows=0,answer_correct=0,
                trace_correct=0,joint_correct=0,consistent=0,invalid=0))
            g['rows']+=1; g['answer_correct']+=int(a['answer_exact'])
            g['trace_correct']+=int(a['trace_exact'])
            g['joint_correct']+=int(a['trace_exact'] and a['answer_exact'])
            g['consistent']+=int(a['consistent']); g['invalid']+=int(not a['valid'])
    return dict(by_group=groups,seconds=time.monotonic()-start)

def run(seed):
    torch.manual_seed(seed)
    old=corpus(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    order=schedule(rows,'mixed',seed+7000,2)
    assert len(order)==3840 and len(val)==864
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    anchor=m.lm_head.weight.detach().clone()
    opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    assert local_correct(m,qs)==32
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    work=batches(order,qs,seed+8000)
    digest=hashlib.sha256(json.dumps([([r['id'] for r in a],b) for a,b in work],
        separators=(',',':')).encode()).hexdigest()
    arms={}
    for mode in ('front','stream'):
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        assert m.lm_head.weight is m.model.embed_tokens.weight
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        stats=fit(m,opt,work,mode)
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[mode]=dict(**stats,geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
            schedule_sha256=digest,local_correct=local_correct(m,qs),
            validation=evaluate(m,val,mode))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_steps=120,
        foundation_local_correct=32,dataset=audit(old),validation_rows=len(val),
        test_rows_used=0,arms=arms)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-stream-learning-v1',runs=[run(42),run(43)],
        claim='matched trace layouts with rehearsal; fixed external stream order'),sort_keys=True))
