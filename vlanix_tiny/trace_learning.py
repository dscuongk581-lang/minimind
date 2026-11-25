"""P0.11G: model-generated transition trace vs direct byte, fixed parameter budget."""
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
from vlanix_tiny.local_rule_learning import inputs,predict_table
from vlanix_tiny.math_curriculum import generate,schedule,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.trace_protocol import prompt,targets,rollout,assess

def step(model,opt,x,y,start):
    model.train(); opt.zero_grad(set_to_none=True)
    z=model(x,use_cache=False).logits[:,start:start+y.shape[1]]
    loss=F.cross_entropy(z.reshape(-1,265),y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return loss.item()

def local_correct(model,qs):
    t=predict_table(model,qs)
    return sum(t[q]==target(q) for q in qs)

def train_batch(model,opt,rows,mode):
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('train only')
    qs=[bytes.fromhex(r['query']) for r in rows]
    p=[prompt(q) for q in qs]; answers=[targets(q) for q in qs]
    assert all(a[-1]==r['answer']+9 for a,r in zip(answers,rows))
    if mode=='direct': answers=[a[-1:] for a in answers]
    elif mode!='trace': raise ValueError('mode')
    x=torch.tensor([a+b[:-1] for a,b in zip(p,answers)])
    y=torch.tensor(answers)
    return step(model,opt,x,y,len(p[0])-1),x.numel(),y.numel()

@torch.no_grad()
def evaluate(model,rows,mode):
    model.eval(); report={}
    if mode not in ('direct','trace'): raise ValueError('mode')
    start=time.monotonic()
    def predict(batch):
        return model(torch.tensor(batch),use_cache=False).logits[:,-1].argmax(-1).tolist()
    for i in range(0,len(rows),32):
        part=rows[i:i+32]; qs=[bytes.fromhex(r['query']) for r in part]
        outputs=rollout([prompt(q) for q in qs],predict,9 if mode=='trace' else 1)
        for r,q,out in zip(part,qs,outputs):
            key=r['tier']+'/'+r['family']
            d=report.setdefault(key,dict(rows=0,answer_correct=0,invalid=0,
                trace_correct=0,consistent=0,joint_correct=0))
            d['rows']+=1
            d['answer_correct']+=int(out[-1]==r['answer']+9)
            if mode=='trace':
                a=assess(q,out)
                d['invalid']+=int(not a['valid'])
                d['trace_correct']+=int(a['trace_exact'])
                d['consistent']+=int(a['consistent'])
                d['joint_correct']+=int(a['trace_exact'] and a['answer_exact'])
            else: d['invalid']+=int(not 9<=out[-1]<=264)
    if mode=='direct':
        for d in report.values():
            for key in ('trace_correct','consistent','joint_correct'): del d[key]
    return dict(by_group=report,seconds=time.monotonic()-start)

def run(seed):
    torch.manual_seed(seed)
    old=generate(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    order=schedule(rows,'mixed',seed+6000,2)
    assert len(val)==864 and len(order)==3840
    assert len({r['id'] for r in rows})==len(rows)
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    anchor=m.lm_head.weight.detach().clone(); opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    base_correct=local_correct(m,qs)
    assert base_correct==32, 'local foundation not mastered'
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    order_hash=hashlib.sha256(''.join(r['id'] for r in order).encode()).hexdigest()
    arms={}
    for mode in ('direct','trace'):
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert m.lm_head.weight is m.model.embed_tokens.weight
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        n_tokens=n_targets=steps=0; loss_sum=0.
        start=time.monotonic()
        for i in range(0,len(order),32):
            loss,nt,ny=train_batch(m,opt,order[i:i+32],mode)
            n_tokens+=nt; n_targets+=ny; steps+=1; loss_sum+=loss*ny
        elapsed=time.monotonic()-start
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[mode]=dict(steps=steps,examples=len(order),input_tokens=n_tokens,
            target_tokens=n_targets,train_ce=loss_sum/n_targets,train_seconds=elapsed,
            geometry=geo,anchor_sha256=tensor_hash(m.lm_head.weight),order_sha256=order_hash,
            local_correct=local_correct(m,qs),validation=evaluate(m,val,mode))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_local_correct=base_correct,
        foundation_steps=120,dataset=audit(old),validation_rows=len(val),test_rows_used=0,arms=arms)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-trace-learning-v1',runs=[run(42),run(43)],
        claim='fixed-length supervised trace, greedy free rollout; no autonomous planning claim'),sort_keys=True))
