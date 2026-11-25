"""P0.11D: matched byte CE versus byte CE plus bit marginal supervision."""
import copy
import hashlib
import json
from collections import Counter
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_curriculum import generate,schedule,OPS
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.math_learning import bit_initialize,tensorize,evaluate
from vlanix_tiny.embedding_conservation import train,tensor_hash,LR
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.bit_objective import objective,bit_loss

FAMILIES=(*OPS.values(),'or')

def fit(model,opt,order,weight):
    if not order or any(r['split']!='train' for r in order):
        raise ValueError('train only')
    model.train()
    n=0; ce_sum=bit_sum=0.; steps=0
    for i in range(0,len(order),32):
        chunk=order[i:i+32]; x,y=tensorize(chunk)
        opt.zero_grad(set_to_none=True)
        z=model(x,use_cache=False).logits[:,-1]
        with torch.no_grad():
            ce_sum+=F.cross_entropy(z,y).item()*len(chunk)
            bit_sum+=bit_loss(z,y).item()*len(chunk)
        loss=objective(z,y,weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step(); n+=len(chunk); steps+=1
    opt.zero_grad(set_to_none=True)
    return dict(steps=steps,supervised_answers=n,train_byte_ce=ce_sum/n,train_bit_nll=bit_sum/n)

def run(seed):
    old=generate(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    assert len(val)==864 and len({r['id'] for r in rows})==len(rows)
    majority={}
    for tier in ('easy','hard'):
        for f in FAMILIES:
            c=Counter(r['answer'] for r in rows if r['split']=='train' and r['tier']==tier and r['family']==f)
            majority[tier,f]=min(c,key=lambda v:(-c[v],v))
    torch.manual_seed(seed)
    m=make_binary_model(); bit_initialize(m)
    opt=make_optimizer(m,LR)
    warm=train(m,opt,schedule(old,'curriculum',seed+1000,2))
    anchor=m.lm_head.weight.detach().clone()
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state().clone()
    baseline=evaluate(m,val,majority,FAMILIES)
    order=schedule(rows,'mixed',seed+4000,2)
    assert len(order)==3840
    order_hash=hashlib.sha256(''.join(r['id'] for r in order).encode()).hexdigest()
    arms={}
    for name,w in [('byte_ce',0.),('byte_plus_bit',1.)]:
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert m.lm_head.weight is m.model.embed_tokens.weight
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        stats=fit(m,opt,order,w)
        g=geometry(anchor,m.lm_head.weight); assert g['exact']
        arms[name]=dict(**stats,bit_weight=w,geometry=g,after=evaluate(m,val,majority,FAMILIES),
            embedding_sha256=tensor_hash(m.lm_head.weight),order_sha256=order_hash)
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation=warm,baseline=baseline,
        arms=arms,validation_rows=len(val),test_rows_used=0)
if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-bit-learning-v1',runs=[run(42),run(43)],
        claim='exploratory matched objective comparison; no growth or GPU authorization'),sort_keys=True))
