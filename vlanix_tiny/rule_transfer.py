"""P0.11C: new OR rule with matched E policy and replay factorial."""
import copy
import hashlib
import json
from collections import Counter
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_curriculum import generate,audit,schedule,OPS
from vlanix_tiny.math_learning import bit_initialize,evaluate
from vlanix_tiny.embedding_conservation import train,distribution,tensor_hash,LR
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.rule_transfer_data import or_rows,adaptation_orders

def aggregate(m):
    n=sum(v['rows'] for v in m.values())
    return dict(rows=n,accuracy=sum(v['correct'] for v in m.values())/n,
        bit_accuracy=sum(v['bit_accuracy']*v['rows'] for v in m.values())/n,
        loss=sum(v['answer_loss']*v['rows'] for v in m.values())/n)

def run(seed):
    old=generate(seed); new=or_rows(old)
    assert len(new)==904
    assert len({r['id'] for r in old+new})==len(old)+len(new)
    old_val=[r for r in old if r['split']=='validation']
    new_val=[r for r in new if r['split']=='validation']
    assert len(old_val)==648 and len(new_val)==216
    majority={}
    for tier in ('easy','hard'):
        for op in (*OPS.values(),'or'):
            c=Counter(r['answer'] for r in old+new if r['split']=='train' and r['tier']==tier and r['family']==op)
            majority[tier,op]=min(c,key=lambda v:(-c[v],v))
    torch.manual_seed(seed)
    m=make_binary_model(); bit_initialize(m)
    opt=make_optimizer(m,LR)
    warm=train(m,opt,schedule(old,'curriculum',seed+1000,2))
    # Common longer old-rule foundation, with fixed E in this continuation.
    set_policy(opt,0.)
    continued=train(m,opt,schedule(old,'mixed',seed+2000,2))
    baseline_old=evaluate(m,old_val,majority)
    baseline_new=evaluate(m,new_val,majority,('or',))
    ref=distribution(m,old_val)
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone(); anchor=m.lm_head.weight.detach().clone()
    pure,replay=adaptation_orders(old,new,seed+3000)
    assert len(pure)==len(replay)==1920
    results={}
    for policy,scale in [('frozen',0.),('slow',.1),('full',1.)]:
        for mode,order in [('pure',pure),('replay',replay)]:
            m=make_binary_model(); m.load_state_dict(weights,strict=True)
            assert m.lm_head.weight is m.model.embed_tokens.weight
            assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
            opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
            torch.set_rng_state(rng.clone())
            set_policy(opt,scale)
            stats=train(m,opt,order)
            geo=geometry(anchor,m.lm_head.weight)
            assert geo['exact']==(scale==0.)
            old_after=evaluate(m,old_val,majority)
            new_after=evaluate(m,new_val,majority,('or',))
            before=aggregate(baseline_old); after=aggregate(old_after)
            nb=aggregate(baseline_new); na=aggregate(new_after)
            kl=F.kl_div(distribution(m,old_val),ref,log_target=True,reduction='batchmean').item()
            assert torch.isfinite(torch.tensor(kl))
            count=Counter(r['family'] for r in order)
            results[policy+'_'+mode]=dict(**stats,embedding_lr=scale*LR,
                new_targets=count['or'],old_targets=len(order)-count['or'],
                order_sha256=hashlib.sha256(''.join(r['id'] for r in order).encode()).hexdigest(),
                geometry=geo,old=old_after,new=new_after,
                old_accuracy_delta=after['accuracy']-before['accuracy'],
                old_loss_delta=after['loss']-before['loss'],
                new_accuracy_delta=na['accuracy']-nb['accuracy'],
                new_loss_delta=na['loss']-nb['loss'],old_kl=max(0.,kl))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,dataset=audit(old),
        new_rows=len(new),foundation=warm,continued=continued,
        anchor_sha256=tensor_hash(anchor),baseline_old=baseline_old,baseline_new=baseline_new,
        arms=results,test_rows_used=0)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-rule-transfer-v1',runs=[run(42),run(43)],
        claim='exploratory novel rule and rehearsal comparison; not mastery'),sort_keys=True))
