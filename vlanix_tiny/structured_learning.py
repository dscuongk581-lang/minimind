"""P0.11E: matched raw versus interleaved bit-plane prompt learning."""
import hashlib
import json
import time
from collections import Counter
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_curriculum import generate,audit,schedule,OPS
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.math_learning import bit_initialize,evaluate
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.structured_input import encode_prompt

FAMILIES=(*OPS.values(),'or')

def tensorize(rows,mode):
    if not rows: raise ValueError('empty batch')
    x=[encode_prompt(bytes.fromhex(r['query']),mode) for r in rows]
    y=[r['answer']+9 for r in rows]
    return torch.tensor(x),torch.tensor(y)

def run(seed):
    old=generate(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    train=[r for r in rows if r['split']=='train']
    assert len(val)==864 and len(train)==1920
    majority={}
    for tier in ('easy','hard'):
        for f in FAMILIES:
            c=Counter(r['answer'] for r in train if r['tier']==tier and r['family']==f)
            majority[tier,f]=min(c,key=lambda v:(-c[v],v))
    order=schedule(rows,'mixed',seed+5000,2)
    assert len(order)==3840
    order_hash=hashlib.sha256(''.join(r['id'] for r in order).encode()).hexdigest()
    arms={}; common_anchor=None
    for mode in ('raw','planes'):
        torch.manual_seed(seed)
        m=make_binary_model(); bit_initialize(m)
        assert m.model.embed_tokens.weight is m.lm_head.weight
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        anchor=m.lm_head.weight.detach().clone(); h=tensor_hash(anchor)
        if common_anchor is None: common_anchor=h
        assert h==common_anchor
        opt=make_optimizer(m,LR); set_policy(opt,0.)
        convert=lambda r:tensorize(r,mode)
        before=evaluate(m,val,majority,FAMILIES,tensorizer=convert)
        n=steps=tokens=0; loss_sum=0.
        start=time.monotonic(); m.train()
        for i in range(0,len(order),32):
            part=order[i:i+32]
            assert all(r['split']=='train' for r in part)
            x,y=convert(part)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(m(x,use_cache=False).logits[:,-1],y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            n+=len(part); steps+=1; tokens+=x.numel(); loss_sum+=loss.item()*len(part)
        elapsed=time.monotonic()-start
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[mode]=dict(steps=steps,supervised_answers=n,prompt_tokens=tokens,
            train_seconds=elapsed,train_ce=loss_sum/n,geometry=geo,
            before=before,after=evaluate(m,val,majority,FAMILIES,tensorizer=convert),
            anchor_sha256=h,order_sha256=order_hash)
    return dict(seed=seed,parameters=EXPECTED_PARAMS,dataset=audit(old),
        validation_rows=len(val),arms=arms,test_rows_used=0)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-structured-learning-v1',runs=[run(42),run(43)],
        claim='matched sample and step budget; different token and compute budgets'),sort_keys=True))
