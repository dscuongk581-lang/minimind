"""P0.11B: paired CPU continuation from a common learned foundation."""
import copy
import hashlib
import json
from collections import Counter
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_curriculum import generate,audit,schedule,OPS
from vlanix_tiny.math_learning import tensorize,evaluate,bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry

LR=.0005
def train(model,opt,order):
    if not order or any(r['split']!='train' for r in order):
        raise ValueError('train only')
    model.train()
    total=0.; steps=0
    for i in range(0,len(order),32):
        part=order[i:i+32]; x,y=tensorize(part)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x,use_cache=False).logits[:,-1],y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        total+=loss.item()*len(part); steps+=1
    opt.zero_grad(set_to_none=True)
    return dict(steps=steps,supervised_answers=len(order),loss=total/len(order))

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@torch.no_grad()
def distribution(model,rows):
    model.eval()
    out=[]
    for i in range(0,len(rows),64):
        x,_=tensorize(rows[i:i+64])
        out.append(F.log_softmax(model(x,use_cache=False).logits[:,-1],dim=-1))
    return torch.cat(out)

def run(seed):
    rows=generate(seed); val=[r for r in rows if r['split']=='validation']
    train_rows=[r for r in rows if r['split']=='train']
    majority={}
    for tier in ('easy','hard'):
        for op in OPS.values():
            c=Counter(r['answer'] for r in train_rows if r['tier']==tier and r['family']==op)
            majority[tier,op]=min(c,key=lambda v:(-c[v],v))
    torch.manual_seed(seed)
    m=make_binary_model(); bit_initialize(m)
    init=m.lm_head.weight.detach().clone()
    opt=make_optimizer(m,LR)
    warm=schedule(rows,'curriculum',seed+1000,2)
    foundation=train(m,opt,warm)
    baseline=evaluate(m,val,majority)
    ref=distribution(m,val)
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    anchor=m.lm_head.weight.detach().clone()
    anchor_hash=tensor_hash(anchor)
    # Same further examples in the same mixed order in all three arms.
    order=schedule(rows,'mixed',seed+2000,2)
    order_hash=hashlib.sha256(''.join(r['id'] for r in order).encode()).hexdigest()
    results={}
    for arm,scale in [('frozen',0.),('slow',.1),('full',1.)]:
        m=make_binary_model()
        m.load_state_dict(weights,strict=True)
        assert m.lm_head.weight is m.model.embed_tokens.weight
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone())
        assert tensor_hash(m.lm_head.weight)==anchor_hash
        set_policy(opt,scale)
        stats=train(m,opt,order)
        geo=geometry(anchor,m.lm_head.weight)
        assert geo['exact']==(scale==0.)
        actual=distribution(m,val)
        kl=F.kl_div(actual,ref,log_target=True,reduction='batchmean').item()
        assert torch.isfinite(torch.tensor(kl))
        results[arm]=dict(**stats,embedding_lr=LR*scale,core_lr=LR,
            geometry=geo,after=evaluate(m,val,majority),validation_kl_from_foundation=max(0.,kl),
            anchor_sha256=anchor_hash,end_embedding_sha256=tensor_hash(m.lm_head.weight),
            order_sha256=order_hash,tied=m.model.embed_tokens.weight is m.lm_head.weight)
    return dict(seed=seed,dataset=audit(rows),parameters=EXPECTED_PARAMS,
        foundation=foundation,foundation_geometry=geometry(init,anchor),
        baseline=baseline,arms=results,test_rows_used=0)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-embedding-conservation-v1',runs=[run(42),run(43)],
        claim='matched continued training; no novel-rule transfer or growth authorization'),sort_keys=True))
