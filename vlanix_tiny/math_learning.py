"""P0.11A: matched CPU trials with curriculum and bit-structured initialization."""
import json
import math
from collections import Counter
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_curriculum import generate,audit,schedule,OPS

def tensorize(rows):
    q=[bytes.fromhex(r['query']) for r in rows]
    if not q or any(len(x)!=3 for x in q): raise ValueError('frame')
    return torch.tensor([[1]+[v+9 for v in x] for x in q]),torch.tensor([r['answer']+9 for r in rows])

@torch.no_grad()
def bit_initialize(model):
    # Preserve each original byte-row norm, assign half its energy to
    # eight bipolar bit coordinates, half to the original random tail.
    w=model.model.embed_tokens.weight
    assert w is model.lm_head.weight and tuple(w.shape)==(265,256)
    rows=w[9:]
    norm=rows.norm(dim=1,keepdim=True)
    values=torch.arange(256,device=w.device)
    bits=((values[:,None]>>torch.arange(8,device=w.device))&1).to(w.dtype)*2-1
    tail=F.normalize(rows[:,8:].clone(),dim=1)
    rows[:,:8]=bits*norm/math.sqrt(16)
    rows[:,8:]=tail*norm/math.sqrt(2)

@torch.no_grad()
def evaluate(model,rows,majority):
    model.eval()
    results={}
    for tier in ('easy','hard','ood'):
        for op in OPS.values():
            part=[r for r in rows if r['tier']==tier and r['family']==op]
            if not part: continue
            loss=correct=bit_correct=invalid=0
            for i in range(0,len(part),64):
                x,y=tensorize(part[i:i+64])
                logits=model(x,use_cache=False).logits[:,-1]
                loss+=F.cross_entropy(logits,y,reduction='sum').item()
                p=logits.argmax(-1)
                valid=(p>=9)&(p<=264)
                correct+=int((p==y).sum()); invalid+=int((~valid).sum())
                xor=(p-9)^(y-9)
                for k in range(8): bit_correct+=int((valid & (((xor>>k)&1)==0)).sum())
            n=len(part)
            base=sum(r['answer']==majority.get((tier,op),majority.get(('hard',op),-1)) for r in part)
            results[tier+'/'+op]=dict(rows=n,correct=correct,accuracy=correct/n,
                bit_accuracy=bit_correct/(8*n),invalid=invalid,answer_loss=loss/n,majority_accuracy=base/n)
    return results

def run(seed=42,epochs=2):
    rows=generate(seed); val=[r for r in rows if r['split']=='validation']
    train=[r for r in rows if r['split']=='train']
    majority={}
    for tier in ('easy','hard'):
        for op in OPS.values():
            c=Counter(r['answer'] for r in train if r['tier']==tier and r['family']==op)
            majority[tier,op]=min(c,key=lambda v:(-c[v],v))
    results={}
    for arm,mode,structured in [('raw_mixed','mixed',False),('raw_curriculum','curriculum',False),
                                 ('bit_curriculum','curriculum',True)]:
        torch.manual_seed(seed)
        m=make_binary_model()
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        if structured: bit_initialize(m)
        opt=torch.optim.AdamW(m.parameters(),lr=.0005)
        order=schedule(rows,mode,seed+1000,epochs)
        before=evaluate(m,val,majority)
        loss_sum=0.; steps=0
        for i in range(0,len(order),32):
            part=order[i:i+32]
            assert all(r['split']=='train' for r in part)
            x,y=tensorize(part); m.train(); opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(m(x,use_cache=False).logits[:,-1],y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step(); steps+=1; loss_sum+=loss.item()*len(part)
        results[arm]=dict(before=before,after=evaluate(m,val,majority),
            steps=steps,supervised_answers=len(order),prompt_tokens=4*len(order),
            train_loss=loss_sum/len(order),tied=m.model.embed_tokens.weight is m.lm_head.weight)
    return dict(seed=seed,dataset=audit(rows),arms=results,test_rows_used=0,parameters=EXPECTED_PARAMS)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-math-curriculum-v1',runs=[run(42),run(43)],
        claim='CPU pilot; no growth or GPU authorization'),sort_keys=True))
