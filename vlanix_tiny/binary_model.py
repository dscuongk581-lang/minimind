"""P0.9D: machine-first CPU training smoke; independent BPE baseline."""
import json
from collections import Counter
import torch
from model.model_minimind import MiniMindForCausalLM
from vlanix_tiny.config import make_config
from vlanix_tiny.binary_logic import encode, generate, verify, FAMILIES

EXPECTED_PARAMS=5_970_176

def make_binary_model():
    cfg=make_config()
    cfg.vocab_size=265
    cfg.intermediate_size=1024
    return MiniMindForCausalLM(cfg)

def batch(rows):
    if not rows: raise ValueError('empty batch')
    pairs=[encode(r) for r in rows]
    length=max(len(ids) for ids,_ in pairs)
    return (torch.tensor([ids+[0]*(length-len(ids)) for ids,_ in pairs]),
            torch.tensor([labels+[-100]*(length-len(labels)) for _,labels in pairs]))

def prompt_batch(rows):
    if not rows: raise ValueError('empty batch')
    # No answer or EOS in input; verify the record before constructing prompt.
    if not all(verify(r) for r in rows): raise ValueError('invalid rows')
    seqs=[[1]+[b+9 for b in bytes.fromhex(r['query'])] for r in rows]
    lengths=torch.tensor([len(s) for s in seqs])
    x=torch.tensor([s+[0]*(int(lengths.max())-len(s)) for s in seqs])
    return x,lengths

def predict(model,rows):
    x,lengths=prompt_batch(rows)
    logits=model(x,use_cache=False).logits[torch.arange(len(rows)),lengths-1]
    # Unconstrained argmax, special-token predictions count as wrong.
    ids=logits.argmax(-1)
    return ids

@torch.no_grad()
def evaluate(model,rows,batch_size=32):
    if not rows or batch_size<=0: raise ValueError('evaluation inputs')
    training=model.training
    model.eval()
    counts=Counter(); hits=Counter(); invalid=0; total_loss=0.0
    try:
        for start in range(0,len(rows),batch_size):
            chunk=rows[start:start+batch_size]
            x,y=batch(chunk)
            loss=model(x,labels=y,use_cache=False).loss
            total_loss+=loss.item()*len(chunk)
            predicted=predict(model,chunk).tolist()
            for row,p in zip(chunk,predicted):
                f=row['family']; counts[f]+=1
                hits[f]+=int(p==row['answer']+9)
                invalid+=int(not 9<=p<=264)
    finally:
        model.train(training)
    return dict(rows=len(rows),loss=total_loss/len(rows),
        answer_accuracy=sum(hits.values())/len(rows),invalid_answers=invalid,
        by_family={f:dict(rows=n,correct=hits[f],accuracy=hits[f]/n) for f,n in counts.items()})

def majority_baseline(train,val):
    # Ties broken by lowest byte; no validation labels used to choose predictions.
    table={}
    for f in FAMILIES:
        c=Counter(r['answer'] for r in train if r['family']==f)
        if c: table[f]=min(c,key=lambda y:(-c[y],y))
    return sum(r['answer']==table.get(r['family'],-1) for r in val)/len(val)

def train_step(model,opt,rows):
    if not rows or any(r['split']!='train' for r in rows): raise ValueError('train only')
    model.train()
    x,y=batch(rows)
    opt.zero_grad(set_to_none=True)
    loss=model(x,labels=y,use_cache=False).loss
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    opt.step()
    return dict(loss=loss.item(),grad_norm=norm.item(),targets=int((y[:,1:]!=-100).sum()))

def run_smoke(seed=42,per_family=128,batch_size=16):
    if batch_size<=0: raise ValueError('batch size')
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed)
    rows=generate(per_family,seed)
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='validation']
    model=make_binary_model()
    n=sum(p.numel() for p in model.parameters())
    assert n==EXPECTED_PARAMS and model.model.embed_tokens.weight is model.lm_head.weight
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    before=evaluate(model,val)
    order=torch.randperm(len(train),generator=rng).tolist()
    steps=[train_step(model,opt,[train[i] for i in order[s:s+batch_size]]) for s in range(0,len(train),batch_size)]
    after=evaluate(model,val)
    return dict(schema='ngc-binary-train-smoke-v1',seed=seed,parameters=n,vocab=265,
        train_rows=len(train),validation_rows=len(val),test_rows_used=0,
        steps=len(steps),target_tokens=sum(x['targets'] for x in steps),
        train_loss=sum(x['loss']*x['targets'] for x in steps)/sum(x['targets'] for x in steps),
        gradient_max=max(x['grad_norm'] for x in steps),before=before,after=after,
        train_majority_accuracy=majority_baseline(train,val),tied=model.model.embed_tokens.weight is model.lm_head.weight,
        claim='engineering smoke; no demonstrated reasoning or GPU readiness')

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(run_smoke(),sort_keys=True))
