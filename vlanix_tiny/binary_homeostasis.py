"""P0.9F: matched local PI pilot with answer-only learning and held-out metrics."""
import hashlib
import json
import math
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_logic import generate,FAMILIES
from vlanix_tiny.binary_model import batch,prompt_batch
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_growth import metric
from vlanix_tiny.organic import observe,count_params
from vlanix_tiny.homeostasis import PI

def scores(model,rows):
    x,l=prompt_batch(rows)
    return model(x,use_cache=False).logits[torch.arange(len(rows)),l-1]

def answer_step(state,rows):
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('train only')
    state.model.train()
    state.opt.zero_grad(set_to_none=True)
    y=torch.tensor([r['answer']+9 for r in rows])
    loss=F.cross_entropy(scores(state.model,rows),y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(state.model.parameters(),1.,error_if_nonfinite=True)
    state.opt.step()
    state.opt.zero_grad(set_to_none=True)
    state.steps+=1

@torch.no_grad()
def reference(model,rows):
    model.eval()
    return F.log_softmax(scores(model,rows),dim=-1)

@torch.no_grad()
def drift(model,rows,ref):
    actual=reference(model,rows)
    # KL(old || current); all 265 IDs, including controls.
    d=F.kl_div(actual,ref,reduction='batchmean',log_target=True).item()
    if not math.isfinite(d): raise ValueError('nonfinite KL')
    return max(0.,d)

def run(seed=42):
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed)
    rows=generate(64,seed)
    old=FAMILIES[:3]; new=FAMILIES[3:]
    def select(split,f): return [r for r in rows if r['split']==split and r['family']==f]
    warm=[]; cal=[]; learn=[]
    for f in old:
        a=select('train',f)
        warm+=a[:48]; cal+=a[48:]
    for f in new: learn+=select('train',f)
    ov=[r for f in old for r in select('validation',f)]
    nv=[r for f in new for r in select('validation',f)]
    assert len({r['id'] for r in warm+cal+learn+ov+nv})==len(warm+cal+learn+ov+nv)
    def shuffled(seq):
        o=torch.randperm(len(seq),generator=rng).tolist()
        return [[seq[i] for i in o[k:k+16]] for k in range(0,len(o),16)]
    warm_batches=shuffled(warm); adapt=shuffled(learn)
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    s=State()
    for x in warm_batches: answer_step(s,x)
    ref=reference(s.model,cal)
    base={'old':metric(s.model,ov),'new':metric(s.model,nv)}
    groups={}
    for f in new:
        groups[f]=[]
        for row in select('train',f)[:4]:
            x,y=batch([row]); y[y==2]=-100
            groups[f].append((x,y))
    sensor=observe(s.model,groups,seed)
    layer=max(sensor,key=lambda x:x['score'])['layer']
    results={}
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'s.pt'; s.save(p,identity)
        for name in ('fixed','growth','growth_pi'):
            t=State.restore(p,identity)
            if name!='fixed': t.grow(layer,32)
            controller=PI()
            trace=[]
            for i,chunk in enumerate(adapt):
                d=drift(t.model,cal,ref)
                scale=controller.update(d) if name=='growth_pi' else 1.
                t.opt.param_groups[0]['lr']=.001*scale
                answer_step(t,chunk)
                after=drift(t.model,cal,ref)
                trace.append(dict(step=i+1,drift_before=d,drift_after=after,scale=scale))
            results[name]=dict(params=count_params(t.model),old=metric(t.model,ov),
                new=metric(t.model,nv),steps=len(adapt),supervised_answers=len(learn),
                final_kl=trace[-1]['drift_after'],peak_kl=max(x['drift_after'] for x in trace),
                trace=trace)
    return dict(seed=seed,layer=layer,baseline=base,arms=results,calibration_rows=len(cal),
        warm_rows=len(warm),adapt_rows=len(learn),test_rows_used=0,growth_authorized=False)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-homeostasis-pilot-v1',runs=[run(42),run(43)],
        target_kl=.025,controller='PI',claim='exploratory CPU pilot; not a stability proof'),sort_keys=True))
