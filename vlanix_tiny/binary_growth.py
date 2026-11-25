"""P0.9E controlled local growth trial; never authorizes production growth."""
import hashlib
import json
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_logic import generate,FAMILIES
from vlanix_tiny.binary_model import batch,prompt_batch
from vlanix_tiny.binary_state import State
from vlanix_tiny.organic import observe,count_params

@torch.no_grad()
def metric(model,rows):
    model.eval()
    loss=0.; hits=0; invalid=0
    for i in range(0,len(rows),32):
        part=rows[i:i+32]; x,l=prompt_batch(part)
        y=torch.tensor([r['answer']+9 for r in part])
        scores=model(x,use_cache=False).logits[torch.arange(len(part)),l-1]
        loss+=F.cross_entropy(scores,y,reduction='sum').item()
        p=scores.argmax(-1)
        hits+=int((p==y).sum()); invalid+=int(((p<9)|(p>264)).sum())
    return dict(rows=len(rows),answer_loss=loss/len(rows),accuracy=hits/len(rows),invalid=invalid)

def run_trial(seed=42,per_family=64):
    if per_family<4: raise ValueError('four probes required')
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed)
    rows=generate(per_family,seed)
    old=set(FAMILIES[:3]); new=set(FAMILIES[3:])
    def subset(split,families): return [r for r in rows if r['split']==split and r['family'] in families]
    a,b=subset('train',old),subset('train',new)
    av,bv=subset('validation',old),subset('validation',new)
    def shuffled(seq):
        order=torch.randperm(len(seq),generator=rng).tolist()
        return [[seq[j] for j in order[i:i+16]] for i in range(0,len(seq),16)]
    warm=shuffled(a); learn=shuffled(b)
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.pt'
        s=State()
        for x in warm: s.step(x)
        base={'old':metric(s.model,av),'new':metric(s.model,bv)}
        groups={f:[batch([r]) for r in b if r['family']==f][:4] for f in FAMILIES[3:]}
        sensor=observe(s.model,groups,seed)
        layer=max(sensor,key=lambda x:x['score'])['layer']
        s.save(path,identity)
        x,_=batch(b[:4])
        with torch.no_grad(): logits=s.model(x).logits.clone()
        results={}
        for name in ('fixed','growth'):
            t=State.restore(path,identity)
            t.model.eval()
            if name=='growth': t.grow(layer,32)
            with torch.no_grad(): delta=(t.model(x).logits-logits).abs().max().item()
            assert delta==0
            for chunk in learn: t.step(chunk)
            results[name]=dict(old=metric(t.model,av),new=metric(t.model,bv),
                params=count_params(t.model),steps=t.steps,insertion_delta=delta,
                target_tokens=2*len(b))
        # Restore the grown model and prove a subsequent step is reproducible.
        grown_path=Path(tmp)/'grown.pt'
        t.save(grown_path,identity)
        q=State.restore(grown_path,identity)
        with torch.no_grad():
            restore_delta=(t.model(x).logits-q.model.eval()(x).logits).abs().max().item()
        assert restore_delta==0
        r1=t.step(learn[0]); r2=q.step(learn[0])
        resume_delta=max((p-u).abs().max().item() for p,u in zip(t.model.parameters(),q.model.parameters()))
        assert resume_delta==0 and r1==r2
    gain=results['fixed']['new']['answer_loss']-results['growth']['new']['answer_loss']
    drift=results['growth']['old']['answer_loss']-base['old']['answer_loss']
    return dict(schema='ngc-binary-growth-v1',seed=seed,selected_layer=layer,baseline=base,
        arms=results,answer_loss_gain_vs_fixed=gain,retention_loss_delta=drift,
        restore_delta=restore_delta,resume_delta=resume_delta,
        test_rows_used=0,growth_authorized=False,
        reason='single-seed pilot, matched steps not matched FLOPs, no calibrated gate')

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(run_trial(),sort_keys=True))
