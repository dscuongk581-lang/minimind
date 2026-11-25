"""P0.11F: neural transition table and external compositional scheduler."""
import json
import time
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import tensor_hash,LR
from vlanix_tiny.local_rule import frames,target,compose,exhaustive

def inputs(q):
    return torch.tensor([[1]+[v+9 for v in row] for row in q])

@torch.no_grad()
def predict_table(m,qs):
    m.eval()
    p=m(inputs(qs),use_cache=False).logits[:,-1].argmax(-1).tolist()
    return {q:v-9 for q,v in zip(qs,p)}

def run(seed):
    torch.manual_seed(seed)
    qs=frames(); x=inputs(qs); y=torch.tensor([target(q)+9 for q in qs])
    m=make_binary_model(); bit_initialize(m)
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    assert m.model.embed_tokens.weight is m.lm_head.weight
    anchor=m.lm_head.weight.detach().clone()
    opt=make_optimizer(m,LR); set_policy(opt,0.)
    trace=[]; total=0.; elapsed=0.
    for i in range(121):
        if i in (0,10,30,60,120):
            table=predict_table(m,qs)
            trace.append(dict(step=i,correct=sum(table[q]==target(q) for q in qs),
                invalid=sum(not 0<=p<=3 for p in table.values()),rows=len(qs)))
        if i==120: break
        start=time.monotonic(); m.train()
        opt.zero_grad(set_to_none=True)
        z=m(x,use_cache=False).logits[:,-1]
        loss=F.cross_entropy(z,y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step(); total+=loss.item(); elapsed+=time.monotonic()-start
    table=predict_table(m,qs)
    # Check batched table against serial inference for every local context.
    direct={q:predict_table(m,[q])[q] for q in qs}
    assert table==direct
    report=exhaustive(table.__getitem__)
    assert sum(v['rows'] for v in report.values())==262144
    geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
    return dict(seed=seed,parameters=EXPECTED_PARAMS,steps=120,unique_local_frames=32,
        supervised_answers=3840,prompt_tokens=19200,train_seconds=elapsed,
        train_ce=total/120,trace=trace,geometry=geo,embedding_sha256=tensor_hash(anchor),
        transition_table=[dict(frame=list(q),prediction=table[q],target=target(q)) for q in qs],
        exhaustive=report,serial_prediction_parity=True,oracle_feedback=False,
        scheduler='external fixed 8-step LSB-first composition; cached model predictions')

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-local-rule-learning-v1',runs=[run(42),run(43)],
        claim='finite local table supervised exhaustively; external composition, not autonomous reasoning'),sort_keys=True))
