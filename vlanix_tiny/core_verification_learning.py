"""P0.11V: fixed Fresh recipe, finite exhaustive verification, no test feedback."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from itertools import product
import torch
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target,OPS
from vlanix_tiny.local_rule_learning import inputs,predict_table
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.math_curriculum import generate as corpus
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import batches
from vlanix_tiny.protected_core import pack_prompt
from vlanix_tiny.protected_learning import evaluate
from vlanix_tiny.learning_state_probe import fit_full,local_stats
from vlanix_tiny.readout_balance import plan,probes
from vlanix_tiny.consolidation import fresh_frames
from vlanix_tiny.core_verification import readout_table,score


def expected(op,a,b):
    if op==1: return (a+b)&255
    if op==2: return a^b
    if op==7: return a&b
    if op==8: return a|b
    raise ValueError('operation')


@torch.no_grad()
def verify(m,qs,train,fresh,held,test):
    m.eval()
    local=predict_table(m,qs)
    assert local=={q:predict_table(m,[q])[q] for q in qs}
    def predict(ts):
        return m(torch.tensor([pack_prompt(t) for t in ts]),use_cache=False).logits[:,-1].argmax(-1).tolist()
    table=readout_table(predict)
    assert len(table)==65536
    scheduled=set(train)|set(fresh); held_set=set(held)
    assert not scheduled&held_set and len(scheduled)==7680
    carry_groups={k:dict(rows=0,correct=0,invalid=0) for k in ('scheduled','held','other')}
    for t,p in table.items():
        b=sum(((v-9)&1)<<k for k,v in enumerate(t))
        c=sum(((v-9)>>1)<<k for k,v in enumerate(t))
        key='scheduled' if (b,c) in scheduled else 'held' if (b,c) in held_set else 'other'
        g=carry_groups[key]; g['rows']+=1; g['correct']+=int(p==b+9); g['invalid']+=int(not 9<=p<=264)
    exhaustive={}
    for op,family in OPS.items():
        exhaustive[family]=score(((op,a,b,expected(op,a,b)) for a,b in product(range(256),repeat=2)),
            local.__getitem__,table.__getitem__)
    test_cache=score(((*bytes.fromhex(r['query']),r['answer']) for r in test),local.__getitem__,table.__getitem__)
    direct=evaluate(m,test,qs)
    direct_correct=sum(g['neural_byte_correct'] for g in direct.values())
    parity=direct_correct==test_cache['correct']
    # Aggregate agreement, not a claim of per-row parity for different batch shapes.
    gate=all(g['correct']==g['rows'] for g in carry_groups.values()) and all(g['correct']==g['rows'] for g in exhaustive.values()) and direct_correct==len(test) and parity
    return dict(carry_groups=carry_groups,exhaustive=exhaustive,test_cached=test_cache,test_direct=direct,
        test_aggregate_parity=parity,finite_gate=gate,local_correct=sum(local[q]==target(q) for q in qs))


def run(seed,out):
    torch.manual_seed(seed)
    old=corpus(seed); rows=old+or_rows(old)
    test=[r for r in rows if r['split']=='test']
    assert len(test)==832
    train=plan(seed+10000); held=probes(seed+10000); fresh=fresh_frames(train,held,seed+12000)
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    anchor=m.lm_head.weight.detach().clone(); opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    foundation=local_stats(m,qs)
    if foundation['correct']!=32:
        return dict(seed=seed,status='foundation_failed',foundation=foundation,finite_gate=False)
    first=fit_full(m,opt,batches(train,qs,seed+11000),qs,.25)
    second=fit_full(m,opt,batches(fresh,qs,seed+13000),qs,.25)
    geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    assert m.lm_head.weight is m.model.embed_tokens.weight and local_correct(m,qs)==32
    # Save before evaluating the test split. No further optimizer updates.
    out.mkdir(parents=True,exist_ok=True)
    path=out/'checkpoint.pt'
    torch.save(dict(schema='ngc-finite-core-v1',seed=seed,parameters=EXPECTED_PARAMS,
        model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),
        recipe='P0.11U-Fresh',steps_proposed=320,steps_accepted=first['accepted_steps']+second['accepted_steps']),path)
    with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    result=verify(m,qs,train,fresh,held,test)
    return dict(seed=seed,status='completed',parameters=EXPECTED_PARAMS,foundation=foundation,
        first=first,second=second,geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
        checkpoint_sha256=sha,checkpoint_bytes=path.stat().st_size,test_rows_used=len(test),
        **result)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=(42,43,44,45))
    p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    report=dict(schema='ngc-core-verification-v1',runs=[run(args.seed,args.out)],torch_version=torch.__version__,
        python=platform.python_version(),platform=platform.platform(),threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='finite supervised transition and readout composition, no autonomous reasoning')
    args.out.mkdir(parents=True,exist_ok=True)
    line=json.dumps(report,sort_keys=True)
    (args.out/'report.json').write_text(line+'\n')
    print(line,flush=True)
    if not report['runs'][0]['finite_gate']: raise SystemExit(1)
