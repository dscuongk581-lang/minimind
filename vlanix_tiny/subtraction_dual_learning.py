"""P0.12B: dual guardian, rotating training sentinels and transactional full-readout blocks."""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import product
import random
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target,OPS
from vlanix_tiny.local_rule_learning import inputs,predict_table
from vlanix_tiny.protected_core import pack_prompt, guarded_update
from vlanix_tiny.dual_guard import sentinel_schedule, full_readout_ok
from vlanix_tiny.adaptive_guard import backtrack
from vlanix_tiny.learning_state import transition_optimizer
from vlanix_tiny.margin_learning import penalty
from vlanix_tiny.readout_balance import plan,states
from vlanix_tiny.core_verification import readout_table,score
from vlanix_tiny.core_verification_learning import expected
from vlanix_tiny import subtraction as sub


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def load(seed,parent):
    report=json.loads((parent/'report.json').read_text())
    assert report['schema']=='ngc-core-verification-v1' and len(report['runs'])==1
    row=report['runs'][0]
    assert row['seed']==seed and row['status']=='completed' and row['finite_gate'] is True
    path=parent/'checkpoint.pt'
    digest=sha(path); assert digest==row['checkpoint_sha256']
    c=torch.load(path,map_location='cpu',weights_only=True)
    assert c['schema']=='ngc-finite-core-v1' and c['seed']==seed and c['parameters']==EXPECTED_PARAMS
    m=make_binary_model(); m.load_state_dict(c['model'],strict=True)
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert tensor_hash(m.lm_head.weight)==row['embedding_sha256']
    opt=make_optimizer(m,LR); opt.load_state_dict(c['optimizer']); set_policy(opt,0.)
    torch.set_rng_state(c['rng'])
    return m,opt,digest


@torch.no_grad()
def predict_readout(m,ts):
    m.eval()
    return m(torch.tensor([pack_prompt(t) for t in ts]),use_cache=False).logits[:,-1].argmax(-1).tolist()


def run(seed,parent,out,optimizer_mode='retain'):
    m,opt,parent_sha=load(seed,parent)
    transition=transition_optimizer(opt,optimizer_mode)
    anchor=m.lm_head.weight.detach().clone()
    old=frames(); novel=sub.frames(); all_q=old+novel
    labels=[target(q)+9 for q in old]+[sub.target(q)+9 for q in novel]
    old_y=labels[:32]; new_y=labels[32:]
    qx=inputs(all_q); qy=torch.tensor(labels)
    train=plan(seed+10000)
    rng=random.Random(seed+15000)
    schedule=sentinel_schedule(train,seed+18000)
    def local_counts():
        table=predict_table(m,all_q)
        return sum(table[q]+9==y for q,y in zip(old,old_y)),sum(table[q]+9==y for q,y in zip(novel,new_y))
    def full_guard():
        return local_counts()[0]==32 and full_readout_ok(readout_table(lambda ts:predict_readout(m,ts)))
    assert full_guard()
    initial=local_counts()[1]; accepted=attempts=tentative=0
    trace=[dict(step=0,new_correct=initial)]
    blocks=[]
    # Fixed schedule; a rejected block is not mined for error examples.
    work=[rng.sample(train,24) for _ in range(120)]
    for start in range(0,120,40):
        block_accepted=block_attempts=0
        def block_update():
            nonlocal block_accepted,block_attempts
            for i in range(start,start+40):
                sent=schedule[i]
                sentinel_ts=[states(b,c) for b,c in sent]
                sentinel_y=[b+9 for b,c in sent]
                rows=work[i]
                rx=torch.tensor([pack_prompt(states(b,c)) for b,c in rows])
                ry=torch.tensor([b+9 for b,c in rows])
                def guard():
                    return local_counts()[0]==32 and predict_readout(m,sentinel_ts)==sentinel_y
                def update():
                    m.train(); opt.zero_grad(set_to_none=True)
                    z=m(qx,use_cache=False).logits[:,-1]
                    rz=m(rx,use_cache=False).logits[:,-1]
                    loss=.5*F.cross_entropy(z[32:],qy[32:])+.25*F.cross_entropy(z[:32],qy[:32])+.25*F.cross_entropy(rz,ry)
                    loss=loss+.25*penalty(z[:32],qy[:32],1.)+.25*penalty(rz,ry,1.)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    return loss.item()
                result=backtrack(m,opt,update,guard)
                block_accepted+=int(result['scale'] is not None)
                block_attempts+=result['attempts']
            return 0.
        committed,_=guarded_update(m,opt,block_update,full_guard)
        attempts+=block_attempts; tentative+=block_accepted
        if committed: accepted+=block_accepted
        blocks.append(dict(step=start+40,committed=committed,tentative_steps=block_accepted,committed_steps=accepted,attempts=attempts))
        trace.append(dict(step=start+40,new_correct=local_counts()[1],accepted=accepted,attempts=attempts))
    geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    out.mkdir(parents=True,exist_ok=True)
    path=out/'checkpoint.pt'
    torch.save(dict(schema='ngc-sub-core-v1',seed=seed,parameters=EXPECTED_PARAMS,parent_sha256=parent_sha,
        model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),
        proposed_steps=120,accepted_steps=accepted,optimizer_mode=optimizer_mode),path)
    table=predict_table(m,all_q)
    assert table=={q:predict_table(m,[q])[q] for q in all_q}
    readout=readout_table(lambda ts:predict_readout(m,ts))
    readout_correct=sum(p==9+sum(((v-9)&1)<<k for k,v in enumerate(t)) for t,p in readout.items())
    old_report={f:score(((op,a,b,expected(op,a,b)) for a,b in product(range(256),repeat=2)),table.__getitem__,readout.__getitem__)
        for op,f in OPS.items()}
    # Generator is consumed inside score, before op changes.
    new_report=sub.score(table.__getitem__,readout.__getitem__)
    old_correct,new_correct=local_counts()
    gate=(old_correct==32 and new_correct==8 and readout_correct==65536 and
          all(g['correct']==g['rows'] for g in old_report.values()) and new_report['correct']==65536)
    return dict(seed=seed,parent_sha256=parent_sha,checkpoint_sha256=sha(path),
        parameters=EXPECTED_PARAMS,geometry=geo,optimizer_mode=optimizer_mode,optimizer_transition=transition,initial_new_correct=initial,local_old_correct=old_correct,
        local_new_correct=new_correct,sentinel_count=128,accepted_steps=accepted,tentative_steps=tentative,optimizer_attempts=attempts,
        blocks=blocks,readout_regression_used_during_training=True,
        learning_trace=trace,readout=dict(rows=65536,correct=readout_correct),
        old_exhaustive=old_report,new_exhaustive=new_report,finite_gate=gate)


if __name__=='__main__':
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=(42,43,44,45))
    p.add_argument('--parent',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--optimizer-mode',choices=('retain','reset'),default='retain')
    args=p.parse_args()
    result=run(args.seed,args.parent,args.out,args.optimizer_mode)
    line=json.dumps(dict(schema='ngc-subtraction-dual-guard-v1',runs=[result],
        claim='supervised SUB; repeated finite regression gate used during training; not independent evidence or autonomous discovery'),sort_keys=True)
    (args.out/'report.json').write_text(line+'\n'); print(line,flush=True)
    if not result['finite_gate']: raise SystemExit(1)
