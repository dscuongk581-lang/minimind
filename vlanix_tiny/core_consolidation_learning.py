"""P0.11X: fixed third-stage consolidation of hash-pinned W parent artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames
from vlanix_tiny.trace_learning import local_correct
from vlanix_tiny.math_curriculum import generate as corpus
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import batches
from vlanix_tiny.learning_state_probe import fit_full
from vlanix_tiny.readout_balance import plan,probes
from vlanix_tiny.consolidation import fresh_frames
from vlanix_tiny.core_consolidation import third_frames,validate_parent
from vlanix_tiny.core_verification_learning import verify

PARENT_RUN=45615004505
PARENT_SHA={
    42:'d97429a57dc8e981affc08707d09219ae63e8b2ea12ff26f53f44c622dfe6eac',
    43:'a69e59537301b938e0abe09acb765e24f99c04b12f046a9b89f7d2d89b2ea3c2',
    44:'a6f8a10b007af3818518a3c9988f236b94a4a4679a6ac444529e09a9bd4f46fd',
    45:'65aa1fe8337dffc04a1aa308d0fd4bba60f8e235afee45f509297c431bb458d1',
}

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def run(seed,parent,out):
    report=json.loads((parent/'report.json').read_text())
    assert report['schema']=='ngc-core-verification-v1' and len(report['runs'])==1
    row=report['runs'][0]; path=parent/'checkpoint.pt'
    digest=sha(path); assert digest==PARENT_SHA[seed]
    validate_parent(row,seed,digest)
    c=torch.load(path,map_location='cpu',weights_only=True)
    assert c['schema']=='ngc-finite-core-v1' and c['seed']==seed and c['parameters']==EXPECTED_PARAMS
    assert c['recipe']=='P0.11U-Fresh' and c['steps_proposed']==320
    m=make_binary_model(); m.load_state_dict(c['model'],strict=True)
    anchor=m.lm_head.weight.detach().clone()
    assert tensor_hash(anchor)==row['embedding_sha256']
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    opt=make_optimizer(m,LR); opt.load_state_dict(c['optimizer']); set_policy(opt,0.)
    torch.set_rng_state(c['rng'])
    qs=frames(); assert local_correct(m,qs)==32
    train=plan(seed+10000); held=probes(seed+10000)
    fresh=fresh_frames(train,held,seed+12000)
    third=third_frames(train,fresh,held,seed+16000)
    assert len(third)==3840 and len(set(train)|set(fresh)|set(third))==11520
    assert not set(third)&set(held)
    # No example selected by previous prediction errors; same schedule for each seed.
    third_stats=fit_full(m,opt,batches(third,qs,seed+17000),qs,.25)
    geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
    assert local_correct(m,qs)==32 and sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
    assert m.lm_head.weight is m.model.embed_tokens.weight
    out.mkdir(parents=True,exist_ok=True); path=out/'checkpoint.pt'
    torch.save(dict(schema='ngc-finite-core-v1',seed=seed,parameters=EXPECTED_PARAMS,
        model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),
        recipe='P0.11X-ThirdFresh',parent_sha256=digest,steps_proposed=480,
        steps_accepted=c['steps_accepted']+third_stats['accepted_steps']),path)
    old=corpus(seed); test=[r for r in old+or_rows(old) if r['split']=='test']
    assert len(test)==832
    result=verify(m,qs,train,fresh,held,test,extra=third)
    return dict(seed=seed,status='completed',parameters=EXPECTED_PARAMS,recipe='P0.11X-ThirdFresh',
        parent_run=PARENT_RUN,parent_sha256=digest,parent_finite_gate=row['finite_gate'],
        parent_readout_correct=sum(g['correct'] for g in row['carry_groups'].values()),
        third=third_stats,geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
        checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,test_rows_used=len(test),**result)

if __name__=='__main__':
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=(42,43,44,45))
    p.add_argument('--parent',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=run(args.seed,args.parent,args.out)
    line=json.dumps(dict(schema='ngc-core-verification-v1',runs=[result],torch_version=torch.__version__,
        claim='third fixed training pass; repeated finite verification; not a new independent test'),sort_keys=True)
    (args.out/'report.json').write_text(line+'\n'); print(line,flush=True)
    if not result['finite_gate']: raise SystemExit(1)
