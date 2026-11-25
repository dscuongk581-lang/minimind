"""P0.11U: second protected learning pass, repeat vs fresh nuisance states."""
import argparse
import copy
import hashlib
import json
import torch
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target
from vlanix_tiny.local_rule_learning import inputs
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.math_curriculum import generate as corpus,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import batches
from vlanix_tiny.protected_learning import evaluate
from vlanix_tiny.learning_state_probe import fit_full,local_stats,probe
from vlanix_tiny.readout_balance import plan,probes
from vlanix_tiny.consolidation import fresh_frames


def digest(rows):
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def run(seed):
    torch.manual_seed(seed)
    old=corpus(seed); val=[r for r in old+or_rows(old) if r['split']=='validation']
    train=plan(seed+10000); held=probes(seed+10000)
    fresh=fresh_frames(train,held,seed+12000)
    assert len(train)==len(fresh)==3840 and len(held)==4096 and len(val)==864
    assert [b for b,c in train]==[b for b,c in fresh]
    assert not set(train)&set(held) and not set(fresh)&(set(train)|set(held))
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    anchor=m.lm_head.weight.detach().clone(); opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    foundation=local_stats(m,qs)
    if foundation['correct']!=32:
        return dict(seed=seed,status='foundation_failed',foundation=foundation,arms={})
    def measure(m):
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        assert m.lm_head.weight is m.model.embed_tokens.weight
        assert local_correct(m,qs)==32
        return dict(geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
            local_correct=32,canonical_probe=probe(m,[(b,0) for b in range(256)]),
            unseen_carry_probe=probe(m,held),validation=evaluate(m,val,qs))
    work=batches(train,qs,seed+11000)
    first=fit_full(m,opt,work,qs,.25)
    before=dict(**first,**measure(m))
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    arms={}
    for name,data in [('repeat',train),('fresh',fresh)]:
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        work=batches(data,qs,seed+13000)
        stats=fit_full(m,opt,work,qs,.25)
        arms[name]=dict(**stats,**measure(m),schedule_sha256=digest(work))
    return dict(seed=seed,status='completed',parameters=EXPECTED_PARAMS,foundation=foundation,
        foundation_steps=120,first_stage=before,arms=arms,train_frames=len(train),
        fresh_frames=len(fresh),held_frames=len(held),held_sha256=digest(held),
        byte_order_sha256=digest([b for b,c in train]),dataset=audit(old),test_rows_used=0)

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=(42,43,44,45))
    args=p.parse_args()
    print(json.dumps(dict(schema='ngc-consolidation-v1',runs=[run(args.seed)],
        torch_version=torch.__version__,deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='shared first stage; matched second-stage byte order; repeat vs fresh carry'),sort_keys=True))
