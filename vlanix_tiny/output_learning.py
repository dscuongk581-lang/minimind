"""P0.11I: factorial weighting of state, final byte, and local-rule loss."""
import copy
import hashlib
import json
import time
import torch
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target
from vlanix_tiny.local_rule_learning import inputs
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.trace_protocol import targets,assess
from vlanix_tiny.math_curriculum import generate as corpus,schedule,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import encode,generate,batches
from vlanix_tiny.stream_learning import fit
from vlanix_tiny.output_objective import weighted_loss,decode_states

@torch.no_grad()
def evaluate(m,rows):
    m.eval(); groups={}; start=time.monotonic()
    def predict(p):
        return m(torch.tensor(p),use_cache=False).logits[:,-1].argmax(-1).tolist()
    for i in range(0,len(rows),32):
        part=rows[i:i+32]; qs=[bytes.fromhex(r['query']) for r in part]
        # Free rollout runs first; the reference is never fed back here.
        outputs=generate(qs,predict,'stream')
        # Separate diagnostic: final-byte prediction conditioned on true trace.
        gold_inputs=[encode(q,targets(q),'stream')[0] for q in qs]
        gold_final=predict(gold_inputs)
        for r,q,out,gold in zip(part,qs,outputs,gold_final):
            a=assess(q,out); decoded=decode_states(out[:8])
            g=groups.setdefault(r['tier']+'/'+r['family'],dict(rows=0,answer_correct=0,
                trace_correct=0,joint_correct=0,consistent=0,invalid=0,
                decoded_correct=0,decoded_valid=0,oracle_trace_final_correct=0))
            g['rows']+=1; g['answer_correct']+=int(a['answer_exact'])
            g['trace_correct']+=int(a['trace_exact'])
            g['joint_correct']+=int(a['trace_exact'] and a['answer_exact'])
            g['consistent']+=int(a['consistent']); g['invalid']+=int(not a['valid'])
            g['decoded_correct']+=int(decoded==r['answer'])
            g['decoded_valid']+=int(decoded is not None)
            g['oracle_trace_final_correct']+=int(gold==r['answer']+9)
    return dict(by_group=groups,seconds=time.monotonic()-start)

def run(seed):
    torch.manual_seed(seed)
    old=corpus(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    order=schedule(rows,'mixed',seed+7000,2)
    assert len(order)==3840 and len(val)==864
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m)
    anchor=m.lm_head.weight.detach().clone()
    opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    assert local_correct(m,qs)==32
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    work=batches(order,qs,seed+8000)
    digest=hashlib.sha256(json.dumps([([r['id'] for r in a],b) for a,b in work],
        separators=(',',':')).encode()).hexdigest()
    arms={}
    for name,fw,lw in [('baseline',1,1),('byte8',8,1),('local8',1,8),('both8',8,8)]:
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        assert m.lm_head.weight is m.model.embed_tokens.weight
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        stats=fit(m,opt,work,'stream',lambda z,y: weighted_loss(z,y,fw,lw))
        stats['train_objective']=stats.pop('train_ce')
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[name]=dict(**stats,final_weight=fw,local_weight=lw,
            loss_weight_mass=[192,24*fw,8*lw],
            geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
            schedule_sha256=digest,local_correct=local_correct(m,qs),
            validation=evaluate(m,val))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_steps=120,
        foundation_local_correct=32,dataset=audit(old),validation_rows=len(val),
        test_rows_used=0,arms=arms)

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(schema='ngc-output-learning-v1',runs=[run(42),run(43)],
        claim='matched objective factorial; diagnostic decodes and oracle-conditioned scores separate from free rollout'),sort_keys=True))
