"""P0.11M: protected local transitions and independent neural byte readout."""
import copy
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from vlanix_tiny.binary_model import make_binary_model,EXPECTED_PARAMS
from vlanix_tiny.math_learning import bit_initialize
from vlanix_tiny.embedding_policy import make_optimizer,set_policy,geometry
from vlanix_tiny.embedding_conservation import LR,tensor_hash
from vlanix_tiny.local_rule import frames,target
from vlanix_tiny.local_rule_learning import inputs,predict_table
from vlanix_tiny.trace_learning import step,local_correct
from vlanix_tiny.trace_protocol import targets
from vlanix_tiny.math_curriculum import generate as corpus,schedule,audit
from vlanix_tiny.rule_transfer_data import or_rows
from vlanix_tiny.stream_trace import batches
from vlanix_tiny.protected_core import guarded_update,pack_prompt
from vlanix_tiny.adaptive_guard import backtrack
from vlanix_tiny.gradient_direction import combine
from vlanix_tiny.state_invariance import reset_rollout
from vlanix_tiny.width_probe import decode
from vlanix_tiny.width_learning_probe import evaluate_width


def batch(rows,local):
    if len(rows)!=24 or len(local)!=8 or any(r['split']!='train' for r in rows):
        raise ValueError('training batch')
    x=[]; y=[]
    for r in rows:
        t=targets(bytes.fromhex(r['query'])); assert t[-1]==r['answer']+9
        x.append(pack_prompt(t[:8])); y.append([-100]*9+[t[-1]])
    for q in local:
        x.append([1]+[v+9 for v in q]+[0]*5)
        y.append([-100]*4+[target(q)+9]+[-100]*5)
    return torch.tensor(x),torch.tensor(y)


def fit(m,opt,work,guarded,qs,adaptive=False,direction=None,batch_fn=batch,loss_fn=None):
    accepted=0; min_correct=32; losses=[]; start=time.monotonic()
    attempts=0; scale_counts={}; scale_sum=0.
    gradient_trials=conflicts=0
    core=[p for p in m.parameters() if p is not m.lm_head.weight]
    for rows,local in work:
        x,y=batch_fn(rows,local)
        def update():
            nonlocal gradient_trials,conflicts
            m.train(); opt.zero_grad(set_to_none=True)
            z=m(x,use_cache=False).logits
            loss=(F.cross_entropy(z.reshape(-1,265),y.reshape(-1),ignore_index=-100)
                if loss_fn is None else loss_fn(z,y))
            if direction is None:
                loss.backward()
            else:
                new=F.cross_entropy(z[:24,9],y[:24,9])
                old=F.cross_entropy(z[24:,4],y[24:,4])
                g=torch.autograd.grad(new,core,retain_graph=True)
                h=torch.autograd.grad(old,core)
                merged,info=combine(g,h,direction=='projected')
                for p,v in zip(core,merged): p.grad=v
                gradient_trials+=1; conflicts+=int(info['conflict'])
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            return loss.item()
        if adaptive:
            assert guarded
            result=backtrack(m,opt,update,lambda: local_correct(m,qs)==32)
            scale=result['scale']; ok=scale is not None; loss=result['loss']
            attempts+=result['attempts']
        elif guarded:
            ok,loss=guarded_update(m,opt,update,lambda: local_correct(m,qs)==32)
            scale=1. if ok else None; attempts+=1
        else: loss=update(); ok=True; scale=1.; attempts+=1
        key=str(scale) if scale is not None else 'rejected'
        scale_counts[key]=scale_counts.get(key,0)+1
        scale_sum+=scale or 0.
        c=local_correct(m,qs); min_correct=min(min_correct,c)
        if guarded: assert c==32
        accepted+=int(ok); losses.append(loss)
    return dict(proposed_steps=len(work),accepted_steps=accepted,rejected_steps=len(work)-accepted,
        min_local_correct=min_correct,mean_proposed_loss=sum(losses)/len(losses),
        target_tokens=32*len(work),optimizer_attempts=attempts,attempt_target_tokens=32*attempts,
        scale_counts=scale_counts,accepted_scale_sum=scale_sum,gradient_trials=gradient_trials,
        gradient_conflicts=conflicts,train_seconds=time.monotonic()-start)


@torch.no_grad()
def evaluate(m,rows,qs):
    m.eval(); groups={}
    table=predict_table(m,qs)
    def predict(p):
        return m(torch.tensor(p),use_cache=False).logits[:,-1].argmax(-1).tolist()
    def table_predict(batch):
        return [table[tuple(v-9 for v in p[1:])]+9 for p in batch]
    # All contexts of the cached table come from the model; verify single-call parity.
    direct={q:predict_table(m,[q])[q] for q in qs}; assert direct==table
    for i in range(0,len(rows),32):
        part=rows[i:i+32]; frames8=[(*bytes.fromhex(r['query']),8) for r in part]
        traces=reset_rollout(frames8,table_predict)
        valid=[(j,pack_prompt(t)) for j,t in enumerate(traces) if pack_prompt(t) is not None]
        predicted=dict(zip([j for j,_ in valid],predict([p for _,p in valid]))) if valid else {}
        truth=[targets(bytes.fromhex(r['query'])) for r in part]
        oracle=predict([pack_prompt(t[:8]) for t in truth])
        for j,(r,t,gold,ref) in enumerate(zip(part,traces,truth,oracle)):
            g=groups.setdefault(r['tier']+'/'+r['family'],dict(rows=0,trace_correct=0,decoded_correct=0,
                neural_byte_correct=0,joint_correct=0,invalid_trace=0,invalid_byte=0,gold_trace_byte_correct=0))
            hit=predicted.get(j)==r['answer']+9
            exact=t==gold[:8]
            g['rows']+=1; g['trace_correct']+=int(exact)
            g['decoded_correct']+=int(decode(t)==r['answer'])
            g['neural_byte_correct']+=int(hit); g['joint_correct']+=int(hit and exact)
            g['invalid_trace']+=int(j not in predicted)
            g['invalid_byte']+=int(j in predicted and not 9<=predicted[j]<=264)
            g['gold_trace_byte_correct']+=int(ref==r['answer']+9)
    return groups


def run(seed,modes=('ordinary','guarded')):
    torch.manual_seed(seed)
    old=corpus(seed); rows=old+or_rows(old)
    val=[r for r in rows if r['split']=='validation']
    order=schedule(rows,'mixed',seed+7000,2)
    assert len(order)==3840 and len(val)==864
    qs=frames(); x=inputs(qs); y=torch.tensor([[target(q)+9] for q in qs])
    m=make_binary_model(); bit_initialize(m); anchor=m.lm_head.weight.detach().clone()
    opt=make_optimizer(m,LR); set_policy(opt,0.)
    for _ in range(120): step(m,opt,x,y,x.shape[1]-1)
    assert local_correct(m,qs)==32
    weights=copy.deepcopy(m.state_dict()); moments=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone(); work=batches(order,qs,seed+8000)
    digest=hashlib.sha256(json.dumps([([r['id'] for r in a],b) for a,b in work],
        separators=(',',':')).encode()).hexdigest()
    arms={}
    for mode in modes:
        m=make_binary_model(); m.load_state_dict(weights,strict=True)
        assert sum(p.numel() for p in m.parameters())==EXPECTED_PARAMS
        assert m.lm_head.weight is m.model.embed_tokens.weight
        opt=make_optimizer(m,LR); opt.load_state_dict(copy.deepcopy(moments))
        torch.set_rng_state(rng.clone()); set_policy(opt,0.)
        stats=fit(m,opt,work,mode!='ordinary',qs,adaptive=mode in ('adaptive','mixed','projected'),
            direction=mode if mode in ('mixed','projected') else None)
        geo=geometry(anchor,m.lm_head.weight); assert geo['exact']
        arms[mode]=dict(**stats,geometry=geo,embedding_sha256=tensor_hash(m.lm_head.weight),
            schedule_sha256=digest,local_correct=local_correct(m,qs),
            validation=evaluate(m,val,qs),width_probe=evaluate_width(m,val,seed))
    return dict(seed=seed,parameters=EXPECTED_PARAMS,foundation_steps=120,
        foundation_local_correct=32,dataset=audit(old),validation_rows=len(val),
        test_rows_used=0,arms=arms)

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(dict(schema='ngc-protected-learning-v1',runs=[run(42),run(43)],
        torch_version=torch.__version__,threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='local argmax preservation under serial CPU transaction; external scheduling and neural readout'),sort_keys=True))
