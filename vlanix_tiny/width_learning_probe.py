"""Inference-only paired width probe on each output-objective arm."""
import hashlib
import json
import random
import torch
from vlanix_tiny.width_probe import OPS,extend,rollout,summarize


@torch.no_grad()
def evaluate_width(m,rows,seed):
    m.eval()
    rng=random.Random(seed+9100)
    qs=[]
    for op in OPS:
        candidates=[(*bytes.fromhex(r['query']),8) for r in rows
                    if r['split']=='validation' and bytes.fromhex(r['query'])[0]==op]
        qs.extend(rng.sample(candidates,16))
    assert len(qs)==64
    result={}; prev_free=None; prev_teacher=None
    def predict(batch):
        return m(torch.tensor(batch),use_cache=False).logits[:,-1].argmax(-1).tolist()
    for w in (8,16,32):
        if w!=8: qs=[extend(q,w,rng.randrange(1<<32)) for q in qs]
        free=[]; teacher=[]
        # Separate complete free rollout before teacher-conditioned diagnostics.
        for i in range(0,len(qs),16): free.extend(rollout(qs[i:i+16],predict))
        for i in range(0,len(qs),16): teacher.extend(rollout(qs[i:i+16],predict,teacher=True))
        prefix=None if prev_free is None else dict(
            free=sum(a[:len(b)]==b for a,b in zip(free,prev_free)),
            teacher=sum(a[:len(b)]==b for a,b in zip(teacher,prev_teacher)))
        result[str(w)]=dict(free=summarize(qs,free),teacher=summarize(qs,teacher),
            prefix_parity=prefix,by_op={str(op):dict(
                free=summarize(qs[i:i+16],free[i:i+16]),
                teacher=summarize(qs[i:i+16],teacher[i:i+16]))
                for i,op in zip(range(0,64,16),OPS)},
            frame_sha256=hashlib.sha256(json.dumps(qs,separators=(',',':')).encode()).hexdigest())
        prev_free=free; prev_teacher=teacher
    return dict(training_updates=0,rows_per_width=64,widths=result,
        output='external decode of predicted states only; no final byte prediction',
        teacher='separate diagnostic with gold prefix; not free performance')
