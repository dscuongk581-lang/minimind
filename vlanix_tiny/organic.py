"""P0.8B: append-only FFN growth lab; single-process FP32 CPU."""
import json
import math
import torch
from torch import nn
from model.model_minimind import FeedForward
from vlanix_tiny.config import make_model
from vlanix_tiny.residual import decompose
from vlanix_tiny.logic import generate, FAMILIES
from vlanix_tiny.token_benchmark import fit

CEILING=30_000_000

def count_params(model):
    return sum(p.numel() for p in model.parameters())

class ExpandableFFN(FeedForward):
    def __init__(self,base):
        nn.Module.__init__(self)
        # Reuse exact parameter objects AND their existing qualified names.
        self.gate_proj=base.gate_proj
        self.up_proj=base.up_proj
        self.down_proj=base.down_proj
        self.act_fn=base.act_fn
        self.microblocks=nn.ModuleList()
        self.train(base.training)
    def forward(self,x):
        y=super().forward(x)
        for block in self.microblocks:
            y=y+block(x)
        return y

def add_microblock(model,layer,width,optimizer):
    """Only at a zero-grad step boundary; one AdamW base group plus appends."""
    if type(layer) is not int or not 0<=layer<len(model.model.layers):
        raise ValueError('layer')
    if type(width) is not int or width<=0 or count_params(model)+3*model.config.hidden_size*width>CEILING:
        raise ValueError('budget')
    if model.config.use_moe or model.config.dropout!=0:
        raise ValueError('dense zero-dropout only')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('clear gradients before growth')
    actual={id(p) for p in model.parameters() if p.requires_grad}
    registered={id(p) for g in optimizer.param_groups for p in g['params']}
    if actual!=registered: raise ValueError('optimizer membership')
    target=model.model.layers[layer]
    base=target.mlp
    if not isinstance(base,FeedForward): raise ValueError('FFN')
    ref=base.down_proj.weight
    block=FeedForward(model.config,intermediate_size=width).to(device=ref.device,dtype=ref.dtype)
    nn.init.zeros_(block.down_proj.weight)
    block.train(base.training)
    if not isinstance(base,ExpandableFFN):
        base=ExpandableFFN(base)
        target.mlp=base
    base.microblocks.append(block)
    optimizer.add_param_group({'params':list(block.parameters())})
    return block

def observe(model,groups,seed=42):
    """Fixed unit projections of real FFN output gradients; no optimizer step.
    Diagnostic score only, not a validated bottleneck detector.
    """
    if not groups or any(len(x)<4 for x in groups.values()):
        raise ValueError('need four observations per group')
    layers=model.model.layers
    rng=torch.Generator(device='cpu').manual_seed(seed)
    directions=[torch.randn(model.config.hidden_size,generator=rng) for _ in layers]
    directions=[u/u.norm() for u in directions]
    samples=[{k:[] for k in groups} for _ in layers]
    outputs={}
    hooks=[l.mlp.register_forward_hook(lambda mod,inp,out,i=i:outputs.__setitem__(i,out)) for i,l in enumerate(layers)]
    try:
        for name,batches in groups.items():
            for x in batches:
                outputs.clear()
                result=model(x,labels=x,use_cache=False)
                grads=torch.autograd.grad(result.loss,[outputs[i] for i in range(len(layers))])
                for i,g in enumerate(grads):
                    # Mean of token-level projected gradients, scaled by
                    # sequence target count to undo mean loss normalization.
                    value=(g.detach().float()@directions[i].to(g.device)).sum().item()
                    if not math.isfinite(value): raise ValueError('nonfinite gradient')
                    samples[i][name].append(value)
    finally:
        for h in hooks: h.remove()
    report=[]
    for i,s in enumerate(samples):
        means={k:sum(v)/len(v) for k,v in s.items()}
        score=math.sqrt(sum(v*v for v in means.values())/len(means))
        report.append(dict(layer=i,group_means=means,score=score,moments=decompose(s)))
    return report

def run_probe(seed=42):
    torch.manual_seed(seed)
    model=make_model()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rows=[r for r in generate(4,seed) if r['split']=='train']
    tok=fit([r['text'] for r in rows],4096,False)
    groups={f:[torch.tensor([[1]+tok.encode(r['text']).ids+[2]]) for r in rows if r['family']==f] for f in FAMILIES}
    x=groups['arithmetic'][0]
    # One engineering warmup update, not a trained foundation.
    model(x,labels=x).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    sensor=observe(model,groups,seed)
    layer=max(sensor,key=lambda x:x['score'])['layer']
    old=dict(model.named_parameters())
    before=model(x,labels=x); before.loss.backward()
    gradients={n:p.grad.clone() for n,p in old.items()}
    logits=before.logits.detach().clone()
    opt.zero_grad(set_to_none=True)
    n=count_params(model)
    block=add_microblock(model,layer,32,opt)
    after=model(x,labels=x); after.loss.backward()
    return dict(schema='ngc-organic-probe-v1',params_before=n,params_after=count_params(model),
        selected_layer=layer,sensor=sensor,logit_max_delta=(after.logits.detach()-logits).abs().max().item(),
        gradient_max_delta=max((p.grad-gradients[k]).abs().max().item() for k,p in old.items()),
        new_down_grad_norm=block.down_proj.weight.grad.norm().item(),
        new_up_grad_norm=block.up_proj.weight.grad.norm().item(),
        tied=model.model.embed_tokens.weight is model.lm_head.weight,
        growth_authorized=False,reason='engineering probe; no held-out gain or retention trial')

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(run_probe(),sort_keys=True))
