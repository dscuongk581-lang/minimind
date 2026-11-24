"""Reference packing with isolated causal attention and reset RoPE positions."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from model.model_minimind import apply_rotary_pos_emb, repeat_kv

@dataclass
class Packed:
    ids: torch.Tensor
    segments: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    used_tokens: int
    target_count: int

def pack(sequences,length):
    if not 3<=length<=512:
        raise ValueError("invalid sequence length")
    rows=[]
    current=[]
    for seq in sequences:
        s=list(seq)
        if not 3<=len(s)<=length or s[0]!=1 or s[-1]!=2 or any(not isinstance(t,int) or isinstance(t,bool) or not 9<=t<4096 for t in s[1:-1]):
            raise ValueError("invalid BOS/content/EOS sequence")
        if sum(map(len,current))+len(s)>length:
            rows.append(current)
            current=[]
        current.append(s)
    if current:
        rows.append(current)
    if not rows:
        raise ValueError("empty pack")
    ids=[]; segments=[]; positions=[]; labels=[]
    used=targets=0
    for row in rows:
        x=[]; seg=[]; pos=[]; y=[]
        for i,s in enumerate(row):
            x.extend(s); seg.extend([i]*len(s)); pos.extend(range(len(s)))
            y.extend([-100]+s[1:])
            used+=len(s); targets+=len(s)-1
        pad=length-len(x)
        ids.append(x+[0]*pad); segments.append(seg+[-1]*pad)
        positions.append(pos+[0]*pad); labels.append(y+[-100]*pad)
    return Packed(*(torch.tensor(x,dtype=torch.long) for x in (ids,segments,positions,labels)),used,targets)

def packed_logits(model,batch):
    # Separate experimental execution path; upstream attention remains unchanged.
    core=model.model
    if core.config.use_moe:
        raise ValueError("dense model required")
    device=next(model.parameters()).device
    ids,seg,pos=(t.to(device) for t in (batch.ids,batch.segments,batch.positions))
    b,n=ids.shape
    same=(seg[:,:,None]==seg[:,None,:]) & (seg[:,:,None]>=0)
    causal=pos[:,:,None]>=pos[:,None,:]
    # Padding queries may attend to self only; avoids all-masked softmax rows.
    mask=(same & causal) | ((seg[:,:,None]<0) & torch.eye(n,device=device,dtype=torch.bool)[None])
    cos,sin=core.freqs_cos[pos].unsqueeze(2),core.freqs_sin[pos].unsqueeze(2)
    h=core.dropout(core.embed_tokens(ids))
    for layer in core.layers:
        a=layer.self_attn
        z=layer.input_layernorm(h)
        q=a.q_norm(a.q_proj(z).view(b,n,a.n_local_heads,a.head_dim))
        k=a.k_norm(a.k_proj(z).view(b,n,a.n_local_kv_heads,a.head_dim))
        v=a.v_proj(z).view(b,n,a.n_local_kv_heads,a.head_dim)
        def rotate(x):
            half=x.shape[-1]//2
            return torch.cat((-x[...,half:],x[...,:half]),dim=-1)
        q=(q*cos+rotate(q)*sin).to(q.dtype)
        k=(k*cos+rotate(k)*sin).to(k.dtype)
        z=F.scaled_dot_product_attention(q.transpose(1,2),repeat_kv(k,a.n_rep).transpose(1,2),repeat_kv(v,a.n_rep).transpose(1,2),attn_mask=mask[:,None],dropout_p=a.dropout if model.training else 0.0)
        h=h+a.resid_dropout(a.o_proj(z.transpose(1,2).reshape(b,n,-1)))
        h=h+layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(core.norm(h))

def packed_loss(model,batch):
    logits=packed_logits(model,batch)
    labels=batch.labels.to(logits.device)
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
