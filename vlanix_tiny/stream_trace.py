"""P0.11H: identical supervision in two causal layouts, plus local rehearsal."""
import random


def chunks(q):
    if not isinstance(q,bytes) or len(q)!=3 or q[0] not in (1,2,7,8):
        raise ValueError('frame')
    op,a,b=q
    return [1,op+9],[[9+((a>>k)&1),9+((b>>k)&1)] for k in range(8)]


def encode(q,t,mode):
    prefix,parts=chunks(q)
    if len(t)!=9 or any(type(v) is not int or not 9<=v<=12 for v in t[:8]) or not 9<=t[-1]<=264:
        raise ValueError('targets')
    if mode=='front':
        x=prefix+sum(parts,[])+list(t[:8]); y=[-100]*17+list(t)
    elif mode=='stream':
        x=prefix[:]; y=[-100]*26
        for k,p in enumerate(parts):
            x.extend(p); y[len(x)-1]=t[k]; x.append(t[k])
        y[-1]=t[-1]
    else: raise ValueError('mode')
    assert len(x)==len(y)==26
    return x,y


def generate(qs,predict,mode):
    if not qs or mode not in ('front','stream'): raise ValueError('mode or frames')
    parsed=[chunks(q) for q in qs]
    x=[p+sum(c,[]) if mode=='front' else p[:] for p,c in parsed]
    out=[[] for q in qs]
    for k in range(9):
        if mode=='stream' and k<8:
            for row,(_,parts) in zip(x,parsed): row.extend(parts[k])
        ids=predict([p[:] for p in x])
        if len(ids)!=len(x) or any(type(v) is not int or not 0<=v<265 for v in ids):
            raise ValueError('prediction contract')
        for row,o,v in zip(x,out,ids): row.append(v); o.append(v)
    return out


def batches(rows,local,seed):
    if not rows or len(rows)%24 or len(local)<8: raise ValueError('batch budget')
    rng=random.Random(seed)
    return [(rows[i:i+24],rng.sample(local,8)) for i in range(0,len(rows),24)]
