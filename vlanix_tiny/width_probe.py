"""Width extrapolation diagnostics; no oracle in free rollout."""
import random

OPS=(1,2,7,8)


def check(q):
    op,a,b,w=q
    if op not in OPS or any(type(v) is not int for v in q) or not 1<=w<=32 or not 0<=a<1<<w or not 0<=b<1<<w:
        raise ValueError('frame')
    return op,a,b,w


def gold(q):
    op,a,b,w=check(q)
    v=a+b if op==1 else a^b if op==2 else a&b if op==7 else a|b
    return [9+((v>>k)&1)+2*(((a&((1<<(k+1))-1))+(b&((1<<(k+1))-1)))>>(k+1) if op==1 else 0) for k in range(w)]


def decode(states):
    if not 1<=len(states)<=32 or any(type(v) is not int or not 9<=v<=12 for v in states): return None
    return sum(((v-9)&1)<<k for k,v in enumerate(states))


def extend(q,width,seed):
    op,a,b,w=check(q)
    if type(width) is not int or not w<=width<=32: raise ValueError('width')
    rng=random.Random(seed)
    return op,a|(rng.getrandbits(width-w)<<w),b|(rng.getrandbits(width-w)<<w),width


def rollout(qs,predict,teacher=False):
    if not qs: raise ValueError('empty batch')
    qs=[check(q) for q in qs]; w=qs[0][3]
    if any(q[3]!=w for q in qs): raise ValueError('mixed widths')
    x=[[1,q[0]+9] for q in qs]; out=[[] for q in qs]
    truth=[gold(q) for q in qs] if teacher else None
    for k in range(w):
        for p,(_,a,b,_) in zip(x,qs): p.extend([9+((a>>k)&1),9+((b>>k)&1)])
        ids=predict([p[:] for p in x])
        if len(ids)!=len(qs) or any(type(v) is not int or not 0<=v<265 for v in ids): raise ValueError('prediction contract')
        for i,(p,o,v) in enumerate(zip(x,out,ids)):
            o.append(v); p.append(truth[i][k] if teacher else v)
    return out


def summarize(qs,outputs):
    if not qs or len(qs)!=len(outputs): raise ValueError('batch')
    w=check(qs[0])[3]; pos=[0]*w; first=[0]*w; exact=correct=valid=0
    for q,o in zip(qs,outputs):
        t=gold(q)
        if len(t)!=w or len(o)!=w: raise ValueError('width')
        errors=[k for k,(a,b) in enumerate(zip(t,o)) if a!=b]
        for k in errors: pos[k]+=1
        if errors: first[errors[0]]+=1
        else: exact+=1
        d=decode(o); valid+=int(d is not None); correct+=int(d==decode(t))
    return dict(rows=len(qs),trace_exact=exact,decoded_correct=correct,valid=valid,position_errors=pos,first_error=first)
