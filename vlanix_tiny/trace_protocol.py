"""Research trace protocol: prompt bits only; oracle used for labels/scoring."""
OPS=(1,2,7,8)

def prompt(q):
    if not isinstance(q,bytes) or len(q)!=3 or q[0] not in OPS:
        raise ValueError('frame')
    op,a,b=q
    return [1,op+9]+[9+((v>>k)&1) for k in range(8) for v in (a,b)]

def targets(q):
    prompt(q)
    op,a,b=q; carry=0; out=[]
    for k in range(8):
        x=(a>>k)&1; y=(b>>k)&1
        p=x+y+carry if op==1 else x^y if op==2 else x&y if op==7 else x|y
        out.append(p+9); carry=p>>1
    answer=((a+b)&255) if op==1 else a^b if op==2 else a&b if op==7 else a|b
    assert sum(((p-9)&1)<<k for k,p in enumerate(out))==answer
    return out+[answer+9]

def rollout(prompts,predict,steps):
    if not prompts or type(steps) is not int or steps<1:
        raise ValueError('rollout')
    batch=[list(p) for p in prompts]; out=[[] for p in batch]
    for _ in range(steps):
        ids=predict([p[:] for p in batch])
        if len(ids)!=len(batch) or any(type(i) is not int or not 0<=i<265 for i in ids):
            raise ValueError('prediction contract')
        for p,o,i in zip(batch,out,ids): p.append(i); o.append(i)
    return out

def assess(q,out):
    if len(out)!=9: raise ValueError('trace length')
    expected=targets(q)
    trace_valid=all(type(p) is int and 9<=p<=12 for p in out[:8])
    valid=trace_valid and type(out[-1]) is int and 9<=out[-1]<=264
    consistent=bool(valid and sum(((p-9)&1)<<k for k,p in enumerate(out[:8]))==out[-1]-9)
    return dict(answer_exact=out[-1]==expected[-1],trace_exact=out[:8]==expected[:8],consistent=consistent,valid=valid)
