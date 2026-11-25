"""Research-only readout delimiter; no change to binary token identity."""
READOUT=18  # existing byte value 9, used as a marker only in this grammar


def chunks(q):
    if not isinstance(q,bytes) or len(q)!=3 or q[0] not in (1,2,7,8):
        raise ValueError('frame')
    op,a,b=q
    return [1,op+9],[[9+((a>>k)&1),9+((b>>k)&1)] for k in range(8)]


def encode(q,t,tagged):
    x,parts=chunks(q)
    if len(t)!=9 or any(type(v) is not int or not 9<=v<=12 for v in t[:8]) or type(t[-1]) is not int or not 9<=t[-1]<=264:
        raise ValueError('targets')
    y=[-100]*27
    for k,p in enumerate(parts):
        x.extend(p); y[len(x)-1]=t[k]; x.append(t[k])
    y[26 if tagged else 25]=t[-1]
    x.append(READOUT if tagged else 0)
    return x,y


def generate(qs,predict,tagged):
    if not qs: raise ValueError('empty batch')
    parsed=[chunks(q) for q in qs]
    x=[p[:] for p,_ in parsed]; out=[[] for q in qs]
    for k in range(9):
        if k<8:
            for p,(_,parts) in zip(x,parsed): p.extend(parts[k])
        elif tagged:
            for p in x: p.append(READOUT)
        ids=predict([p[:] for p in x])
        if len(ids)!=len(x) or any(type(v) is not int or not 0<=v<265 for v in ids):
            raise ValueError('prediction contract')
        for p,o,v in zip(x,out,ids): p.append(v); o.append(v)
    return out
