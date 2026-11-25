"""SUB modulo 256: explicit learned borrow recurrence."""
from itertools import product

SUB=10


def frames():
    return list(product((SUB,),range(2),range(2),range(2)))


def target(q):
    op,a,b,c=q
    if op!=SUB or any(type(v) is not int or v not in (0,1) for v in (a,b,c)):
        raise ValueError('sub frame')
    d=a-b-c
    return (d&1)+2*int(d<0)


def trace(a,b,predict):
    if any(type(v) is not int or not 0<=v<256 for v in (a,b)):
        raise ValueError('byte frame')
    out=[]; borrow=0
    for k in range(8):
        p=predict((SUB,(a>>k)&1,(b>>k)&1,borrow))
        if type(p) is not int or not 0<=p<=3: return None
        out.append(p+9); borrow=p>>1
    return tuple(out)


def score(local,readout):
    result=dict(rows=0,correct=0,invalid_trace=0,invalid_byte=0)
    for a,b in product(range(256),repeat=2):
        result['rows']+=1
        t=trace(a,b,local)
        if t is None:
            result['invalid_trace']+=1; continue
        p=readout(t)
        if type(p) is not int or not 9<=p<=264:
            result['invalid_byte']+=1; continue
        result['correct']+=int(p==((a-b)&255)+9)
    return result
