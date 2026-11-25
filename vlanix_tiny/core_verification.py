"""Exhaustive finite readout and composition using model-produced IDs."""
from itertools import product, islice


def trace(op,a,b,predict):
    if op not in (1,2,7,8) or any(type(v) is not int or not 0<=v<256 for v in (a,b)):
        raise ValueError('frame')
    out=[]; carry=0
    for k in range(8):
        p=predict((op,(a>>k)&1,(b>>k)&1,carry))
        if type(p) is not int or not 0<=p<=3: return None
        out.append(p+9); carry=p>>1
    return tuple(out)


def readout_table(predict):
    frames=product(range(9,13),repeat=8); table={}
    while batch:=list(islice(frames,64)):
        ids=predict(batch)
        if len(ids)!=len(batch) or any(type(v) is not int or not 0<=v<265 for v in ids):
            raise ValueError('prediction contract')
        table.update(zip(batch,ids))
    return table


def score(rows,local,readout):
    result=dict(rows=0,correct=0,invalid_trace=0,invalid_byte=0)
    for op,a,b,expected in rows:
        result['rows']+=1
        t=trace(op,a,b,local)
        if t is None:
            result['invalid_trace']+=1; continue
        p=readout(t)
        if type(p) is not int or not 9<=p<=264:
            result['invalid_byte']+=1; continue
        result['correct']+=int(p==expected+9)
    return result
