"""Learn a finite transition, then compose it without oracle feedback."""
from itertools import product
OPS={1:'add',2:'xor',7:'and',8:'or'}

def frames():
    return list(product(OPS,range(2),range(2),range(2)))

def target(q):
    op,a,b,c=q
    if op not in OPS or any(type(x) is not int or x not in (0,1) for x in (a,b,c)):
        raise ValueError('frame')
    if op==1: return a+b+c
    if op==2: return a^b
    if op==7: return a&b
    return a|b

def compose(op,a,b,predict):
    if op not in OPS or any(type(x) is not int or not 0<=x<256 for x in (a,b)):
        raise ValueError('byte frame')
    carry=0; out=0
    for k in range(8):
        p=predict((op,(a>>k)&1,(b>>k)&1,carry))
        if type(p) is not int or not 0<=p<=3: return None
        out|=(p&1)<<k; carry=p>>1
    return out

def exhaustive(predict):
    result={}
    for op,f in OPS.items():
        groups={k:dict(rows=0,correct=0,invalid=0) for k in ('easy','hard','ood')}
        for a,b in product(range(256),repeat=2):
            expected=((a+b)&255) if op==1 else a^b if op==2 else a&b if op==7 else a|b
            p=compose(op,a,b,predict)
            tier='easy' if max(a,b)<16 else 'ood' if min(a,b)>=128 else 'hard'
            g=groups[tier]; g['rows']+=1; g['correct']+=int(p==expected); g['invalid']+=int(p is None)
        for tier,g in groups.items(): result[tier+'/'+f]=g
    return result
