"""Novel OR rule, same globally disjoint operand-pair splits."""
import hashlib
import random

def or_rows(old):
    result=[]; pairs=set()
    for r in old:
        if r['family']!='and': continue
        op,a,b=bytes.fromhex(r['query'])
        key=tuple(sorted((a,b)))
        if op!=7 or key in pairs or r['split'] not in ('train','validation','test'):
            raise ValueError('AND source pair')
        pairs.add(key)
        q=bytes((8,a,b)); y=a|b
        if y!=sum(1<<i for i in range(8) if ((a>>i)&1)+((b>>i)&1)>0):
            raise AssertionError('OR oracle')
        result.append(dict(r,id=hashlib.sha256(q).hexdigest(),query=q.hex(),family='or',answer=y))
    if not result: raise ValueError('empty OR source')
    return result

def adaptation_orders(old,new,seed,epochs=3):
    a=[r for r in old if r['split']=='train']
    b=[r for r in new if r['split']=='train']
    if not a or not b or len(b)%24 or len(a)<8 or len(b)<8 or type(epochs) is not int or epochs<1:
        raise ValueError('batch budget')
    if any(r['family']=='or' for r in a) or any(r['family']!='or' for r in b):
        raise ValueError('rule separation')
    rng=random.Random(seed); pure=[]; replay=[]
    for _ in range(epochs):
        order=list(b); rng.shuffle(order)
        for i in range(0,len(order),24):
            common=order[i:i+24]
            pure.extend(common+rng.sample(b,8))
            replay.extend(common+rng.sample(a,8))
    return pure,replay
