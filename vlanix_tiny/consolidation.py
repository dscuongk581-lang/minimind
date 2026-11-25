"""Generate disjoint carry augmentations preserving target order."""
import random
from collections import Counter


def fresh_frames(train,held,seed):
    if not train or any(len(q)!=2 or any(type(v) is not int or not 0<=v<256 for v in q) for q in list(train)+list(held)):
        raise ValueError('frames')
    a,b=set(train),set(held)
    if len(a)!=len(train) or len(b)!=len(held) or a&b:
        raise ValueError('split')
    rng=random.Random(seed)
    forbidden=a|b; pools={}
    for byte,n in sorted(Counter(v for v,c in train).items()):
        available=[c for c in range(256) if (byte,c) not in forbidden]
        if len(available)<n: raise ValueError('capacity')
        pools[byte]=iter(rng.sample(available,n))
    return [(v,next(pools[v])) for v,c in train]
