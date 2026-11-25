"""Training-only rotating sentinels and exact finite readout gate."""
import random
from itertools import product


def sentinel_schedule(train,seed,steps=120):
    fixed=[(b,0) for b in range(0,256,4)]
    pool=sorted(set(train)-set(fixed))
    if not set(fixed)<=set(train) or len(pool)<64 or steps<1:
        raise ValueError('sentinel schedule')
    rng=random.Random(seed)
    return [fixed+rng.sample(pool,64) for _ in range(steps)]


def full_readout_ok(table):
    return len(table)==65536 and all(
        table.get(t)==9+sum(((v-9)&1)<<k for k,v in enumerate(t))
        for t in product(range(9,13),repeat=8))
