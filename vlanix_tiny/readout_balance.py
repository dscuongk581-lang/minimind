"""Balanced byte readout frames; carry bits are nuisance variables."""
import random


def states(byte,carry):
    if any(type(v) is not int or not 0<=v<256 for v in (byte,carry)):
        raise ValueError('byte or carry')
    return [9+((byte>>k)&1)+2*((carry>>k)&1) for k in range(8)]


def _split(seed):
    rng=random.Random(seed); train=[]; held=[]
    for b in range(256):
        masks=rng.sample(range(1,256),30)
        train.extend((b,c) for c in [0]+masks[:14])
        held.extend((b,c) for c in masks[14:])
    rng.shuffle(train); rng.shuffle(held)
    return train,held


def plan(seed):
    return _split(seed)[0]


def probes(seed):
    return _split(seed)[1]
