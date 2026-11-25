"""P0.11A: commutative byte tasks with full train-byte coverage."""
import hashlib
import json
import random
from collections import Counter

OPS={1:'add',2:'xor',7:'and'}

def solve(op,a,b):
    if op not in OPS or any(type(x) is not int or not 0<=x<256 for x in (a,b)):
        raise ValueError('invalid math frame')
    if op==1: return (a+b)&255
    if op==2: return a^b
    return a&b

def oracle(op,a,b):
    if op==1:
        carry=0; out=0
        for i in range(8):
            s=((a>>i)&1)+((b>>i)&1)+carry
            out|=(s%2)<<i; carry=s//2
        return out
    return sum(1<<i for i in range(8) if
        ((a>>i)&1)+((b>>i)&1)==(1 if op==2 else 2))

def group(a,b):
    if max(a,b)<16: return 'easy'
    if min(a,b)>=128: return 'ood'
    return 'hard'

def generate(seed=42):
    rng=random.Random(seed)
    pools={k:[] for k in ('easy','hard','ood')}
    for a in range(256):
        for b in range(a,256):
            pools[group(a,b)].append((a,b))
    for p in pools.values(): rng.shuffle(p)
    # Force every byte to occur as an operand in hard training; no OOD pair.
    coverage={tuple(sorted((a,rng.randrange(16,128)))) for a in range(256)}
    hard=sorted(coverage)
    hard += [p for p in pools['hard'] if p not in coverage][:384-len(hard)]
    reserved=set(hard)
    rest=[p for p in pools['hard'] if p not in reserved]
    groups={('easy','train'):pools['easy'][:96],('easy','validation'):pools['easy'][96:120],
        ('easy','test'):pools['easy'][120:],('hard','train'):hard,
        ('hard','validation'):rest[:96],('hard','test'):rest[96:192],
        ('ood','validation'):pools['ood'][:96],('ood','test'):pools['ood'][96:192]}
    rows=[]
    for (tier,split),pairs in groups.items():
        for a,b in pairs:
            if rng.randrange(2): a,b=b,a
            for op in OPS:
                q=bytes((op,a,b)); y=solve(op,a,b)
                assert y==oracle(op,a,b)
                rows.append(dict(id=hashlib.sha256(q).hexdigest(),query=q.hex(),
                    family=OPS[op],tier=tier,split=split,answer=y))
    audit(rows)
    return rows

def audit(rows):
    ids=set(); pairs={}; seen=set(); counts=Counter()
    for r in rows:
        q=bytes.fromhex(r['query'])
        if len(q)!=3: raise ValueError('frame')
        op,a,b=q
        if (r['id']!=hashlib.sha256(q).hexdigest() or r['id'] in ids or
            r['family']!=OPS.get(op) or r['tier']!=group(a,b) or
            r['split'] not in ('train','validation','test') or
            type(r['answer']) is not int or r['answer']!=solve(op,a,b) or
            r['answer']!=oracle(op,a,b)):
            raise ValueError('row')
        ids.add(r['id'])
        pair=tuple(sorted((a,b)))
        if pair in pairs and pairs[pair]!=r['split']: raise ValueError('pair leakage')
        pairs[pair]=r['split']
        if r['split']=='train':
            if r['tier']=='ood': raise ValueError('OOD leakage')
            seen.update((a,b))
        counts[(r['split'],r['tier'])]+=1
    if seen!=set(range(256)): raise ValueError('missing train operands')
    return dict(rows=len(rows),unique_pairs=len(pairs),train_byte_coverage=len(seen),
        counts={a+'/'+b:n for (a,b),n in counts.items()},
        sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())

def schedule(rows,mode,seed,epochs=2):
    if mode not in ('mixed','curriculum') or type(epochs) is not int or epochs<1:
        raise ValueError('schedule')
    train=[r for r in rows if r['split']=='train']
    if not train: raise ValueError('empty train')
    rng=random.Random(seed); out=[]
    groups=[train] if mode=='mixed' else [[r for r in train if r['tier']==t] for t in ('easy','hard')]
    for g in groups:
        for _ in range(epochs):
            p=list(g); rng.shuffle(p); out.extend(p)
    return out
