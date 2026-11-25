"""P0.9C: bounded binary task grammar; no execution of payload bytes."""
import hashlib
import json
import random
from collections import Counter

FAMILIES=('arithmetic','boolean','state','pattern','inference','recurrence')
LENGTHS=(3,3,4,4,10,4)
RANGES={'train':range(128),'validation':range(128,192),'test':range(192,256)}

def split_for(a):
    return 'train' if a<128 else 'validation' if a<192 else 'test'

def solve(q):
    """Reference interpreter, not used as model input or at inference."""
    if not q or not 1<=q[0]<=6 or len(q)!=LENGTHS[q[0]-1]:
        raise ValueError('opcode or length')
    op,a,*v=q
    if op==1: return (a+v[0])%256
    if op==2: return a^v[0]
    if op==3: return ((a+v[0])*v[1])%256
    if op==4:
        d=(v[0]-a)%256
        if (v[1]-v[0])%256!=d: raise ValueError('pattern')
        return (v[1]+d)%256
    if op==5:
        seen=a
        for _ in range(8):
            for i,edges in enumerate(v):
                if seen&(1<<i): seen|=edges
        return seen
    n=v[1]
    if not 2<=n<=8: raise ValueError('recurrence depth')
    b=v[0]
    for _ in range(n): a,b=b,(a+b)%256
    return a

def oracle(q):
    """Independent alternative formulas / graph traversal for verification."""
    op,a,*v=q
    if op==1: return (a+v[0])&255
    if op==2: return sum((1<<i) for i in range(8) if (a>>i&1)!=(v[0]>>i&1))
    if op==3: return sum([a+v[0]]*v[1])&255
    if op==4: return (3*v[0]-2*a)&255
    if op==5:
        todo=[i for i in range(8) if a&(1<<i)]; seen=set(todo)
        while todo:
            i=todo.pop()
            for j in range(8):
                if v[i]&(1<<j) and j not in seen: seen.add(j); todo.append(j)
        return sum(1<<i for i in seen)
    fib=[0,1,1,2,3,5,8,13,21]
    return (fib[v[1]-1]*a+fib[v[1]]*v[0])&255

def generate(per_family=128,seed=42):
    if type(per_family) is not int or not 1<=per_family<=4096: raise ValueError('count')
    rng=random.Random(seed); rows=[]
    for split,domain in RANGES.items():
        for op,family in enumerate(FAMILIES,1):
            used=set()
            while len(used)<per_family:
                a=rng.choice(domain); b=rng.randrange(256)
                if op in (1,2): tail=[b]
                elif op==3: tail=[b,rng.randrange(256)]
                elif op==4: tail=[(a+b)%256,(a+2*b)%256]
                elif op==5: tail=[rng.choice([0]+[1<<i for i in range(8)]) for _ in range(8)]
                else: tail=[b,rng.randrange(2,9)]
                q=bytes([op,a,*tail])
                if q in used: continue
                used.add(q)
                rows.append(dict(id=hashlib.sha256(q).hexdigest(),split=split,family=family,query=q.hex(),answer=solve(q)))
    return rows

def verify(row):
    try:
        q=bytes.fromhex(row['query']); y=solve(q)
        return (row['id']==hashlib.sha256(q).hexdigest() and row['family']==FAMILIES[q[0]-1]
            and row['split']==split_for(q[1]) and type(row['answer']) is int and row['answer']==y==oracle(q))
    except (ValueError,TypeError,KeyError,IndexError): return False

def encode(row):
    if not verify(row): raise ValueError('invalid row')
    ids=[1]+[b+9 for b in bytes.fromhex(row['query'])]+[row['answer']+9,2]
    return ids,[-100]*(len(ids)-2)+ids[-2:]

def report(rows):
    if not rows or not all(verify(r) for r in rows) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('invalid corpus')
    lengths=[len(encode(r)[0]) for r in rows]
    return dict(schema='ngc-binary-logic-v1',rows=len(rows),splits=dict(Counter(r['split'] for r in rows)),
        min_tokens=min(lengths),max_tokens=max(lengths),total_tokens=sum(lengths),target_tokens=2*len(rows),
        families=dict(Counter(r['family'] for r in rows)),verified=True)

if __name__=='__main__':
    print(json.dumps(report(generate()),sort_keys=True))
