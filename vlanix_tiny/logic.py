"""Deterministic English procedural reasoning fixtures; no external data."""
import argparse
import hashlib
import json
import random
from pathlib import Path

SPLITS = ('train', 'validation', 'test')
FAMILIES = ('arithmetic', 'boolean', 'state', 'rules')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def instance(family, split, i):
    k = SPLITS.index(split)
    if family == 'arithmetic':
        return dict(a=1+32*k+i//16, b=1+i%16, op=('-' if i%2 else '+'))
    if family == 'boolean':
        return dict(bits=[(i>>j)&1 for j in range(8)], mode=('pair_or','pair_and','xor_pairs')[k])
    if family == 'state':
        return dict(start=1+i//16, ops=[['+',1+i%16],['*',2]]+[['-',1]]*k)
    if family == 'rules':
        depth=k+2
        return dict(facts=(['P0'] if i&1 else [])+[f'N{j}' for j in range(7) if i&(2<<j)], rules=[[f'P{j}',f'P{j+1}'] for j in range(depth)], query=f'P{depth}')
    raise ValueError('unknown family')


def solve(family, data):
    if family == 'arithmetic':
        a,b,op=data['a'],data['b'],data['op']
        if op not in ('+','-'): raise ValueError('operator')
        return str(a+b if op=='+' else a-b)
    if family == 'boolean':
        bits=data['bits']; mode=data['mode']
        if len(bits)!=8 or any(x not in (0,1) for x in bits): raise ValueError('bits')
        pairs=list(zip(bits[::2],bits[1::2]))
        if mode=='pair_or': result=any(a and b for a,b in pairs)
        elif mode=='pair_and': result=all(a or b for a,b in pairs)
        elif mode=='xor_pairs': result=sum(a!=b for a,b in pairs)%2==1
        else: raise ValueError('mode')
        return 'true' if result else 'false'
    if family == 'state':
        v=data['start']
        for op,n in data['ops']:
            if op=='+': v+=n
            elif op=='-': v-=n
            elif op=='*': v*=n
            else: raise ValueError('operator')
        return str(v)
    if family == 'rules':
        known=set(data['facts'])
        while True:
            new=known|{b for a,b in data['rules'] if a in known}
            if new==known: break
            known=new
        return 'yes' if data['query'] in known else 'unknown'
    raise ValueError('unknown family')


def prompt(family, data):
    if family=='arithmetic': return f"Calculate {data['a']} {data['op']} {data['b']}."
    if family=='boolean':
        pairs=list(zip(data['bits'][::2],data['bits'][1::2]))
        inner,outer={'pair_or':('AND','OR'),'pair_and':('OR','AND'),'xor_pairs':('XOR','XOR')}[data['mode']]
        expr=f' {outer} '.join(f'({a} {inner} {b})' for a,b in pairs)
        return f'Evaluate {expr}. Here 1 means true and 0 means false.'
    if family=='state': return f"Start at {data['start']}. Apply in order: "+', '.join(f'{op}{n}' for op,n in data['ops'])+'. What is the final value?'
    if family=='rules':
        facts=', '.join(data['facts']) or 'none'
        rules='; '.join(f'If {a}, then {b}' for a,b in data['rules'])
        return f"Facts: {facts}. Rules: {rules}. Is {data['query']} entailed? Answer yes or unknown. Missing evidence is not false."
    raise ValueError('unknown family')


def make_row(family, split, i):
    data=instance(family,split,i)
    ident=digest([family,data]); q=prompt(family,data); a=solve(family,data)
    return dict(schema='ngc-logic-v1',family=family,split=split,index=i,instance=data,instance_id=ident,document_id=ident,source='vlanix-logic-v1',license='CC0-1.0',prompt=q,answer=a,text=q+'\nAnswer: '+a)


def verify(row):
    try:
        return row==make_row(row['family'],row['split'],row['index']) and 0<=row['index']<256
    except (KeyError,TypeError,ValueError,IndexError):
        return False


def generate(per_family=128, seed=42):
    if type(per_family) is not int or not 1<=per_family<=256: raise ValueError('per_family must be 1..256')
    rows=[]
    for split in SPLITS:
        for family in FAMILIES:
            ids=list(range(256)); random.Random(f'{seed}:{split}:{family}').shuffle(ids)
            rows.extend(make_row(family,split,i) for i in ids[:per_family])
    if len({r['instance_id'] for r in rows})!=len(rows): raise RuntimeError('instance overlap')
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--per-family',type=int,default=128); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    rows=generate(a.per_family,a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='ngc-logic-v1',seed=a.seed,per_family=a.per_family,files={})
    for split in SPLITS:
        subset=[r for r in rows if r['split']==split]
        payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in subset).encode()
        (out/(split+'.jsonl')).write_bytes(payload)
        manifest['files'][split]=dict(rows=len(subset),sha256=hashlib.sha256(payload).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))

if __name__=='__main__': main()
