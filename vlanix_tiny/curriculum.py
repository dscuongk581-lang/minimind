"""Reproducible order-only curriculum experiment for fixed chunks."""
import argparse
import hashlib
import json
import random
from pathlib import Path

def schedules(rows,seed=42,epochs=1,batch_size=4):
    if not 1<=epochs<=100 or not 1<=batch_size<=1024:
        raise ValueError("invalid epoch or batch budget")
    lengths={}
    for row in rows:
        key,n=row.get("chunk_id"),row.get("token_count")
        if row.get("split")!="train" or not isinstance(key,str) or not key or key in lengths:
            raise ValueError("train-only unique chunk IDs required")
        if type(n) is not int or not 3<=n<=512:
            raise ValueError("invalid token count")
        lengths[key]=n
    if not lengths:
        raise ValueError("empty curriculum")
    baseline=[]; curriculum=[]
    for epoch in range(epochs):
        ids=sorted(lengths)
        random.Random(seed+epoch).shuffle(ids)
        ordered=sorted(ids,key=lengths.__getitem__)
        baseline.extend(ids[i:i+batch_size] for i in range(0,len(ids),batch_size))
        curriculum.extend(ordered[i:i+batch_size] for i in range(0,len(ids),batch_size))
    manifest=dict(schema="vlanix-curriculum-v1",seed=seed,epochs=epochs,batch_size=batch_size,
        unique_chunks=len(lengths),optimizer_steps=len(baseline),
        target_tokens=epochs*sum(n-1 for n in lengths.values()),
        input_tokens=epochs*sum(lengths.values()),
        chunk_index_sha256=hashlib.sha256(json.dumps(sorted(lengths.items())).encode()).hexdigest(),
        difficulty_proxy="token_length",packing=False)
    return baseline,curriculum,manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args()
    source=Path(a.input)
    raw=source.read_bytes()
    rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
    baseline,curriculum,m=schedules(rows,a.seed,a.epochs,a.batch_size)
    m["input_sha256"]=hashlib.sha256(raw).hexdigest()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    for name,batches in (("random",baseline),("curriculum",curriculum)):
        payload="".join(json.dumps({"step":i,"chunk_ids":b})+"\n" for i,b in enumerate(batches)).encode()
        (out/(name+".jsonl")).write_bytes(payload)
        m[name+"_sha256"]=hashlib.sha256(payload).hexdigest()
    (out/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps(m))

if __name__=="__main__":
    main()
