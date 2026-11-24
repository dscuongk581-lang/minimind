"""Small offline corpus gate. Exact duplicates only; no claim of PII completeness."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0", "MIT", "Apache-2.0"}
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def prepare(rows, seed="tiny-p03", validation_percent=10):
    if not 0 < validation_percent < 100:
        raise ValueError("invalid split")
    rejected=Counter()
    valid=[]
    parent={}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        parent[max(a,b)]=min(a,b)
    for r in rows:
        if not isinstance(r,dict) or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ("text","document_id","source","license")):
            rejected["schema"]+=1
            continue
        if r["license"] not in LICENSES:
            rejected["license"]+=1
            continue
        s=unicodedata.normalize("NFC",r["text"]).strip()
        if len(s)<12 or len(s)>100000:
            rejected["length"]+=1
            continue
        if "<|" in s or any(unicodedata.category(c) in {"Cc","Cs"} and c not in "\n\t\r" for c in s):
            rejected["control"]+=1
            continue
        if EMAIL.search(s):
            rejected["email"]+=1
            continue
        doc=json.dumps([r["source"],r["document_id"]],ensure_ascii=False)
        parent.setdefault(doc,doc)
        valid.append((digest(s),doc,dict(text=s,source=r["source"],document_id=r["document_id"],license=r["license"])))
    valid.sort(key=lambda x:(x[0],x[1],x[2]["license"]))
    seen={}
    for h,doc,r in valid:
        if h in seen:
            union(doc,seen[h])
        else:
            seen[h]=doc
    train,validation=[],[]
    emitted=set()
    for h,doc,r in valid:
        if h in emitted:
            continue
        emitted.add(h)
        bucket=int(digest(seed+":"+root(doc))[:16],16)%100
        (validation if bucket<validation_percent else train).append(r)
    manifest=dict(train=len(train),validation=len(validation),duplicates=len(valid)-len(emitted),rejected=dict(sorted(rejected.items())),seed=seed,validation_percent=validation_percent)
    return train,validation,manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    with open(a.input,encoding="utf-8") as f:
        train,val,m=prepare(json.loads(line) for line in f if line.strip())
    if not train or not val:
        raise SystemExit("both train and validation must be nonempty")
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in (("train",train),("validation",val)):
        content="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows)
        (out/(name+".jsonl")).write_text(content,encoding="utf-8")
        m[name+"_sha256"]=digest(content)
    (out/"manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(m))

if __name__=="__main__":
    main()
