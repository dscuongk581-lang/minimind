"""Lossless boundary-aware chunking for already split, reviewed NFC documents."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from vlanix_tiny.tokenizer import load_tokenizer

BUDGETS=(64,128,256,512)
def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def chunk_document(row,tok,budget,split):
    if budget not in BUDGETS or split not in ("train","validation"):
        raise ValueError("invalid budget or split")
    if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ("text","source","document_id","license")):
        raise ValueError("missing provenance")
    text=row["text"]
    if not unicodedata.is_normalized("NFC",text) or "<|" in text:
        raise ValueError("expected clean NFC document")
    limit=budget-2
    tok_hash=sha(tok.to_str())
    doc_hash=sha(text)
    def count(s):
        return len(tok.encode(s,add_special_tokens=False).ids)
    # Candidate boundaries preserve original spaces and punctuation.
    paragraphs={m.end() for m in re.finditer(r"\n\s*\n",text)}
    sentences={m.end() for m in re.finditer(r"[.!?…]+\s+",text)}
    words={m.end() for m in re.finditer(r"\s+",text)}
    start=0
    out=[]
    while start<len(text):
        # Bounded search window avoids quadratic encoding of long documents.
        upper=min(len(text),start+limit*16)
        end=upper
        while count(text[start:end])>limit:
            end=start+(end-start)//2
            if end==start:
                raise ValueError("single character exceeds token budget")
        # Expand by binary search; token counts need not be monotonic,
        # so we always keep a verified fitting prefix, not an optimality claim.
        lo,hi=end,upper
        while lo<hi:
            mid=(lo+hi+1)//2
            if count(text[start:mid])<=limit:
                lo=mid
            else:
                hi=mid-1
        end=lo
        if end<len(text):
            minimum=start+max(1,(end-start)//2)
            for candidates in (paragraphs,sentences,words):
                fits=[p for p in candidates if minimum<=p<=end]
                if fits:
                    candidate=max(fits)
                    if count(text[start:candidate])<=limit:
                        end=candidate
                        break
        piece=text[start:end]
        ids=tok.encode(piece,add_special_tokens=False).ids
        if not ids or len(ids)>limit or tok.decode(ids,skip_special_tokens=False)!=piece:
            raise ValueError("chunk encoding contract violated")
        identity=json.dumps([row["source"],row["document_id"],doc_hash,tok_hash,budget,start,end,split],ensure_ascii=False)
        out.append(dict(text=piece,source=row["source"],document_id=row["document_id"],
            license=row["license"],split=split,chunk_id=sha(identity),sha256=sha(piece),
            document_sha256=doc_hash,tokenizer_sha256=tok_hash,char_start=start,char_end=end,
            token_count=len(ids)+2,budget=budget,index=len(out),schema="vlanix-chunk-v1"))
        start=end
    return out

def main():
    p=argparse.ArgumentParser()
    for name in ("input","output","tokenizer"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--split",choices=["train","validation"],required=True)
    p.add_argument("--budget",type=int,choices=BUDGETS,default=128)
    a=p.parse_args()
    tok=load_tokenizer(a.tokenizer)
    out=Path(a.output)
    tmp=out.with_suffix(out.suffix+".tmp")
    if out.exists():
        p.error("output exists")
    docs=chunks=tokens=0
    with open(a.input,encoding="utf-8") as src,tmp.open("x",encoding="utf-8") as dst:
        try:
            for line in src:
                if not line.strip(): continue
                docs+=1
                for c in chunk_document(json.loads(line),tok,a.budget,a.split):
                    dst.write(json.dumps(c,ensure_ascii=False,sort_keys=True)+"\n")
                    chunks+=1
                    tokens+=c["token_count"]
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    if not chunks:
        tmp.unlink()
        p.error("no chunks")
    tmp.rename(out)
    print(json.dumps(dict(documents=docs,chunks=chunks,tokens=tokens,budget=a.budget,split=a.split)))

if __name__=="__main__":
    main()
