"""Bounded pilot ingestion, no automatic license approval or remote code."""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
from pathlib import Path
import re

@dataclass(frozen=True)
class Source:
    repo: str
    config: str
    revision: str
    license: str
    review_id: str
    def __post_init__(self):
        if (self.repo,self.config,self.license)!=("wikimedia/wikipedia","20231101.vi","CC-BY-SA-3.0"):
            raise ValueError("source not enabled for pilot")
        if not re.fullmatch(r"[0-9a-f]{40}",self.revision) or not self.review_id.strip():
            raise ValueError("immutable revision and human review reference required")

def collect(rows,source,max_rows,max_bytes):
    if max_rows<1 or max_bytes<1:
        raise ValueError("positive budgets required")
    out=[]
    seen=set()
    scanned=rejected=used=0
    stop="source_exhausted_or_row_budget"
    for row in itertools.islice(rows,max_rows):
        scanned+=1
        if not isinstance(row,dict):
            rejected+=1
            continue
        text,doc,url=row.get("text"),row.get("id"),row.get("url")
        if not isinstance(text,str) or not text.strip() or not isinstance(doc,(str,int)) or not str(doc).strip() or not isinstance(url,str) or not url.startswith("https://vi.wikipedia.org/") or str(doc) in seen:
            rejected+=1
            continue
        record=dict(text=text,document_id=str(doc),source=f"https://huggingface.co/datasets/{source.repo}/tree/{source.revision}/{source.config}",license=source.license,source_url=url,review_id=source.review_id)
        n=len((json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n").encode("utf-8"))
        if used+n>max_bytes:
            stop="byte_budget"
            break
        seen.add(str(doc))
        out.append(record)
        used+=n
    return out,dict(source=asdict(source),scanned=scanned,rejected=rejected,accepted=len(out),output_bytes=used,stop=stop)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",required=True)
    p.add_argument("--review-id",required=True,help="reference to completed human provenance/license review")
    p.add_argument("--output",required=True)
    p.add_argument("--max-rows",type=int,default=1000)
    p.add_argument("--max-bytes",type=int,default=10_000_000)
    a=p.parse_args()
    if not 1<=a.max_rows<=100000 or not 1<=a.max_bytes<=100_000_000:
        p.error("pilot budgets exceeded")
    source=Source("wikimedia/wikipedia","20231101.vi",a.revision,"CC-BY-SA-3.0",a.review_id)
    from datasets import load_dataset
    rows=load_dataset(source.repo,source.config,revision=source.revision,split="train",streaming=True,trust_remote_code=False)
    out,m=collect(rows,source,a.max_rows,a.max_bytes)
    if not out:
        p.error("no accepted rows")
    dest=Path(a.output)
    dest.mkdir(parents=True,exist_ok=False)
    payload="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in out).encode("utf-8")
    (dest/"raw.jsonl").write_bytes(payload)
    m["sha256"]=hashlib.sha256(payload).hexdigest()
    (dest/"manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(m))

if __name__=="__main__":
    main()
