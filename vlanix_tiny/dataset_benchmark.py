"""Bounded multi-shard source screening and chunk geometry comparison."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
from vlanix_tiny.preflight import REPO,CONFIG,REVISION
from vlanix_tiny.corpus import prepare
from vlanix_tiny.tokenizer import train_tokenizer
from vlanix_tiny.chunks import chunk_document,BUDGETS

def sample_shards(streams,per_shard=128,max_bytes=2_000_000):
    if len(streams)!=4 or not 1<=per_shard<=256 or not 4<=max_bytes<=4_000_000:
        raise ValueError("invalid sampling budget")
    rows=[]; seen=set(); stats=[]
    for i,stream in enumerate(streams):
        s=dict(shard=i,scanned=0,accepted=0,length_rejected=0,schema_rejected=0,duplicate=0,byte_rejected=0,bytes=0)
        for row in itertools.islice(stream,per_shard):
            s["scanned"]+=1
            if not isinstance(row,dict):
                s["schema_rejected"]+=1
                continue
            t,doc,url=row.get("text"),row.get("id"),row.get("url")
            if not isinstance(t,str) or not isinstance(doc,(str,int)) or not str(doc).strip() or not isinstance(url,str) or not url.startswith("https://vi.wikipedia.org/"):
                s["schema_rejected"]+=1
                continue
            if not 12<=len(t.strip())<=12000:
                s["length_rejected"]+=1
                continue
            if str(doc) in seen:
                s["duplicate"]+=1
                continue
            record=dict(text=t,document_id=str(doc),source=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{CONFIG}",license="CC-BY-SA-3.0",source_url=url)
            n=len((json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n").encode())
            if s["bytes"]+n>max_bytes//4:
                s["byte_rejected"]+=1
                continue
            rows.append(record);seen.add(str(doc))
            s["accepted"]+=1;s["bytes"]+=n
        stats.append(s)
    return rows,stats

def density(lengths,budget):
    lengths=list(lengths)
    if not lengths or any(not 3<=n<=budget for n in lengths):
        raise ValueError("invalid chunk lengths")
    rows=1;used=0
    for n in lengths:
        if used+n>budget:
            rows+=1;used=0
        used+=n
    total=sum(lengths)
    return dict(tokens=total,unpacked_rows=len(lengths),packed_rows=rows,unpacked_density=total/(len(lengths)*budget),packed_density=total/(rows*budget))

def benchmark(streams,workdir,per_shard=128,max_bytes=2_000_000):
    rows,sampling=sample_shards(streams,per_shard,max_bytes)
    train,val,corpus=prepare(rows)
    if not train or not val:
        raise ValueError("both document splits required")
    report=dict(schema="vlanix-dataset-benchmark-v1",revision=REVISION,sampling=sampling,corpus=corpus,budgets={},training_authorized=False)
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        path=Path(d)/"tokenizer.json"
        tok=train_tokenizer((r["text"] for r in train),path)
        report["reserved_tokens"]=sum(t.startswith("<|reserved_") for t in tok.get_vocab())
        report["tokenizer_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        report["document_tokens"]={name:sum(len(tok.encode(r["text"],add_special_tokens=False).ids) for r in rs) for name,rs in (("train",train),("validation",val))}
        for budget in BUDGETS:
            cs={name:[c for r in rs for c in chunk_document(r,tok,budget,name)] for name,rs in (("train",train),("validation",val))}
            result={name:density((c["token_count"] for c in chunks),budget) for name,chunks in cs.items()}
            for name,chunks in cs.items():
                result[name]["target_tokens"]=sum(c["token_count"]-1 for c in chunks)
            result["exact_chunk_overlap"]=len({c["sha256"] for c in cs["train"]}&{c["sha256"] for c in cs["validation"]})
            report["budgets"][str(budget)]=result
    v=report["document_tokens"]["validation"];t=report["document_tokens"]["train"]
    checks=dict(vocab_full=report["reserved_tokens"]==0,min_validation_docs=len(val)>=10,min_validation_tokens=v>=5000,validation_fraction=0.05<=v/(v+t)<=0.20,all_shards=all(s["accepted"]>0 for s in sampling),no_exact_chunk_overlap=all(x["exact_chunk_overlap"]==0 for x in report["budgets"].values()))
    report["gate"]=dict(checks=checks,ready=all(checks.values()))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-shard",type=int,default=128)
    p.add_argument("--max-bytes",type=int,default=2_000_000)
    a=p.parse_args()
    from datasets import load_dataset
    streams=[load_dataset("parquet",data_files={"train":f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{CONFIG}/train-{i:05d}-of-00004.parquet"},split="train",streaming=True) for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(benchmark(streams,d,a.per_shard,a.max_bytes),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
