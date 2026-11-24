"""Research-only source screening. No model training or persisted raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from vlanix_tiny.ingestion import Source, collect
from vlanix_tiny.corpus import prepare
from vlanix_tiny.tokenizer import train_tokenizer
from vlanix_tiny.chunks import chunk_document, BUDGETS

REVISION="859a94672c6c0a0cfd1c19c5506d02e0e4604cda"
REPO="wikimedia/wikipedia"
CONFIG="20231101.vi"
SHARD=f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{CONFIG}/train-00000-of-00004.parquet"
def inspect_rows(rows,workdir,max_rows=128,max_bytes=1_000_000,budget=128):
    if not 1<=max_rows<=512 or not 1<=max_bytes<=2_000_000 or budget not in BUDGETS:
        raise ValueError("preflight budget exceeded")
    # Preliminary agent screening is not the human review required for real training.
    source=Source(REPO,CONFIG,REVISION,"CC-BY-SA-3.0","agent-screening-only-not-training-approval")
    records,ing=collect(rows,source,max_rows,max_bytes)
    train,val,corpus=prepare(records)
    if not train or not val:
        raise ValueError("preflight requires both document splits")
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        path=Path(d)/"tokenizer.json"
        tok=train_tokenizer((r["text"] for r in train),path)
        reserved=sum(t.startswith("<|reserved_") for t in tok.get_vocab())
        token_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        chunks={name:[c for row in rs for c in chunk_document(row,tok,budget,name)] for name,rs in (("train",train),("validation",val))}
        overlap=len({c["sha256"] for c in chunks["train"]}&{c["sha256"] for c in chunks["validation"]})
        stats={k:dict(chunks=len(v),input_tokens=sum(c["token_count"] for c in v),target_tokens=sum(c["token_count"]-1 for c in v)) for k,v in chunks.items()}
    return dict(schema="vlanix-source-preflight-v1",source=dict(repo=REPO,config=CONFIG,revision=REVISION,shard=SHARD,license_declared="CC-BY-SA-3.0/GFDL"),
        ingestion={k:v for k,v in ing.items() if k!="source"},corpus=corpus,train_documents=len(train),validation_documents=len(val),
        tokenizer_sha256=token_hash,reserved_tokens=reserved,vocab_size=4096,exact_chunk_overlap=overlap,budget=budget,stats=stats,
        training_authorized=False,review_status="preliminary_agent_screening")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-rows",type=int,default=128)
    p.add_argument("--max-bytes",type=int,default=1_000_000)
    p.add_argument("--budget",type=int,choices=BUDGETS,default=128)
    a=p.parse_args()
    from datasets import load_dataset
    rows=load_dataset("parquet",data_files={"train":SHARD},split="train",streaming=True)
    with tempfile.TemporaryDirectory() as d:
        report=inspect_rows(rows,d,a.max_rows,a.max_bytes,a.budget)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
