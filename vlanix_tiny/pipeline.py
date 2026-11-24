"""Offline, bounded paired experiment using reviewed local JSONL."""
import argparse
import hashlib
import json
import math
import time
import resource
from pathlib import Path
import torch
from vlanix_tiny.corpus import prepare
from vlanix_tiny.tokenizer import train_tokenizer
from vlanix_tiny.chunks import chunk_document, BUDGETS
from vlanix_tiny.curriculum import schedules
from vlanix_tiny.training import Trainer, encode_documents
from vlanix_tiny.packing import pack
from vlanix_tiny.pilot_gate import audit_chunks,check_review,review_digest

def write_jsonl(path,rows):
    payload="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

def run(rows,out,budget=128,batch_size=4,seed=42,epochs=1,smoke=False,review_receipt=None):
    if budget not in BUDGETS or not 1<=batch_size<=16 or not 1<=epochs<=5:
        raise ValueError("invalid pilot budget")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    rows=list(rows)
    train,val,corpus=prepare(rows)
    if not train or not val:
        raise ValueError("both splits required")
    hashes={}
    for name,data in (("train",train),("validation",val)):
        hashes[name]=write_jsonl(out/(name+".jsonl"),data)
    tok=train_tokenizer((r["text"] for r in train),out/"tokenizer.json")
    hashes["tokenizer"]=hashlib.sha256((out/"tokenizer.json").read_bytes()).hexdigest()
    reserved=sum(t.startswith("<|reserved_") for t in tok.get_vocab())
    if reserved and not smoke:
        raise ValueError("placeholder tokenizer cannot train in real mode")
    review_id=None
    audit=None
    if not smoke:
        review_id=check_review(rows,review_receipt)
        raw={b:{name:[c for r in data for c in chunk_document(r,tok,b,name)] for name,data in (("train",train),("validation",val))} for b in BUDGETS}
        cleaned,audit=audit_chunks(raw)
        blocked={tuple(x) for x in audit["blocked_documents"]}
        val=[r for r in val if (r["source"],r["document_id"]) not in blocked]
        t=sum(len(tok.encode(r["text"],add_special_tokens=False).ids) for r in train)
        v=sum(len(tok.encode(r["text"],add_special_tokens=False).ids) for r in val)
        if len(val)<10 or v<5000 or not 0.05<=v/(t+v)<=0.20:
            raise ValueError("audited validation below pilot gate")
        audit.update(train_document_tokens=t,validation_document_tokens=v,retained_validation_documents=len(val))
        chunks=cleaned[budget]
        hashes["validation"]=write_jsonl(out/"validation.jsonl",val)
    else:
        chunks={name:[c for r in data for c in chunk_document(r,tok,budget,name)] for name,data in (("train",train),("validation",val))}
    for name,cs in chunks.items():
        hashes[name+"_chunks"]=write_jsonl(out/(name+"-chunks.jsonl"),cs)
    if {c["sha256"] for c in chunks["train"]}&{c["sha256"] for c in chunks["validation"]}:
        raise ValueError("exact chunk overlap between splits")
    tensors={}
    for name,cs in chunks.items():
        x=encode_documents(tok,(c["text"] for c in cs),budget)
        if len(x)!=len(cs) or [int((v!=0).sum()) for v in x]!=[c["token_count"] for c in cs]:
            raise ValueError("chunk/token contract mismatch")
        tensors[name]=x
    a,b,schedule=schedules(chunks["train"],seed,epochs,batch_size)
    if schedule["optimizer_steps"]>2000:
        raise ValueError("pilot step budget exceeded")
    index={c["chunk_id"]:i for i,c in enumerate(chunks["train"])}
    packed=pack([v[v!=0].tolist() for v in tensors["train"]],budget)
    density=dict(unpacked=packed.used_tokens/tensors["train"].numel(),packed=packed.used_tokens/packed.ids.numel(),training_enabled=False)
    report=dict(schema="vlanix-pipeline-v1",smoke_only=smoke,reserved_tokens=reserved,hashes=hashes,corpus=corpus,schedule=schedule,packing=density,arms={},audit=audit,review_id=review_id,input_sha256=review_digest(rows))
    for name,batches in (("random",a),("curriculum",b)):
        hashes[name+"_schedule"]=write_jsonl(out/(name+"-schedule.jsonl"),[dict(step=i,chunk_ids=batch) for i,batch in enumerate(batches)])
        identity=hashlib.sha256(json.dumps([hashes["train_chunks"],hashes["validation_chunks"],hashes["tokenizer"],hashes[name+"_schedule"],seed,budget,batch_size,epochs,smoke],sort_keys=True).encode()).hexdigest()
        trainer=Trainer("cpu",seed,0.001)
        initial=trainer.evaluate(tensors["validation"],batch_size)
        history=[]
        train_start=time.monotonic()
        for batch in batches:
            loss=trainer.step(tensors["train"][[index[k] for k in batch]])
            history.append(dict(step=trainer.steps,tokens=trainer.tokens,train_loss=loss))
        elapsed=max(time.monotonic()-train_start,1e-9)
        hashes[name+"_metrics"]=write_jsonl(out/(name+"-metrics.jsonl"),history)
        final=trainer.evaluate(tensors["validation"],batch_size)
        if trainer.tokens!=schedule["target_tokens"]:
            raise RuntimeError("target token budget mismatch")
        path=out/(name+".pt")
        trainer.save(path,identity)
        restored=Trainer("cpu",seed,0.001)
        restored.restore(path,identity)
        restored_loss=restored.evaluate(tensors["validation"],batch_size)
        ok=restored.steps==trainer.steps and restored.tokens==trainer.tokens and math.isclose(final,restored_loss,abs_tol=1e-6)
        if not ok:
            raise RuntimeError("checkpoint restore mismatch")
        report["arms"][name]=dict(initial_loss=initial,final_loss=final,steps=trainer.steps,tokens=trainer.tokens,restore_ok=ok,train_seconds=elapsed,tokens_per_second=trainer.tokens/elapsed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        del trainer,restored
    report["elapsed_seconds"]=time.monotonic()-start
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--budget",type=int,choices=BUDGETS,default=128)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--review-receipt")
    a=p.parse_args()
    path=Path(a.input)
    if path.stat().st_size>10_000_000:
        p.error("pilot input exceeds 10 MB")
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt=json.loads(Path(a.review_receipt).read_text(encoding="utf-8")) if a.review_receipt else None
    print(json.dumps(run(rows,a.output,a.budget,a.batch_size,a.seed,a.epochs,a.smoke,receipt),indent=2))

if __name__=="__main__":
    main()
