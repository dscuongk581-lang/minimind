"""Single-device FP32 pretraining baseline. No downloads, no external telemetry."""
import argparse
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
import torch
from vlanix_tiny.config import make_model
from vlanix_tiny.tokenizer import load_tokenizer, read_jsonl

def encode_documents(tok, texts, length):
    if not 4 <= length <= 512:
        raise ValueError("sequence length must be 4..512")
    rows=[]
    for text in texts:
        ids=tok.encode(text,add_special_tokens=False).ids
        if not ids:
            continue
        if any(i<9 for i in ids):
            raise ValueError("unexpected special token in document")
        for start in range(0,len(ids),length-2):
            row=[1]+ids[start:start+length-2]+[2]
            rows.append(row+[0]*(length-len(row)))
    if not rows:
        raise ValueError("empty encoded corpus")
    return torch.tensor(rows,dtype=torch.long)

class Trainer:
    def __init__(self,device="cpu",seed=42,lr=0.001):
        torch.manual_seed(seed)
        self.device=torch.device(device)
        self.model=make_model().to(self.device)
        self.optim=torch.optim.AdamW(self.model.parameters(),lr=lr)
        self.steps=0
        self.tokens=0
        self.seed=seed
        self.lr=lr

    def _loss(self,x):
        x=x.to(self.device)
        labels=x.masked_fill(x==0,-100)
        count=int((labels[:,1:]!=-100).sum().item())
        if not count:
            raise ValueError("no target tokens")
        loss=self.model(x,labels=labels).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        return loss,count

    def step(self,x):
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        loss,count=self._loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,error_if_nonfinite=True)
        self.optim.step()
        self.steps+=1
        self.tokens+=count
        return loss.item()

    @torch.no_grad()
    def evaluate(self,x,batch_size):
        self.model.eval()
        total=count=0
        for batch in x.split(batch_size):
            loss,n=self._loss(batch)
            total+=loss.item()*n
            count+=n
        return total/count

    def save(self,path,identity):
        path=Path(path)
        tmp=path.with_suffix(path.suffix+".tmp")
        torch.save(dict(version=1,identity=identity,model=self.model.state_dict(),
            optimizer=self.optim.state_dict(),steps=self.steps,tokens=self.tokens,
            seed=self.seed,lr=self.lr,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type=="cuda" else None),tmp)
        os.replace(tmp,path)

    def restore(self,path,identity):
        s=torch.load(path,map_location="cpu",weights_only=True)
        if s["version"]!=1 or s["identity"]!=identity or s["seed"]!=self.seed or s["lr"]!=self.lr:
            raise ValueError("checkpoint identity mismatch")
        self.model.load_state_dict(s["model"])
        self.optim.load_state_dict(s["optimizer"])
        self.steps,self.tokens=s["steps"],s["tokens"]
        torch.set_rng_state(s["rng"])
        if self.device.type=="cuda" and s["cuda_rng"] is not None:
            torch.cuda.set_rng_state(s["cuda_rng"],self.device)

def main():
    p=argparse.ArgumentParser()
    for name in ("train","validation","tokenizer","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--length",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--save-every",type=int,default=10)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.save_every)<1 or not math.isfinite(a.lr) or a.lr<=0:
        p.error("invalid training budget")
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    hashes=[hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (a.train,a.validation,a.tokenizer)]
    if hashes[0]==hashes[1]:
        p.error("train and validation files identical")
    identity=hashlib.sha256(json.dumps([hashes,a.length,a.batch_size,a.seed,a.lr],sort_keys=True).encode()).hexdigest()
    tok=load_tokenizer(a.tokenizer)
    if any(t.startswith("<|reserved_") for t in tok.get_vocab()):
        p.error("placeholder tokenizer is not eligible for real training")
    train=encode_documents(tok,read_jsonl(a.train),a.length)
    val=encode_documents(tok,read_jsonl(a.validation),a.length)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=a.resume)
    checkpoint=out/"state.pt"
    trainer=Trainer(a.device,a.seed,a.lr)
    if a.resume:
        trainer.restore(checkpoint,identity)
    start=time.monotonic()
    initial_tokens=trainer.tokens
    if a.device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"step":trainer.steps,"validation_loss":trainer.evaluate(val,a.batch_size)}),flush=True)
    while trainer.steps<a.steps:
        # Stateless sampling gives the same next batch after resume.
        g=torch.Generator().manual_seed(a.seed+trainer.steps)
        idx=torch.randint(len(train),(a.batch_size,),generator=g)
        loss=trainer.step(train[idx])
        if trainer.steps%a.save_every==0 or trainer.steps==a.steps:
            trainer.save(checkpoint,identity)
            metrics=dict(step=trainer.steps,train_loss=loss,validation_loss=trainer.evaluate(val,a.batch_size),
                tokens=trainer.tokens,tokens_per_second=(trainer.tokens-initial_tokens)/max(time.monotonic()-start,1e-9),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=="cuda" else 0)
            print(json.dumps(metrics),flush=True)
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics)+"\n")

if __name__=="__main__":
    main()
