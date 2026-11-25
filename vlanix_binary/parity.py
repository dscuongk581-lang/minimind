"""Independent PyTorch vs Rust forward parity for VLXEMB01."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

V, H, B = 265, 256, 8
HEADER = b"VLXEMB01" + struct.pack("<II", V, H)

def floats(t):
    values=t.detach().cpu().contiguous().flatten().tolist()
    return struct.pack("<" + "f"*len(values), *values)

def read_f32(path,shape):
    raw=path.read_bytes()
    count=1
    for n in shape: count*=n
    if len(raw)!=4*count:
        raise AssertionError(f"wrong output length {path.name}")
    return torch.tensor(struct.unpack("<"+"f"*count,raw),dtype=torch.float32).reshape(shape)

def compare(executor,root,kind):
    g=torch.Generator().manual_seed(42)
    if kind=="dyadic":
        w=(torch.arange(V*H).reshape(V,H)%17-8).float()/32
        h=(torch.arange(B*H).reshape(B,H)%13-6).float()/16
    elif kind=="random":
        w=torch.randn(V,H,generator=g)*.125
        h=torch.randn(B,H,generator=g)*.5
    else:
        raise ValueError(kind)
    source=root/"input"; output=root/"output"
    source.mkdir(parents=True); output.mkdir()
    snapshot=HEADER+floats(w)
    (source/"embedding.bin").write_bytes(snapshot)
    (source/"hidden.bin").write_bytes(floats(h))
    payload=bytes(range(256))
    (source/"payload.bin").write_bytes(payload)
    subprocess.run([str(executor),str(source),str(output)],check=True,timeout=30)
    assert (output/"embedding.bin").read_bytes()==snapshot
    assert (output/"payload.bin").read_bytes()==payload
    assert (output/"ids.bin").read_bytes()==struct.pack("<258H",1,*range(9,265),2)
    expected_rows=F.embedding(torch.arange(V),w)
    assert (output/"rows.bin").read_bytes()==floats(expected_rows)
    actual=read_f32(output/"logits.bin",(B,V))
    expected=F.linear(h,w)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-5)
    if kind=="dyadic":
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Independently verify a write to the shared matrix changes lookup and logits.
    w[9,0]+=.25
    mutated=HEADER+floats(w)
    assert (output/"mutated.bin").read_bytes()==mutated
    assert (output/"mutated_row.bin").read_bytes()==floats(F.embedding(torch.tensor([9]),w))
    new=read_f32(output/"mutated_logits.bin",(B,V))
    torch.testing.assert_close(new,F.linear(h,w),rtol=1e-5,atol=2e-5)
    torch.testing.assert_close(new[:,1:9],actual[:,1:9],rtol=0,atol=0)
    return dict(case=kind,bytes_roundtrip=256,rows_exact=V,logits_checked=B*V,
        mutation_logits_checked=B*V,snapshot_bytes=len(snapshot),
        sha256=hashlib.sha256(snapshot).hexdigest(),
        max_abs_error=(actual-expected).abs().max().item(),
        mutated_max_abs_error=(new-F.linear(h,w)).abs().max().item())

def main():
    if len(sys.argv)!=2: raise SystemExit("usage: parity.py /path/to/parity-executor")
    torch.set_num_threads(2)
    executor=Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory() as d:
        cases=[compare(executor,Path(d)/kind,kind) for kind in ("dyadic","random")]
    print(json.dumps(dict(schema="ngc-binary-parity-v1",cases=cases,torch_version=torch.__version__,
        atol=2e-5,rtol=1e-5,status="PASS"),sort_keys=True))

if __name__=="__main__":
    main()
