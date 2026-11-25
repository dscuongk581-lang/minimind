"""P0.10A: reconstruct checkpoint from RAM replicas and resume in fresh CPU process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from vlanix_tiny.binary_logic import generate
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_homeostasis import answer_step
from vlanix_tiny.guarded_session import Session
from vlanix_tiny.virtual_ram import VirtualRAM,BLOCK

def equal(a,b):
    if type(a)!=type(b): return False
    if isinstance(a,torch.Tensor):
        return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def worker(path,plan_path,out):
    cal,plan,identity=json.loads(Path(plan_path).read_text())
    s=Session.restore(path,cal,plan,identity)
    event=s.attempt()
    s.save(out)
    print(json.dumps(dict(pid=os.getpid(),event=event),sort_keys=True))

def run():
    torch.manual_seed(42)
    rows=[r for r in generate(4,42) if r['split']=='train']
    s=State(); answer_step(s,rows[:4]); s.grow(2,32)
    cal=rows[4:8]; plan=[rows[8:12],rows[12:16]]
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    # Deliberately permissive cap for migration parity, not a safety claim.
    session=Session(s,cal,plan,identity,cap=10.)
    assert session.attempt()['accepted']
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); source=d/'source.pt'; rebuilt=d/'rebuilt.pt'
        plan_path=d/'plan.json'; remote=d/'worker.pt'; local=d/'local.pt'
        session.save(source)
        plan_path.write_text(json.dumps([cal,plan,identity]))
        data=source.read_bytes()
        ram=VirtualRAM(); manifest,digest=ram.put(data)
        # Remove an entire logical node; erase its map, not just hide it.
        ram.online.remove(0); ram.nodes[0].clear()
        recovered=ram.get(manifest,digest)
        assert recovered==data
        rebuilt.write_bytes(recovered)
        event=session.attempt(); assert event['accepted']
        session.save(local)
        cp=subprocess.run([sys.executable,'-m','vlanix_tiny.ram_migration','--worker',
            str(rebuilt),str(plan_path),str(remote)],check=True,capture_output=True,text=True,timeout=120)
        other=json.loads(cp.stdout)
        assert other['pid']!=os.getpid() and other['event']==event
        a=torch.load(local,weights_only=True,map_location='cpu')
        b=torch.load(remote,weights_only=True,map_location='cpu')
        assert equal(a,b), 'resumed checkpoint mismatch'
        return dict(schema='ngc-ram-migration-v1',status='PASS',
            bytes=len(data),block_size=BLOCK,blocks=len(manifest['blocks']),
            unique_blocks=len(set(manifest['blocks'])),manifest_root=digest,
            source_sha256=manifest['sha256'],logical_nodes=3,replication=2,
            lost_nodes=1,byte_exact=True,distinct_process=True,
            next_update_exact=True,optimizer_exact=True,pi_exact=True,rng_exact=True,
            cursor=a['runtime']['cursor'],growth_authorized=False,
            cap=10.,scope='local simulation, same CPU host')

if __name__=='__main__':
    torch.set_num_threads(2)
    if len(sys.argv)==5 and sys.argv[1]=='--worker':
        worker(*sys.argv[2:])
    elif len(sys.argv)==1:
        print(json.dumps(run(),sort_keys=True))
    else: raise SystemExit('invalid arguments')
