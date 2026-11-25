"""Real guarded checkpoint handoff through local durable logical replicas."""
import hashlib
import json
import tempfile
from pathlib import Path
import torch
from vlanix_tiny.binary_logic import generate
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_homeostasis import answer_step
from vlanix_tiny.guarded_session import Session
from vlanix_tiny.ram_migration import equal
from vlanix_tiny.state_commit import StateStore
from vlanix_tiny.executor_lease import Rejected

def run():
    torch.manual_seed(42)
    rows=[r for r in generate(4,42) if r['split']=='train']
    s=State(); answer_step(s,rows[:4]); s.grow(2,32)
    cal=rows[4:8]; plan=[rows[8:12],rows[12:16]]
    ident=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    a=Session(s,cal,plan,ident,cap=10.)
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp); p=d/'state.pt'; q=d/'worker.pt'
        store=StateStore(d/'authority.db')
        a.save(p); h0=store.stage(p.read_bytes()); store.register(ident,h0)
        e1=store.acquire(ident,'A',5)
        assert a.attempt()['accepted']
        a.save(p); h1=store.stage(p.read_bytes())
        assert store.commit(ident,'A',e1,0,h0,h1)==1
        e2=store.transfer(ident,'A',e1,'B',5)
        store=StateStore(d/'authority.db')
        row,data=store.read_current(ident)
        assert row['root']==h1 and row['owner']=='B' and row['epoch']==e2
        q.write_bytes(data); del data
        b=Session.restore(q,cal,plan,ident)
        ea=a.attempt(); assert ea['accepted']
        a.save(p); stale=store.stage(p.read_bytes())
        try: store.commit(ident,'A',e1,1,h1,stale)
        except Rejected: pass
        else: raise AssertionError('stale writer committed')
        assert store.inspect(ident)['root']==h1
        eb=b.attempt(); assert eb==ea
        b.save(q)
        assert equal(torch.load(p,weights_only=True),torch.load(q,weights_only=True))
        h2=store.stage(q.read_bytes())
        assert store.commit(ident,'B',e2,1,h1,h2)==2
        store.set_online(0,False)
        row,data=store.read_current(ident)
        assert row['root']==h2 and row['version']==2 and data==q.read_bytes()
        return dict(schema='ngc-state-commit-v1',status='PASS',version=row['version'],
            epoch=row['epoch'],cursor=b.cursor,checkpoint_bytes=len(data),
            stale_rejected=True,next_update_exact=True,degraded_read_exact=True,
            logical_replicas=2,logical_nodes=3,scope='one local SQLite database',
            growth_authorized=False,cap=10.)
if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(run(),sort_keys=True))
