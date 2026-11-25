"""Real CPU replay, forged candidate, stale writer and hard-cap rejection."""
import hashlib
import json
import tempfile
from pathlib import Path
import torch
from vlanix_tiny.binary_logic import generate,FAMILIES
from vlanix_tiny.binary_state import State
from vlanix_tiny.binary_homeostasis import answer_step
from vlanix_tiny.guarded_session import Session
from vlanix_tiny.learning_receipt import LearningStore,CAP
from vlanix_tiny.executor_lease import Rejected

def reject(fn):
    try: fn()
    except Rejected: return
    raise AssertionError('invalid publication accepted')

def run():
    torch.manual_seed(42)
    rows=generate(16,42)
    a=[r for r in rows if r['split']=='train' and r['family'] in FAMILIES[:3]]
    b=[r for r in rows if r['split']=='train' and r['family'] in FAMILIES[3:]]
    warm=[r for f in FAMILIES[:3] for r in [x for x in a if x['family']==f][:12]]
    cal=[r for f in FAMILIES[:3] for r in [x for x in a if x['family']==f][12:]]
    plan=[b[i:i+8] for i in range(0,len(b),8)]
    s=State()
    for i in range(0,len(warm),12): answer_step(s,warm[i:i+12])
    s.grow(2,32)
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    session=Session(s,cal,plan,identity)
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp); p=d/'state.pt'; fake=d/'fake.pt'; db=d/'authority.db'
        store=LearningStore(db)
        session.save(p); h0=store.stage(p.read_bytes()); store.register(identity,h0,cal,plan)
        e1=store.acquire(identity,'A',5)
        event=session.attempt(); assert event['accepted']
        session.save(p); h1=store.stage(p.read_bytes())
        # Valid archive with one modified weight, same cursor and plan.
        data=torch.load(p,weights_only=True)
        w=data['model']['model.embed_tokens.weight']; w[9,0]+=.125
        torch.save(data,fake); bad=store.stage(fake.read_bytes())
        del data,w
        reject(lambda:store.publish(identity,'A',e1,0,h0,bad,cal,plan))
        reject(lambda:store.commit(identity,'A',e1,0,h0,h1))
        assert store.audit(identity)['version']==0
        e2=store.transfer(identity,'A',e1,'B',5)
        reject(lambda:store.publish(identity,'A',e1,0,h0,h1,cal,plan))
        receipt=store.publish(identity,'B',e2,0,h0,h1,cal,plan)
        assert receipt['receipt']['event']==event
        store=LearningStore(db)
        audit=store.audit(identity)
        assert audit['version']==1 and audit['head']==receipt['digest']
        rejected=session.attempt(); assert not rejected['accepted']
        session.save(p); h2=store.stage(p.read_bytes())
        reject(lambda:store.publish(identity,'B',e2,1,h1,h2,cal,plan))
        assert store.audit(identity)==audit
        with store.tx() as c:
            c.execute("UPDATE learning_receipt SET digest=? WHERE id=? AND version=1",('0'*64,identity))
        reject(lambda:store.audit(identity))
        return dict(schema='ngc-learning-receipt-probe-v1',status='PASS',version=1,epoch=e2,
            accepted_kl=event['proposed'],rejected_kl=rejected['proposed'],cap=CAP,
            receipt_digest=receipt['digest'],forged_rejected=True,stale_rejected=True,
            bypass_rejected=True,overcap_rejected=True,audit_reopened=True,tamper_detected=True,
            scope='trusted local CPU replay',growth_authorized=False)
if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(run(),sort_keys=True))
