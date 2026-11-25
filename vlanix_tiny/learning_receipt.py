"""P0.10D: trusted local deterministic replay before state publication."""
import hashlib
import json
import tempfile
from pathlib import Path
import torch
from vlanix_tiny.executor_lease import Rejected,name,digest,integer
from vlanix_tiny.state_commit import StateStore
from vlanix_tiny.guarded_session import Session,validate_plan
from vlanix_tiny.binary_homeostasis import drift
from vlanix_tiny.ram_migration import equal

CAP=.075
SCHEMA='ngc-learning-receipt-v1'

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)

def hash_object(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def replay(parent,candidate,cal,plan,identity):
    """Only trusted local PyTorch archives; not an untrusted-file sandbox."""
    rng=torch.get_rng_state()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'parent.pt'; q=Path(tmp)/'candidate.pt'; out=Path(tmp)/'expected.pt'
            p.write_bytes(parent); q.write_bytes(candidate)
            session=Session.restore(p,cal,plan,identity)
            if session.cap!=CAP or session.blocked or session.cursor>=len(plan):
                raise Rejected('learning policy')
            cursor=session.cursor
            event=session.attempt()
            if not event['accepted']: raise Rejected('learning gate rejected')
            session.save(out)
            expected=torch.load(out,map_location='cpu',weights_only=True)
            actual=torch.load(q,map_location='cpu',weights_only=True)
            if not equal(expected,actual): raise Rejected('replay mismatch')
            return dict(plan_digest=session.plan_digest,batch_digest=hash_object(plan[cursor]),
                cursor_before=cursor,cursor_after=session.cursor,cap=CAP,event=event)
    finally:
        torch.set_rng_state(rng)

class LearningStore(StateStore):
    def __init__(self,path):
        super().__init__(path)
        with self.tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS learning_genesis (id TEXT PRIMARY KEY, root TEXT NOT NULL, plan TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS learning_receipt (id TEXT NOT NULL, version INTEGER NOT NULL, digest TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(id,version))')

    def register(self,identity,root,cal,plan):
        name(identity); digest(root)
        plan_hash=validate_plan(cal,plan)
        with self.tx() as c:
            if c.execute('SELECT 1 FROM organism WHERE id=?',(identity,)).fetchone():
                raise Rejected('already registered')
            data=self._verify(c,root,require_two=True,materialize=True)
            rng=torch.get_rng_state()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'genesis.pt'; p.write_bytes(data)
                    s=Session.restore(p,cal,plan,identity)
                    if s.cursor!=0 or s.blocked or s.cap!=CAP or drift(s.state.model,cal,s.ref)>1e-6:
                        raise Rejected('invalid genesis policy')
            finally:
                torch.set_rng_state(rng)
            c.execute('INSERT INTO organism VALUES(?,?,0,0,NULL,0)',(identity,root))
            c.execute('INSERT INTO learning_genesis VALUES(?,?,?)',(identity,root,plan_hash))

    def commit(self,*args,**kwargs):
        raise Rejected('verified learning receipt required')

    def _parent(self,c,identity,owner,epoch,version,parent,plan_hash):
        row=self._check(c,identity,owner,epoch)
        if row['version']!=version or row['root']!=parent: raise Rejected('state CAS')
        g=c.execute('SELECT * FROM learning_genesis WHERE id=?',(identity,)).fetchone()
        if g is None or g['plan']!=plan_hash: raise Rejected('plan binding')
        return g

    def publish(self,identity,owner,epoch,version,parent,new_root,cal,plan):
        integer(version); digest(parent); digest(new_root)
        if parent==new_root: raise Rejected('unchanged root')
        plan_hash=validate_plan(cal,plan)
        with self.tx() as c:
            self._parent(c,identity,owner,epoch,version,parent,plan_hash)
            a=self._verify(c,parent,materialize=True)
            b=self._verify(c,new_root,require_two=True,materialize=True)
        proof=replay(a,b,cal,plan,identity)
        del a,b
        if proof['cursor_before']!=version or proof['cursor_after']!=version+1:
            raise Rejected('cursor/version mismatch')
        with self.tx() as c:
            g=self._parent(c,identity,owner,epoch,version,parent,plan_hash)
            self._verify(c,new_root,require_two=True)
            if version:
                prev=c.execute('SELECT digest,body FROM learning_receipt WHERE id=? AND version=?',(identity,version)).fetchone()
                if prev is None: raise Rejected('missing prior receipt')
                prior=json.loads(prev['body'])
                if hash_object(prior)!=prev['digest'] or prior['root']!=parent:
                    raise Rejected('prior receipt mismatch')
                anchor=prev['digest']
            else:
                if g['root']!=parent: raise Rejected('genesis mismatch')
                anchor=hash_object(dict(identity=identity,root=g['root'],plan=g['plan']))
            integer(version+1)
            receipt=dict(schema=SCHEMA,identity=identity,owner=owner,epoch=epoch,
                version=version+1,parent=parent,root=new_root,previous=anchor,**proof)
            h=hash_object(receipt)
            c.execute('INSERT INTO learning_receipt VALUES(?,?,?,?)',(identity,version+1,h,canonical(receipt)))
            c.execute('UPDATE organism SET root=?,version=? WHERE id=?',(new_root,version+1,identity))
            return dict(receipt=receipt,digest=h)

    def audit(self,identity):
        with self.tx() as c:
            current=self.row(c,identity)
            g=c.execute('SELECT * FROM learning_genesis WHERE id=?',(identity,)).fetchone()
            if g is None: raise Rejected('missing genesis')
            parent=g['root']
            anchor=hash_object(dict(identity=identity,root=parent,plan=g['plan']))
            records=[]
            for i,row in enumerate(c.execute('SELECT * FROM learning_receipt WHERE id=? ORDER BY version',(identity,)),1):
                d=json.loads(row['body'])
                if (row['version']!=i or d['version']!=i or d['identity']!=identity or
                    d['schema']!=SCHEMA or d['parent']!=parent or d['previous']!=anchor or
                    d['plan_digest']!=g['plan'] or d['cursor_before']!=i-1 or d['cursor_after']!=i or
                    d['cap']!=CAP or not d['event']['accepted'] or not 0<=d['event']['proposed']<=CAP or
                    hash_object(d)!=row['digest']):
                    raise Rejected('receipt chain mismatch')
                parent=d['root']; anchor=row['digest']; records.append(d)
            if len(records)!=current['version'] or parent!=current['root']:
                raise Rejected('canonical pointer mismatch')
            return dict(version=len(records),head=anchor,root=parent,records=records)
