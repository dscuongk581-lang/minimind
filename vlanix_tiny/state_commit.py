"""P0.10C: local durable state publication; one trusted SQLite database."""
import hashlib
import json
from vlanix_tiny.executor_lease import Authority, Rejected, name, digest, integer
from vlanix_tiny.virtual_ram import BLOCK, LIMIT, SCHEMA, sha, root, validate

class StateStore(Authority):
    def __init__(self,path):
        super().__init__(path)
        with self.tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS ram_manifest (root TEXT PRIMARY KEY, body TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS ram_node (id INTEGER PRIMARY KEY, online INTEGER NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS ram_block (node INTEGER NOT NULL, hash TEXT NOT NULL, body BLOB NOT NULL, PRIMARY KEY(node,hash))')
            for n in range(3):
                c.execute('INSERT OR IGNORE INTO ram_node VALUES(?,1)',(n,))

    @staticmethod
    def placement(h):
        n=int(h[:8],16)%3
        return n,(n+1)%3

    def set_online(self,node,online):
        if type(node) is not int or not 0<=node<3 or type(online) is not bool:
            raise ValueError('node status')
        with self.tx() as c:
            c.execute('UPDATE ram_node SET online=? WHERE id=?',(int(online),node))

    @staticmethod
    def online(c):
        return {r[0] for r in c.execute('SELECT id FROM ram_node WHERE online=1')}

    def _verify(self,c,h,require_two=False,materialize=False):
        digest(h)
        row=c.execute('SELECT body FROM ram_manifest WHERE root=?',(h,)).fetchone()
        if row is None: raise Rejected('unknown manifest')
        try:
            m=json.loads(row[0]); validate(m,h)
        except (ValueError,TypeError,KeyError) as e:
            raise Rejected('invalid manifest') from e
        online=self.online(c)
        if require_two and online!={0,1,2}:
            raise Rejected('publish requires all logical nodes')
        payload=hashlib.sha256()
        out=bytearray() if materialize else None
        for i,key in enumerate(m['blocks']):
            length=min(BLOCK,m['size']-i*BLOCK)
            good=[]
            for node in self.placement(key):
                if node not in online: continue
                row=c.execute('SELECT body FROM ram_block WHERE node=? AND hash=?',(node,key)).fetchone()
                if row is not None and isinstance(row[0],bytes) and len(row[0])==length and sha(row[0])==key:
                    good.append(row[0])
            if len(good)<(2 if require_two else 1):
                raise Rejected('insufficient verified replicas')
            payload.update(good[0])
            if out is not None: out.extend(good[0])
        if payload.hexdigest()!=m['sha256']:
            raise Rejected('payload digest')
        return bytes(out) if out is not None else None

    def stage(self,data):
        if not isinstance(data,bytes) or not 0<len(data)<=LIMIT:
            raise ValueError('payload bounds')
        keys=[]
        with self.tx() as c:
            if self.online(c)!={0,1,2}: raise Rejected('stage requires all logical nodes')
            for off in range(0,len(data),BLOCK):
                part=data[off:off+BLOCK]; h=sha(part); keys.append(h)
                for n in self.placement(h):
                    c.execute('INSERT OR IGNORE INTO ram_block VALUES(?,?,?)',(n,h,part))
            m=dict(schema=SCHEMA,block_size=BLOCK,size=len(data),blocks=keys,sha256=sha(data))
            h=root(m)
            c.execute('INSERT OR IGNORE INTO ram_manifest VALUES(?,?)',
                (h,json.dumps(m,sort_keys=True,separators=(',',':'))))
            self._verify(c,h,require_two=True)
            return h

    def register(self,identity,root):
        name(identity); digest(root)
        with self.tx() as c:
            if c.execute('SELECT 1 FROM organism WHERE id=?',(identity,)).fetchone():
                raise Rejected('already registered')
            self._verify(c,root,require_two=True)
            c.execute('INSERT INTO organism VALUES(?,?,0,0,NULL,0)',(identity,root))

    def commit(self,identity,owner,epoch,version,parent,new_root):
        integer(version); digest(parent); digest(new_root)
        with self.tx() as c:
            r=self._check(c,identity,owner,epoch)
            if r['version']!=version or r['root']!=parent: raise Rejected('state CAS')
            if new_root==parent: raise Rejected('unchanged root')
            # Verify both replicas while holding the same transaction as the CAS.
            self._verify(c,new_root,require_two=True)
            integer(version+1)
            c.execute('UPDATE organism SET root=?,version=? WHERE id=?',(new_root,version+1,identity))
            return version+1

    def read_current(self,identity):
        with self.tx() as c:
            row=self.row(c,identity)
            return dict(row),self._verify(c,row['root'],materialize=True)
