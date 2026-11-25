"""Trusted local SQLite authority; logical ticks, not wall-clock leases."""
import re
import sqlite3
from contextlib import contextmanager

class Rejected(RuntimeError): pass

MAX=(1<<62)-1

def integer(n,minimum=0):
    if type(n) is not int or not minimum<=n<=MAX: raise ValueError('integer bounds')

def name(s):
    if not isinstance(s,str) or not 1<=len(s)<=128: raise ValueError('identity')

def digest(s):
    if not isinstance(s,str) or not re.fullmatch('[0-9a-f]{64}',s): raise ValueError('root')

class Authority:
    def __init__(self,path):
        self.path=str(path)
        with self.tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS clock (id INTEGER PRIMARY KEY CHECK(id=1), tick INTEGER NOT NULL)')
            c.execute('INSERT OR IGNORE INTO clock VALUES(1,0)')
            c.execute('CREATE TABLE IF NOT EXISTS organism (id TEXT PRIMARY KEY, root TEXT NOT NULL, version INTEGER NOT NULL, epoch INTEGER NOT NULL, owner TEXT, expiry INTEGER NOT NULL)')

    @contextmanager
    def tx(self):
        c=sqlite3.connect(self.path,timeout=10,isolation_level=None)
        c.row_factory=sqlite3.Row
        try:
            c.execute('PRAGMA synchronous=FULL')
            c.execute('BEGIN IMMEDIATE')
            yield c
            c.commit()
        except BaseException:
            c.rollback(); raise
        finally: c.close()

    @staticmethod
    def now(c): return c.execute('SELECT tick FROM clock WHERE id=1').fetchone()[0]

    @staticmethod
    def row(c,identity):
        name(identity)
        r=c.execute('SELECT * FROM organism WHERE id=?',(identity,)).fetchone()
        if r is None: raise Rejected('unknown organism')
        return r

    def register(self,identity,root):
        name(identity); digest(root)
        with self.tx() as c:
            if c.execute('SELECT 1 FROM organism WHERE id=?',(identity,)).fetchone(): raise Rejected('already registered')
            c.execute('INSERT INTO organism VALUES(?,?,0,0,NULL,0)',(identity,root))

    def inspect(self,identity):
        with self.tx() as c:
            return dict(self.row(c,identity),tick=self.now(c))

    def advance(self,ticks):
        integer(ticks,1)
        with self.tx() as c:
            n=self.now(c)+ticks; integer(n)
            c.execute('UPDATE clock SET tick=? WHERE id=1',(n,))
            return n

    def _grant(self,c,r,owner,ttl):
        name(owner); integer(ttl,1)
        epoch=r['epoch']+1; expiry=self.now(c)+ttl
        integer(epoch,1); integer(expiry,1)
        c.execute('UPDATE organism SET owner=?,epoch=?,expiry=? WHERE id=?',(owner,epoch,expiry,r['id']))
        return epoch

    def acquire(self,identity,owner,ttl):
        with self.tx() as c:
            r=self.row(c,identity)
            if r['owner'] is not None and self.now(c)<r['expiry']: raise Rejected('lease active')
            return self._grant(c,r,owner,ttl)

    def _check(self,c,identity,owner,epoch):
        name(owner); integer(epoch,1)
        r=self.row(c,identity)
        if r['owner']!=owner or r['epoch']!=epoch or self.now(c)>=r['expiry']:
            raise Rejected('stale or expired lease')
        return r

    def transfer(self,identity,owner,epoch,new_owner,ttl):
        with self.tx() as c:
            r=self._check(c,identity,owner,epoch)
            if new_owner==owner: raise ValueError('same owner')
            return self._grant(c,r,new_owner,ttl)

    def commit(self,identity,owner,epoch,version,parent,new_root):
        integer(version); digest(parent); digest(new_root)
        with self.tx() as c:
            r=self._check(c,identity,owner,epoch)
            if r['version']!=version or r['root']!=parent: raise Rejected('state CAS')
            if new_root==parent: raise Rejected('unchanged root')
            integer(version+1)
            c.execute('UPDATE organism SET root=?,version=? WHERE id=?',(new_root,version+1,identity))
            return version+1
