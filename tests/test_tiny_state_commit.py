import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from vlanix_tiny.state_commit import StateStore
from vlanix_tiny.executor_lease import Rejected
from vlanix_tiny.virtual_ram import BLOCK,sha

class StateCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'state.db'
        self.s=StateStore(self.path)
        self.old=b'initial'*10000; self.a=self.s.stage(self.old)
        self.s.register('one',self.a)
        self.e=self.s.acquire('one','A',5)
    def test_stage_not_publish_and_restart(self):
        b=self.s.stage(b'candidate')
        self.assertEqual(self.s.read_current('one')[1],self.old)
        self.assertEqual(self.s.commit('one','A',self.e,0,self.a,b),1)
        other=StateStore(self.path)
        self.assertEqual(other.read_current('one')[1],b'candidate')
        self.assertEqual(other.inspect('one')['version'],1)
    def test_stale_and_expired(self):
        b=self.s.stage(b'next')
        e=self.s.transfer('one','A',self.e,'B',2)
        with self.assertRaises(Rejected): self.s.commit('one','A',self.e,0,self.a,b)
        self.s.advance(2)
        with self.assertRaises(Rejected): self.s.commit('one','B',e,0,self.a,b)
        self.assertEqual(self.s.read_current('one')[1],self.old)
    def test_corruption_and_degraded_read(self):
        data=b'candidate'; b=self.s.stage(data); key=sha(data)
        n=self.s.placement(key)[0]
        with self.s.tx() as c:
            c.execute('UPDATE ram_block SET body=? WHERE node=? AND hash=?',(b'xxxxxxxxx',n,key))
        with self.assertRaises(Rejected): self.s.commit('one','A',self.e,0,self.a,b)
        self.assertEqual(self.s.inspect('one')['version'],0)
        for n in range(3):
            self.s.set_online(n,False)
            self.assertEqual(self.s.read_current('one')[1],self.old)
            with self.assertRaises(Rejected): self.s.stage(b'other')
            self.s.set_online(n,True)
    def test_missing_root_and_duplicate(self):
        with self.assertRaises(Rejected): self.s.register('bad','f'*64)
        with self.assertRaises(Rejected): self.s.commit('one','A',self.e,0,self.a,'f'*64)
        b=self.s.stage(b'next')
        self.assertEqual(self.s.stage(b'next'),b)
        self.s.commit('one','A',self.e,0,self.a,b)
        with self.assertRaises(Rejected): self.s.commit('one','A',self.e,0,self.a,b)
    def test_concurrent_cas(self):
        roots=[self.s.stage(x) for x in (b'first',b'second')]
        def attempt(h):
            try: return StateStore(self.path).commit('one','A',self.e,0,self.a,h)
            except Rejected: return None
        with ThreadPoolExecutor(max_workers=2) as p:
            results=list(p.map(attempt,roots))
        self.assertEqual(results.count(1),1)
        self.assertEqual(self.s.inspect('one')['root'],roots[results.index(1)])
    def test_failed_transaction_keeps_pointer(self):
        import sqlite3
        b=self.s.stage(b'next')
        with self.s.tx() as c:
            c.execute("CREATE TRIGGER fail_commit BEFORE UPDATE OF root ON organism BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.s.commit('one','A',self.e,0,self.a,b)
        self.assertEqual(StateStore(self.path).read_current('one')[1],self.old)
        with self.s.tx() as c: c.execute('DROP TRIGGER fail_commit')
        self.assertEqual(self.s.commit('one','A',self.e,0,self.a,b),1)
