import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from vlanix_tiny.executor_lease import Authority, Rejected

A='a'*64; B='b'*64; C='c'*64

class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'authority.db'
        self.a=Authority(self.path); self.a.register('organism',A)
    def test_transfer_and_stale_writer(self):
        x=self.a.acquire('organism','A',3)
        self.a.commit('organism','A',x,0,A,B)
        y=self.a.transfer('organism','A',x,'B',3)
        self.assertGreater(y,x)
        with self.assertRaises(Rejected): self.a.commit('organism','A',x,1,B,C)
        self.a.commit('organism','B',y,1,B,C)
        self.assertEqual(self.a.inspect('organism')['root'],C)
    def test_expiry_restart(self):
        x=self.a.acquire('organism','A',2)
        self.a.advance(2)
        other=Authority(self.path)
        with self.assertRaises(Rejected): other.commit('organism','A',x,0,A,B)
        y=other.acquire('organism','B',2)
        self.assertEqual(y,x+1)
        other.commit('organism','B',y,0,A,B)
    def test_cas_and_duplicate(self):
        x=self.a.acquire('organism','A',5)
        with self.assertRaises(Rejected): self.a.acquire('organism','B',5)
        with self.assertRaises(Rejected): self.a.commit('organism','A',x,0,C,B)
        self.a.commit('organism','A',x,0,A,B)
        with self.assertRaises(Rejected): self.a.commit('organism','A',x,0,A,B)
        self.assertEqual(self.a.inspect('organism')['version'],1)
    def test_concurrent_acquire(self):
        def attempt(owner):
            try: return Authority(self.path).acquire('organism',owner,5)
            except Rejected: return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(attempt,['A','B']))
        self.assertEqual(sum(x is not None for x in values),1)
    def test_invalid_and_isolated(self):
        self.a.register('other',B)
        x=self.a.acquire('organism','A',2)
        with self.assertRaises(Rejected): self.a.commit('other','A',x,0,B,C)
        with self.assertRaises(ValueError): self.a.advance(-1)
        with self.assertRaises(ValueError): self.a.commit('organism','A',x,0,A,'bad')
        self.assertEqual(self.a.inspect('organism')['root'],A)
