import unittest
from vlanix_tiny.core_consolidation import third_frames, validate_parent

class CoreConsolidationTests(unittest.TestCase):
    def fixture(self):
        return dict(seed=42,status='completed',local_correct=32,finite_gate=False,
            checkpoint_sha256='a'*64,test_rows_used=832,test_cached={'rows':832,'correct':832},
            exhaustive={k:{'rows':65536,'correct':65536} for k in ('add','xor','and','or')})
    def test_third(self):
        train=[(b,c) for b in range(256) for c in range(15)]
        fresh=[(b,c) for b in range(256) for c in range(15,30)]
        held=[(b,c) for b in range(256) for c in range(30,46)]
        out=third_frames(train,fresh,held,42)
        self.assertEqual(len(set(out)),3840)
        self.assertEqual([b for b,c in out],[b for b,c in train])
        self.assertFalse(set(out)&(set(train)|set(fresh)|set(held)))
        self.assertEqual(out,third_frames(train,fresh,held,42))
    def test_overlap(self):
        with self.assertRaises(ValueError): third_frames([(0,0)],[(0,1)],[(0,1)],42)
    def test_parent(self):
        row=self.fixture(); self.assertIsNone(validate_parent(row,42,'a'*64))
        row['finite_gate']=True; self.assertIsNone(validate_parent(row,42,'a'*64))
    def test_reject(self):
        for key,value in [('seed',43),('checkpoint_sha256','b'*64),('local_correct',31),('test_cached',{'rows':832,'correct':831}),('finite_gate',None),('exhaustive',{})]:
            row=self.fixture(); row[key]=value
            with self.assertRaises(ValueError): validate_parent(row,42,'a'*64)
