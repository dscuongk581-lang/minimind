import unittest
from vlanix_tiny.core_verification import trace, readout_table, score

class CoreVerificationTests(unittest.TestCase):
    def test_trace(self):
        self.assertEqual(trace(1,255,1,lambda q:sum(q[1:])),(11,)*8)
        self.assertIsNone(trace(1,0,0,lambda q:4))
    def test_readout_complete(self):
        table=readout_table(lambda batch:[9+sum(((v-9)&1)<<k for k,v in enumerate(t)) for t in batch])
        self.assertEqual(len(table),65536)
        self.assertEqual(table[(12,)*8],264)
        self.assertEqual(table[(11,)*8],9)
    def test_score_uses_prediction(self):
        rows=[(1,255,1,0),(2,12,3,15)]
        def local(q): return sum(q[1:]) if q[0]==1 else q[1]^q[2]
        result=score(rows,local,lambda t:9)
        self.assertEqual(result,dict(rows=2,correct=1,invalid_trace=0,invalid_byte=0))
        self.assertEqual(score(rows,local,lambda t:0)['invalid_byte'],2)
        self.assertEqual(score(rows,lambda q:-1,lambda t:9)['invalid_trace'],2)
    def test_contract(self):
        with self.assertRaises(ValueError): readout_table(lambda batch:[265]*len(batch))
        with self.assertRaises(ValueError): readout_table(lambda batch:[])
        with self.assertRaises(ValueError): trace(1,256,0,lambda q:0)
