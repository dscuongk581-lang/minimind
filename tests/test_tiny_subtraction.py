import unittest
from vlanix_tiny.subtraction import SUB, frames, target, trace, score

class SubtractionTests(unittest.TestCase):
    def test_table(self):
        self.assertEqual(SUB,10)
        self.assertEqual(len(set(frames())),8)
        self.assertEqual([target(q) for q in frames()],[0,3,3,2,1,0,0,3])
    def test_exhaustive_reference(self):
        result=score(target,lambda t:9+sum(((v-9)&1)<<k for k,v in enumerate(t)))
        self.assertEqual(result,dict(rows=65536,correct=65536,invalid_trace=0,invalid_byte=0))
    def test_feedback(self):
        seen=[]
        def predict(q):
            seen.append(q); return 3 if len(seen)==1 else 0
        self.assertEqual(trace(0,0,predict),(12,9,9,9,9,9,9,9))
        self.assertEqual(seen[1],(SUB,0,0,1))
        self.assertEqual(seen[2],(SUB,0,0,0))
    def test_invalid_and_wrong(self):
        self.assertIsNone(trace(0,0,lambda q:4))
        with self.assertRaises(ValueError): trace(256,0,target)
        with self.assertRaises(ValueError): target((SUB,0,0,2))
        self.assertEqual(score(target,lambda t:9)['correct'],256)
        self.assertEqual(score(target,lambda t:0)['invalid_byte'],65536)
