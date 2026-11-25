import unittest
from vlanix_tiny.trace_protocol import prompt, targets, rollout, assess

class TraceTests(unittest.TestCase):
    def test_carry(self):
        q=bytes([1,255,1])
        self.assertEqual(targets(q),[11]*8+[9])
        self.assertEqual(len(prompt(q)),18)
    def test_feedback(self):
        p=prompt(bytes([1,0,0])); seen=[]
        def predict(batch):
            seen.append([list(x) for x in batch]); return [11 if len(seen)==1 else 9]
        out=rollout([p],predict,9)
        self.assertEqual(seen[1][0],p+[11])
        self.assertEqual(seen[-1][0],p+[11]+[9]*7)
        self.assertEqual(out,[[11]+[9]*8])
        self.assertFalse(assess(bytes([1,0,0]),out[0])['trace_exact'])
    def test_score(self):
        q=bytes([2,13,6]); t=targets(q)
        self.assertEqual(t[-1],20)
        self.assertEqual(assess(q,t),dict(answer_exact=True,trace_exact=True,consistent=True,valid=True))
        t[0]=0
        self.assertFalse(assess(q,t)['valid'])
        self.assertTrue(assess(q,t)['answer_exact'])
    def test_bad_contract(self):
        with self.assertRaises(ValueError): prompt(bytes([3,0,0]))
        with self.assertRaises(ValueError): rollout([[1]],lambda b:[],9)
        with self.assertRaises(ValueError): rollout([[1]],lambda b:[265],9)
        with self.assertRaises(ValueError): assess(bytes([1,0,0]),[9])
