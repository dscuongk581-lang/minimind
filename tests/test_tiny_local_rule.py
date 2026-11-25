import unittest
from vlanix_tiny.local_rule import frames,target,compose,exhaustive

class LocalRuleTests(unittest.TestCase):
    def test_frames(self):
        self.assertEqual(len(set(frames())),32)
        self.assertEqual(target((1,1,1,1)),3)
        self.assertEqual(target((2,1,1,1)),0)
    def test_carry_feedback(self):
        seen=[]
        def predict(q):
            seen.append(q)
            return 2 if len(seen)==1 else 1 if q[-1] else 0
        self.assertEqual(compose(1,0,0,predict),2)
        self.assertEqual([q[-1] for q in seen],[0,1,0,0,0,0,0,0])
    def test_invalid(self):
        self.assertIsNone(compose(1,0,0,lambda q:4))
        with self.assertRaises(ValueError): compose(1,256,0,target)
    def test_exhaustive(self):
        report=exhaustive(target)
        self.assertEqual(sum(x['correct'] for x in report.values()),262144)
        self.assertTrue(all(x['invalid']==0 for x in report.values()))
