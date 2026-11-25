import unittest
from vlanix_tiny.width_probe import gold, decode, rollout, summarize, extend

class WidthProbeTests(unittest.TestCase):
    def test_carry_and_width(self):
        q=(1,65535,1,16)
        self.assertEqual(gold(q),[11]*16)
        self.assertEqual(decode(gold(q)),0)
        self.assertEqual(decode(gold((8,65535,1,16))),65535)
    def test_feedback(self):
        q=(1,0,0,4); seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [13 if len(seen)==1 else 9]
        out=rollout([q],predict)
        self.assertEqual(out,[[13,9,9,9]])
        self.assertEqual(seen[1],[1,10,9,9,13,9,9])
        self.assertEqual(summarize([q],out)['first_error'][0],1)
        self.assertEqual(summarize([q],out)['valid'],0)
        seen.clear(); rollout([q],predict,teacher=True)
        self.assertEqual(seen[1],[1,10,9,9,9,9,9])
    def test_extension(self):
        q=(2,137,254,8)
        p=extend(q,32,43)
        self.assertEqual((p[0],p[1]&255,p[2]&255,p[3]),(2,137,254,32))
        self.assertEqual(p,extend(q,32,43))
        self.assertEqual(gold(p)[:8],gold(q))
    def test_bad(self):
        with self.assertRaises(ValueError): gold((1,256,0,8))
        with self.assertRaises(ValueError): rollout([(1,0,0,4)],lambda p:[265])
        with self.assertRaises(ValueError): rollout([(1,0,0,4),(1,0,0,8)],lambda p:[9,9])
        self.assertIsNone(decode([9,13]))
