import unittest
from vlanix_tiny.state_invariance import bounded_rollout, reset_rollout

class StateInvarianceTests(unittest.TestCase):
    def test_window_bound(self):
        seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [9]
        out=bounded_rollout([(1,255,1,16)],predict,8)
        self.assertEqual(out,[[9]*16])
        self.assertEqual([len(x) for x in seen[:8]],list(range(4,26,3)))
        self.assertEqual(max(map(len,seen)),25)
        self.assertTrue(all(x[:2]==[1,10] for x in seen))
    def test_window_feedback(self):
        seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [13 if len(seen)==1 else 9]
        bounded_rollout([(1,0,0,4)],predict,2)
        self.assertEqual(seen[1],[1,10,9,9,13,9,9])
        self.assertEqual(seen[2],[1,10,9,9,9,9,9])
    def test_reset_feedback(self):
        seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [11 if len(seen)==1 else 9]
        self.assertEqual(reset_rollout([(1,0,0,4)],predict),[[11,9,9,9]])
        self.assertEqual(seen[1],[1,10,9,9,10])
        self.assertEqual(seen[2],[1,10,9,9,9])
    def test_invalid_reset(self):
        self.assertEqual(reset_rollout([(1,0,0,4)],lambda b:[13]*len(b)),[[13,None,None,None]])
        with self.assertRaises(ValueError): bounded_rollout([(1,0,0,4)],lambda b:[265],8)
        with self.assertRaises(ValueError): bounded_rollout([(1,0,0,4)],lambda b:[9],0)
