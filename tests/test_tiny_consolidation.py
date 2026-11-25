import unittest
from collections import Counter
from vlanix_tiny.consolidation import fresh_frames

class ConsolidationTests(unittest.TestCase):
    def test_order_and_balance(self):
        train=[(b,c) for b in range(256) for c in range(15)]
        held=[(b,c) for b in range(256) for c in range(15,31)]
        fresh=fresh_frames(train,held,42)
        self.assertEqual([b for b,c in fresh],[b for b,c in train])
        self.assertEqual(Counter(b for b,c in fresh),{b:15 for b in range(256)})
        self.assertEqual(len(set(fresh)),3840)
        self.assertFalse(set(fresh)&(set(train)|set(held)))
    def test_determinism(self):
        train=[(0,0),(1,1),(0,1)]; held=[(0,2),(1,2)]
        self.assertEqual(fresh_frames(train,held,42),fresh_frames(train,held,42))
    def test_invalid(self):
        for train,held in [([],[]),([(0,0)],[(0,0)]),([(0,0),(0,0)],[]),([(256,0)],[]),([(0,True)],[])]:
            with self.assertRaises(ValueError): fresh_frames(train,held,42)
    def test_capacity(self):
        with self.assertRaises(ValueError): fresh_frames([(0,c) for c in range(129)],[],42)
