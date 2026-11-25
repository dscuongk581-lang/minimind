import unittest
from itertools import product
from vlanix_tiny.dual_guard import sentinel_schedule, full_readout_ok

class DualGuardTests(unittest.TestCase):
    def test_schedule(self):
        train=[(b,c) for b in range(256) for c in range(15)]
        a=sentinel_schedule(train,42)
        self.assertEqual(a,sentinel_schedule(train,42))
        self.assertEqual(len(a),120)
        fixed={(b,0) for b in range(0,256,4)}
        self.assertTrue(all(len(set(row))==128 and fixed<=set(row)<=set(train) for row in a))
        self.assertGreater(len(set().union(*map(set,a))),3000)
    def test_invalid_schedule(self):
        with self.assertRaises(ValueError): sentinel_schedule([(0,0)],42)
    def test_complete_readout(self):
        table={t:9+sum(((v-9)&1)<<k for k,v in enumerate(t)) for t in product(range(9,13),repeat=8)}
        self.assertTrue(full_readout_ok(table))
        table[(9,)*8]=10
        self.assertFalse(full_readout_ok(table))
        del table[(9,)*8]
        self.assertFalse(full_readout_ok(table))
    def test_invalid_readout(self):
        self.assertFalse(full_readout_ok({(9,)*8:9}))
