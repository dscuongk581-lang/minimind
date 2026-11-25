import unittest
from collections import Counter
from vlanix_tiny.readout_balance import states, plan, probes

class ReadoutBalanceTests(unittest.TestCase):
    def test_states(self):
        for b in range(256):
            for c in (0,1,85,170,255):
                t=states(b,c)
                self.assertEqual(sum(((v-9)&1)<<k for k,v in enumerate(t)),b)
                self.assertEqual(sum(((v-9)>>1)<<k for k,v in enumerate(t)),c)
    def test_plan(self):
        rows=plan(42)
        self.assertEqual(len(rows),3840)
        self.assertEqual(Counter(b for b,c in rows),{b:15 for b in range(256)})
        self.assertEqual(len(set(rows)),3840)
        self.assertEqual(rows,plan(42))
    def test_disjoint(self):
        train=plan(42); held=probes(42)
        self.assertEqual(len(held),4096)
        self.assertEqual(len(set(held)),4096)
        self.assertFalse(set(train)&set(held))
        self.assertTrue(all(c!=0 for b,c in held))
        self.assertEqual(Counter(b for b,c in held),{b:16 for b in range(256)})
        self.assertTrue(all((b,0) in set(train) for b in range(256)))
    def test_invalid(self):
        for a,b in ((-1,0),(256,0),(0,256),(True,0),(0,0.5)):
            with self.assertRaises(ValueError): states(a,b)
