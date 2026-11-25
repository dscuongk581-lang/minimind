import unittest
from vlanix_tiny.homeostasis import PI

class PITests(unittest.TestCase):
    def test_nominal(self):
        p=PI()
        self.assertEqual(p.update(0),1.0)
        self.assertEqual(p.integral,0.0)
    def test_accumulation_and_recovery(self):
        p=PI(smoothing=1.0)
        a=p.update(.06); b=p.update(.06)
        self.assertGreater(a,b)
        for _ in range(20): c=p.update(0)
        self.assertEqual(c,1.0)
    def test_saturation_antiwindup(self):
        p=PI(smoothing=1.0)
        for _ in range(100): self.assertAlmostEqual(p.update(10),.05)
        self.assertEqual(p.integral,0.0)
        self.assertEqual(p.update(0),1.0)
    def test_invalid(self):
        p=PI()
        for x in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): p.update(x)
        self.assertEqual(p.integral,0.)
        self.assertEqual(p.filtered,0.)
        with self.assertRaises(ValueError): PI(smoothing=0)
    def test_filter(self):
        p=PI(smoothing=.5)
        p.update(.1)
        self.assertAlmostEqual(p.filtered,.05)
        p.update(0)
        self.assertAlmostEqual(p.filtered,.025)
