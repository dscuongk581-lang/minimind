import unittest
from vlanix_tiny.residual import decompose, decide, toy_probe

class ResidualTests(unittest.TestCase):
    def test_weighted_cumulant(self):
        r=decompose({'a':[-1,1,-1,1], 'b':[-2,2,-2,2,-2,2,-2,2]})
        self.assertAlmostEqual(r['variance'],3)
        self.assertAlmostEqual(r['within_k4'],-22)
        self.assertAlmostEqual(r['between_k4'],6)
        self.assertAlmostEqual(r['k4'],-16)
        self.assertAlmostEqual(r['closure_error'],0)
    def test_group_mean_signal(self):
        r=decompose({'a':[2,4,2,4],'b':[-4,-2,-4,-2]})
        self.assertAlmostEqual(r['mean_variance'],9)
        self.assertAlmostEqual(r['variance'],1)
        self.assertAlmostEqual(r['raw_variance'],10)
    def test_invalid(self):
        for groups in ({}, {'a':[1,2]}, {'a':[1,2,3,float('nan')]}):
            with self.assertRaises(ValueError): decompose(groups)
    def test_gate(self):
        self.assertEqual(decide(6_000_000,24576,[.2,.2,.2],.1,.09,.001),'PROBE_ACCEPT')
        self.assertEqual(decide(29_990_000,24576,[.2]*3,.1,.09,0),'BUDGET')
        self.assertEqual(decide(6_000_000,24576,[.2]*3,.1,.09,.1),'RETENTION')
        self.assertEqual(decide(6_000_000,24576,[.2]*3,.1,.11,0),'NO_GAIN')
        self.assertEqual(decide(6_000_000,24576,[.2,.0,.2],.1,.09,0),'NOT_PERSISTENT')
    def test_toy_structured_vs_noise(self):
        a,b=toy_probe('structured'),toy_probe('noise')
        self.assertEqual(a['decision'],'PROBE_ACCEPT')
        self.assertEqual(b['decision'],'NOT_PERSISTENT')
        self.assertEqual(a['retention_delta'],0)
        self.assertGreater(a['validation_gain'],0)
        self.assertEqual(b['validation_gain'],0)
