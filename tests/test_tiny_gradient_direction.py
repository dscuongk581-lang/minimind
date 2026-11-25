import unittest
import torch
from vlanix_tiny.gradient_direction import combine

class DirectionTests(unittest.TestCase):
    def test_conflict(self):
        g=[torch.tensor([-2.,3.])]; h=[torch.tensor([1.,0.])]
        out,stats=combine(g,h,True)
        torch.testing.assert_close(out[0],torch.tensor([.25,2.25]))
        self.assertTrue(stats['conflict']); self.assertAlmostEqual(stats['projected_dot'],0.)
    def test_control_and_alignment(self):
        g=[torch.tensor([-2.,3.])]; h=[torch.tensor([1.,0.])]
        out,_=combine(g,h,False)
        torch.testing.assert_close(out[0],torch.tensor([-1.25,2.25]))
        g=[torch.tensor([2.,3.])]
        a,_=combine(g,h,False); b,s=combine(g,h,True)
        torch.testing.assert_close(a[0],b[0]); self.assertFalse(s['conflict'])
    def test_global_projection(self):
        g=[torch.tensor([-2.]),torch.tensor([1.])]; h=[torch.tensor([1.]),torch.tensor([1.])]
        out,s=combine(g,h,True)
        self.assertAlmostEqual(s['projected_dot'],0.)
        torch.testing.assert_close(torch.cat(out),torch.tensor([-.875,1.375]))
    def test_zero_and_invalid(self):
        g=[torch.tensor([2.])]; h=[torch.tensor([0.])]
        out,s=combine(g,h,True)
        torch.testing.assert_close(out[0],torch.tensor([1.5])); self.assertFalse(s['conflict'])
        with self.assertRaises(ValueError): combine(g,[],True)
        with self.assertRaises(ValueError): combine(g,[torch.tensor([float('nan')])],True)
