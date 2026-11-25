import unittest
import torch
from vlanix_tiny.acquisition_direction import project_acquisition

class AcquisitionDirectionTests(unittest.TestCase):
    def test_conflict(self):
        g=[torch.tensor([-2.,3.])]; h=[torch.tensor([1.,0.])]
        d,info=project_acquisition(g,h)
        self.assertTrue(info['conflict'])
        torch.testing.assert_close(d[0],torch.tensor([1.,3.]))
        self.assertGreaterEqual(info['projected_dot'],-1e-6)
        torch.testing.assert_close(g[0],torch.tensor([-2.,3.]))
    def test_agreement(self):
        d,info=project_acquisition([torch.tensor([2.,3.])],[torch.tensor([1.,0.])])
        self.assertFalse(info['conflict'])
        torch.testing.assert_close(d[0],torch.tensor([3.,3.]))
    def test_global_and_zero(self):
        d,info=project_acquisition([torch.tensor([-2.]),torch.tensor([1.])],[torch.tensor([1.]),torch.tensor([1.])])
        torch.testing.assert_close(torch.cat(d),torch.tensor([-.5,2.5]))
        self.assertAlmostEqual(info['projected_dot'],0.)
        d,info=project_acquisition([torch.tensor([2.])],[torch.zeros(1)])
        torch.testing.assert_close(d[0],torch.tensor([2.]))
    def test_invalid(self):
        for g,h in [([],[]),([torch.ones(2)],[torch.ones(1)]),([torch.tensor([float('nan')])],[torch.ones(1)])]:
            with self.assertRaises(ValueError): project_acquisition(g,h)
