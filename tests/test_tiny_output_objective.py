import unittest
import torch
import torch.nn.functional as F
from vlanix_tiny.output_objective import weighted_loss, decode_states

class OutputObjectiveTests(unittest.TestCase):
    def fixture(self):
        y=torch.full((32,26),-100,dtype=torch.long)
        y[:24,[3,6,9,12,15,18,21,24]]=9
        y[:24,25]=20; y[24:,4]=11
        z=torch.randn(32,26,265,dtype=torch.float64,requires_grad=True)
        return z,y
    def test_baseline(self):
        z,y=self.fixture()
        torch.testing.assert_close(weighted_loss(z,y,1,1),F.cross_entropy(z.reshape(-1,265),y.reshape(-1)))
    def test_gradient_ratios(self):
        z,y=self.fixture(); z.data.zero_()
        weighted_loss(z,y,8,8).backward()
        g=z.grad.abs().sum(-1)
        self.assertAlmostEqual((g[0,25]/g[0,3]).item(),8.)
        self.assertAlmostEqual((g[24,4]/g[0,3]).item(),8.)
        self.assertEqual(g[0,0].item(),0.)
        self.assertTrue(torch.isfinite(z.grad).all())
    def test_invalid(self):
        z,y=self.fixture()
        with self.assertRaises(ValueError): weighted_loss(z,y,0,1)
        y[0,0]=9
        with self.assertRaises(ValueError): weighted_loss(z,y,1,1)
    def test_decode(self):
        self.assertEqual(decode_states([10,9,12,11,9,9,9,10]),133)
        self.assertIsNone(decode_states([9]*7+[13]))
        self.assertIsNone(decode_states([9]*7))
        self.assertIsNone(decode_states([9.0]*8))
