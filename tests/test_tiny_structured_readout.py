import unittest
import torch
import torch.nn.functional as F
from vlanix_tiny.structured_readout import objective, marginal_bytes

class StructuredReadoutTests(unittest.TestCase):
    def fixture(self):
        z=torch.randn(32,10,265,requires_grad=True)
        y=torch.full((32,10),-100,dtype=torch.long)
        y[:24,9]=torch.arange(24)+9; y[24:,4]=11
        return z,y
    def test_baseline(self):
        z,y=self.fixture()
        torch.testing.assert_close(objective(z,y,0),F.cross_entropy(z.reshape(-1,265),y.reshape(-1)))
    def test_auxiliary(self):
        z,y=self.fixture()
        a=objective(z,y,0); b=objective(z,y,8)
        self.assertGreater(b.item(),a.item()); b.backward()
        self.assertTrue(torch.isfinite(z.grad).all()); self.assertEqual(z.grad[0,0].abs().sum().item(),0.)
        ga=torch.autograd.grad(a,z)[0]
        torch.testing.assert_close(z.grad[24:],ga[24:])
    def test_marginals(self):
        z=torch.full((256,265),-20.)
        z[torch.arange(256),torch.arange(256)+9]=20.
        self.assertEqual(marginal_bytes(z).tolist(),list(range(256)))
        z[:,0]=100.
        self.assertEqual(marginal_bytes(z).tolist(),list(range(256)))
    def test_invalid(self):
        z,y=self.fixture()
        with self.assertRaises(ValueError): objective(z,y,-1)
        y[0,0]=9
        with self.assertRaises(ValueError): objective(z,y,8)
        with self.assertRaises(ValueError): marginal_bytes(torch.zeros(1,256))
