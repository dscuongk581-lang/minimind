import unittest
import torch
import torch.nn.functional as F
from vlanix_tiny.bit_objective import bit_loss, objective

class BitObjectiveTests(unittest.TestCase):
    def test_uniform(self):
        z=torch.zeros(2,265,dtype=torch.float64)
        y=torch.tensor([9,264])
        torch.testing.assert_close(bit_loss(z,y),torch.tensor(0.6931471805599453,dtype=z.dtype))
    def test_distance(self):
        y=torch.tensor([9]); near=torch.full((1,265),-10.); far=near.clone()
        near[0,10]=10.; far[0,264]=10.
        self.assertLess(bit_loss(near,y).item(),bit_loss(far,y).item())
        torch.testing.assert_close(F.cross_entropy(near,y),F.cross_entropy(far,y))
    def test_gradient(self):
        z=torch.randn(4,265,dtype=torch.float64,requires_grad=True); y=torch.tensor([9,10,137,264])
        self.assertTrue(torch.autograd.gradcheck(lambda x: bit_loss(x,y),(z,),fast_mode=True))
        loss=objective(z,y,1.); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad[:,:9].abs().sum().item(),0.)
    def test_zero_matches(self):
        z=torch.randn(3,265); y=torch.tensor([9,44,264])
        self.assertTrue(torch.equal(objective(z,y,0.),F.cross_entropy(z,y)))
    def test_invalid(self):
        with self.assertRaises(ValueError): bit_loss(torch.zeros(1,265),torch.tensor([2]))
        with self.assertRaises(ValueError): objective(torch.zeros(1,265),torch.tensor([9]),-1.)
