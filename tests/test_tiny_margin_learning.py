import unittest
import torch
from vlanix_tiny.margin_learning import penalty

class MarginLearningTests(unittest.TestCase):
    def test_hinge(self):
        z=torch.tensor([[3.,0.,1.],[0.,2.,1.],[0.,1.,2.]])
        y=torch.tensor([0,1,1])
        self.assertAlmostEqual(penalty(z,y,1.).item(),2/3,places=6)
    def test_gradient(self):
        z=torch.tensor([[.2,0.,-1.],[3.,0.,1.]],requires_grad=True)
        penalty(z,torch.tensor([0,0]),1.).backward()
        torch.testing.assert_close(z.grad,torch.tensor([[-.5,.5,0.],[0.,0.,0.]]))
    def test_shift(self):
        z=torch.tensor([[.1,0.,-1.],[3.,0.,1.]])
        y=torch.tensor([0,0])
        torch.testing.assert_close(penalty(z,y,1.),penalty(z+10,y,1.))
    def test_invalid(self):
        z=torch.zeros(2,3); y=torch.tensor([0,0])
        for m in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): penalty(z,y,m)
        with self.assertRaises(ValueError): penalty(z,torch.tensor([0,3]),1.)
