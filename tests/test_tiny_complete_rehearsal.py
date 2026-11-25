import unittest
import torch
import torch.nn.functional as F
from vlanix_tiny.complete_rehearsal import rehearsal, objective, margins

class RehearsalTests(unittest.TestCase):
    def test_coverage(self):
        qs=list(range(32)); sample=qs[:8]
        self.assertEqual(rehearsal(qs,sample,False),sample*4)
        self.assertEqual(rehearsal(qs,sample,True),qs)
        with self.assertRaises(ValueError): rehearsal(qs,[0]*8,True)
    def test_weight(self):
        z=torch.zeros(56,10,265,requires_grad=True)
        y=torch.full((56,10),-100,dtype=torch.long); y[:24,9]=10; y[24:,4]=11
        loss=objective(z,y); loss.backward()
        self.assertAlmostEqual(loss.item(),F.cross_entropy(z[:24,9],y[:24,9]).item(),places=5)
        g=z.grad.abs().sum(-1)
        self.assertAlmostEqual((g[0,9]/g[24,4]).item(),4.,places=5)
        self.assertEqual(g[0,0].item(),0.)
        y[0,0]=9
        with self.assertRaises(ValueError): objective(z,y)
    def test_duplicate_equivalence(self):
        a=torch.randn(24,265); b=torch.randn(8,265); t=torch.arange(24)+9; u=torch.arange(8)%4+9
        z=torch.zeros(56,10,265); y=torch.full((56,10),-100,dtype=torch.long)
        z[:24,9]=a; z[24:,4]=b.repeat(4,1); y[:24,9]=t; y[24:,4]=u.repeat(4)
        torch.testing.assert_close(objective(z,y),.75*F.cross_entropy(a,t)+.25*F.cross_entropy(b,u))
    def test_margin(self):
        z=torch.tensor([[2.,1.,0.],[0.,3.,5.],[1.,1.,0.]])
        torch.testing.assert_close(margins(z,torch.tensor([0,1,0])),torch.tensor([1.,-2.,0.]))
