import unittest
import torch
from vlanix_tiny.actual_update import project_step

class ActualUpdateTest(unittest.TestCase):
    def test_adamw_conflict(self):
        p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=.01)
        h=[torch.tensor([2.,1.],dtype=p.dtype)]
        before=[p.detach().clone()]
        p.grad=torch.tensor([-1.,2.],dtype=p.dtype)
        opt.step()
        self.assertGreater(torch.dot(p.detach()-before[0],h[0]).item(),0)
        info=project_step([p],before,h)
        self.assertTrue(info['conflict'])
        self.assertLessEqual(torch.dot(p.detach()-before[0],h[0]).item(),1e-12)
        self.assertEqual(opt.state[p]['step'].item(),1)
    def test_compatible_and_zero(self):
        p=torch.nn.Parameter(torch.tensor([-1.,2.]))
        for h in (torch.tensor([1.,0.]),torch.zeros(2)):
            self.assertFalse(project_step([p],[torch.zeros(2)],[h])['conflict'])
            self.assertTrue(torch.equal(p,torch.tensor([-1.,2.])))
    def test_global(self):
        p=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
        q=torch.nn.Parameter(torch.tensor([-1.],dtype=torch.float64))
        info=project_step([p,q],[torch.zeros_like(p),torch.zeros_like(q)],[torch.ones_like(p),torch.ones_like(q)])
        self.assertTrue(info['conflict'])
        self.assertAlmostEqual(p.item(),1.5)
        self.assertAlmostEqual(q.item(),-1.5)
    def test_nonfinite_no_partial_write(self):
        p=torch.nn.Parameter(torch.ones(1)); q=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            project_step([p,q],[torch.zeros(1),torch.zeros(1)],[torch.ones(1),torch.tensor([float('nan')])])
        self.assertEqual(p.item(),1.)
    def test_layout(self):
        p=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError): project_step([p],[torch.zeros(2)],[torch.ones(1)])
