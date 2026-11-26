import unittest
import torch
from vlanix_tiny.safe_step import project_core_step

class SafeStepTest(unittest.TestCase):
    def test_hidden_conflict_global(self):
        p=torch.nn.Parameter(torch.tensor([1.])); q=torch.nn.Parameter(torch.tensor([-2.]))
        info=project_core_step([p,q],[torch.zeros(1),torch.zeros(1)],[torch.ones(1),torch.zeros(1)],[torch.zeros(1),torch.ones(1)])
        self.assertEqual(p.item(),0.); self.assertEqual(q.item(),-2.)
        self.assertEqual(info['raw_local'],1.); self.assertEqual(info['raw_readout'],-2.)
        self.assertTrue(info['moved'])
    def test_compatible_unchanged(self):
        p=torch.nn.Parameter(torch.tensor([-1.,-2.]))
        info=project_core_step([p],[torch.zeros(2)],[torch.tensor([1.,0.])],[torch.tensor([0.,1.])])
        self.assertTrue(torch.equal(p,torch.tensor([-1.,-2.])))
        self.assertTrue(info['moved'])
    def test_zero_step(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        info=project_core_step([p],[torch.zeros(2)],[torch.tensor([1.,0.])],[torch.tensor([0.,1.])])
        self.assertFalse(info['moved']); self.assertTrue(torch.equal(p,torch.zeros(2)))
    def test_nonfinite_no_write(self):
        p=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError): project_core_step([p],[torch.zeros(1)],[torch.ones(1)],[torch.tensor([float('nan')])])
        self.assertEqual(p.item(),1.)
    def test_float32_actual_displacement(self):
        gen=torch.Generator().manual_seed(42)
        for _ in range(100):
            before=torch.randn(16,generator=gen)
            p=torch.nn.Parameter(before+0.01*torch.randn(16,generator=gen))
            a,b=torch.randn(2,16,generator=gen)
            info=project_core_step([p],[before],[a],[b])
            d=p.detach().double()-before.double()
            for v,k in ((a,'local'),(b,'readout')):
                dot=torch.dot(v.double(),d).item()
                self.assertAlmostEqual(dot,info['after_'+k],places=12)
                self.assertLessEqual(dot,1e-8)
    def test_layout(self):
        p=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError): project_core_step([p],[torch.zeros(2)],[torch.ones(1)],[torch.ones(1)])
