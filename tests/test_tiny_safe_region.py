import unittest
import torch
from vlanix_tiny.safe_region import project_two

class SafeRegionTest(unittest.TestCase):
    def t(self, x): return torch.tensor(x, dtype=torch.float64)
    def test_hidden_conflict(self):
        d,a,b=map(self.t,([1,-2],[1,0],[0,1]))
        self.assertLess(torch.dot(d,a+b),0)
        self.assertTrue(torch.equal(project_two(d,a,b),self.t([0,-2])))
    def test_oblique_single_active(self):
        d,a,b=map(self.t,([1,2,3],[1,0,0],[1,1,0]))
        self.assertTrue(torch.allclose(project_two(d,a,b),self.t([-0.5,0.5,3])))
    def test_corner(self):
        d,a,b=map(self.t,([2,2,3],[1,0,0],[0,1,0]))
        self.assertTrue(torch.equal(project_two(d,a,b),self.t([0,0,3])))
    def test_opposite(self):
        d,a,b=map(self.t,([1,2],[1,0],[-1,0]))
        self.assertTrue(torch.equal(project_two(d,a,b),self.t([0,2])))
    def test_zero_and_compatible(self):
        d,a,b=map(self.t,([-1,-2],[0,0],[0,1]))
        self.assertTrue(torch.equal(project_two(d,a,b),d))
    def test_nonfinite(self):
        with self.assertRaises(ValueError): project_two(self.t([1,2]),self.t([float('nan'),0]),self.t([0,1]))
    def test_random_feasibility(self):
        gen=torch.Generator().manual_seed(42)
        for _ in range(200):
            d,a,b=torch.randn(3,8,generator=gen,dtype=torch.float64)
            x=project_two(d,a,b)
            self.assertLessEqual(max(torch.dot(a,x),torch.dot(b,x)),1e-9)
            self.assertLessEqual((x-d).norm(),d.norm()+1e-9)
