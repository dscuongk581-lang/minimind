import unittest
from vlanix_tiny.wide_dataset import compose, reference


def local(q):
    op,a,b,c=q
    if op==1: return a+b+c
    if op==10:
        d=a-b-c
        return (d&1)+2*int(d<0)
    return {2:a^b,7:a&b,8:a|b}[op]


def readout(t):
    return 9+sum(((v-9)&1)<<i for i,v in enumerate(t))


class WideDatasetTests(unittest.TestCase):
    def test_boundary(self):
        self.assertEqual(compose(1,255,1,16,local,readout),256)
        self.assertEqual(compose(10,256,1,16,local,readout),255)
        self.assertEqual(compose(10,0,1,32,local,readout),2**32-1)
    def test_all_ops(self):
        for op in (1,2,7,8,10):
            for a,b in ((0,65535),(65535,1),(0xaaaa,0x5555),(256,255)):
                self.assertEqual(compose(op,a,b,16,local,readout),reference(op,a,b,16))
    def test_carry_is_predicted(self):
        seen=[]
        def predict(q):
            seen.append(q)
            return local(q)
        compose(1,255,1,16,predict,readout)
        self.assertEqual(seen[8],(1,0,0,1))
    def test_no_oracle_repair(self):
        self.assertEqual(compose(1,255,1,16,local,lambda t:9),0)
        self.assertIsNone(compose(1,255,1,16,lambda q:4,readout))
        self.assertIsNone(compose(1,255,1,16,local,lambda t:265))
    def test_contract(self):
        for args in ((1,65536,0,16),(1,0,0,12),(99,0,0,16),(1,-1,0,16)):
            with self.assertRaises(ValueError): compose(*args,local,readout)
