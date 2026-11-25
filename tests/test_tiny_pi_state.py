import unittest
from vlanix_tiny.homeostasis import PI
class PIStateTests(unittest.TestCase):
    def test_continuation(self):
        a=PI(smoothing=.7)
        for d in (.02,.07,.1): a.update(d)
        b=PI.restore(a.snapshot())
        for d in (.3,.04,0.,.02):
            self.assertEqual(a.update(d),b.update(d))
            self.assertEqual(a.snapshot(),b.snapshot())
    def test_invalid(self):
        for field,value in [('integral',-1.),('integral',99.),('filtered',float('nan')),('target',0.),('kp',True),('schema','bad')]:
            d=PI().snapshot(); d[field]=value
            with self.assertRaises(ValueError): PI.restore(d)
        with self.assertRaises(ValueError): PI.restore({})
