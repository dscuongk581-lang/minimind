import unittest
from vlanix_tiny.logic import generate, solve, verify

class LogicTests(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(solve('arithmetic',{'a':9,'b':4,'op':'-'}),'5')
        self.assertEqual(solve('boolean',{'bits':[1,0,0,0,0,0,0,0],'mode':'pair_or'}),'false')
        self.assertEqual(solve('state',{'start':2,'ops':[['+',3],['*',4]]}),'20')
        self.assertEqual(solve('rules',{'facts':['P0'],'rules':[['P0','P1'],['P1','P2']],'query':'P2'}),'yes')
        self.assertEqual(solve('rules',{'facts':[],'rules':[['P0','P1']],'query':'P1'}),'unknown')
    def test_split_and_reproducibility(self):
        a=generate(256,42)
        self.assertEqual(a,generate(256,42))
        self.assertEqual(len(a),3072)
        self.assertEqual(len({r['instance_id'] for r in a}),len(a))
        self.assertTrue(all(verify(r) for r in a))
        self.assertTrue(all(r['text'].isascii() for r in a))
        self.assertEqual({r['split'] for r in a},{'train','validation','test'})
    def test_tamper_and_limits(self):
        row=generate(1,42)[0]
        self.assertFalse(verify(dict(row,answer='wrong')))
        self.assertFalse(verify(dict(row,text='changed')))
        self.assertFalse(verify(dict(row,split='test')))
        with self.assertRaises(ValueError): generate(257,42)
