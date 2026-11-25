import unittest
from vlanix_tiny.binary_logic import generate, solve, verify, encode, FAMILIES, report

class BinaryLogicTests(unittest.TestCase):
    def test_known_vectors(self):
        cases=[('0107ff',6),('02a50f',165^15),('03020304',20),('04fafd00',3),('05010204081020408000',255),('06010205',13)]
        for h,y in cases: self.assertEqual(solve(bytes.fromhex(h)),y,h)
    def test_invalid(self):
        for q in (b'',b'\x00',b'\x01\x01',b'\x06\x01\x02\x00',b'\x04\x01\x02\x03\x04'):
            with self.assertRaises(ValueError): solve(q)
    def test_corpus(self):
        rows=generate(128,42)
        self.assertEqual(len(rows),2304)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(rows,generate(128,42))
        for r in rows:
            self.assertTrue(verify(r))
            a=bytes.fromhex(r['query'])[1]
            self.assertEqual(r['split'],'train' if a<128 else 'validation' if a<192 else 'test')
        self.assertEqual(set(r['family'] for r in rows),set(FAMILIES))
        bad=dict(rows[0],answer=rows[0]['answer']^1)
        self.assertFalse(verify(bad))
    def test_labels(self):
        r=generate(1,42)[0]; ids,labels=encode(r)
        self.assertEqual(ids[0],1); self.assertEqual(ids[-1],2)
        self.assertEqual(labels[:-2],[-100]*(len(ids)-2))
        self.assertEqual(labels[-2:], [r['answer']+9,2])
    def test_report(self):
        r=report(generate(128,42))
        self.assertEqual(r['rows'],2304)
        self.assertEqual(r['max_tokens'],13)
        self.assertEqual(r['target_tokens'],4608)
        self.assertEqual(r['splits'],{'train':768,'validation':768,'test':768})
