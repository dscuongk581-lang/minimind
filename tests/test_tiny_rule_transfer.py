import unittest
import hashlib
from vlanix_tiny.rule_transfer_data import or_rows, adaptation_orders


def row(op,a,b,split='train'):
    q=bytes((op,a,b))
    return dict(id=hashlib.sha256(q).hexdigest(),query=q.hex(),split=split,
        tier='easy' if max(a,b)<16 else 'hard',family='and',answer=a&b)

class TransferTests(unittest.TestCase):
    def test_or(self):
        old=[row(7,1,2),row(7,4,8,'validation')]
        new=or_rows(old)
        self.assertEqual([r['answer'] for r in new],[3,12])
        self.assertEqual([r['split'] for r in new],['train','validation'])
        self.assertEqual([r['query'] for r in new],['080102','080408'])
        self.assertNotEqual(new[0]['id'],old[0]['id'])
    def test_orders(self):
        old=[row(7,a,19) for a in range(24)]
        old.append(row(7,33,19,'validation'))
        new=or_rows(old)
        a,b=adaptation_orders(old,new,42)
        self.assertEqual((len(a),len(b)),(96,96))
        for i in range(0,96,32):
            self.assertEqual(a[i:i+24],b[i:i+24])
            self.assertTrue(all(r['family']=='or' for r in a[i:i+32]))
            self.assertTrue(all(r['family']=='and' for r in b[i+24:i+32]))
        self.assertTrue(all(r['split']=='train' for r in a+b))
        self.assertEqual((a,b),adaptation_orders(old,new,42))
    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError): or_rows([row(7,1,2),row(7,2,1,'validation')])
    def test_invalid_batch_rejected(self):
        old=[row(7,1,2)]
        with self.assertRaises(ValueError): adaptation_orders(old,or_rows(old),42)
