import unittest
from collections import Counter
from vlanix_tiny.curriculum import schedules

class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(chunk_id=str(i),split="train",token_count=n) for i,n in enumerate([64,12,256,128,32,512,50,100])]
    def test_same_examples_and_targets(self):
        a,b,m=schedules(self.rows,seed=42,epochs=3,batch_size=2)
        expected=Counter({r["chunk_id"]:3 for r in self.rows})
        self.assertEqual(Counter(x for batch in a for x in batch),expected)
        self.assertEqual(Counter(x for batch in b for x in batch),expected)
        self.assertEqual(m["target_tokens"],3*sum(r["token_count"]-1 for r in self.rows))
        self.assertEqual(len(a),len(b))
    def test_length_order_and_reproducibility(self):
        a,b,m=schedules(self.rows,42,1,2)
        sizes={r["chunk_id"]:r["token_count"] for r in self.rows}
        lengths=[sizes[x] for batch in b for x in batch]
        self.assertEqual(lengths,sorted(lengths))
        self.assertEqual((a,b,m),schedules(list(reversed(self.rows)),42,1,2))
        self.assertNotEqual(a,b)
    def test_reject_validation_duplicates_and_bad_counts(self):
        for rows in ([dict(self.rows[0],split="validation")],self.rows+[self.rows[0]],[dict(self.rows[0],token_count=513)]):
            with self.assertRaises(ValueError):
                schedules(rows,42,1,2)
    def test_partial_batches_preserved(self):
        a,b,m=schedules(self.rows[:5],42,2,3)
        self.assertEqual([len(x) for x in a],[3,2,3,2])
        self.assertEqual([len(x) for x in b],[3,2,3,2])
