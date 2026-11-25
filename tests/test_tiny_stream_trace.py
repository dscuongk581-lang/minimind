import unittest
from vlanix_tiny.stream_trace import encode, generate, batches

class StreamTraceTests(unittest.TestCase):
    def test_alignment(self):
        q=bytes([1,255,1]); t=[11]*8+[9]
        for mode in ('front','stream'):
            x,y=encode(q,t,mode)
            self.assertEqual(len(x),26)
            self.assertEqual([v for v in y if v!=-100],t)
            self.assertEqual(y[-1],9)
        x,y=encode(q,t,'stream')
        self.assertEqual(x[:5],[1,10,10,10,11])
        self.assertEqual([i for i,v in enumerate(y) if v!=-100],[3,6,9,12,15,18,21,24,25])
    def test_feedback(self):
        q=bytes([1,0,0]); seen=[]
        def predict(p):
            seen.append(p[0][:]); return [11 if len(seen)==1 else 9]
        out=generate([q],predict,'stream')
        self.assertEqual(out,[[11]+[9]*8])
        self.assertEqual(seen[1],[1,10,9,9,11,9,9])
        self.assertEqual(len(seen[-1]),26)
    def test_front(self):
        q=bytes([2,13,6]); seen=[]
        def predict(p):
            seen.append(p[0][:]); return [0]
        self.assertEqual(generate([q],predict,'front'),[[0]*9])
        self.assertEqual(len(seen[0]),18)
        self.assertEqual(seen[-1][-8:],[0]*8)
    def test_rehearsal(self):
        rows=list(range(48)); local=list(range(32))
        b=batches(rows,local,42)
        self.assertEqual([v for a,_ in b for v in a],rows)
        self.assertEqual([len(c) for _,c in b],[8,8])
        self.assertEqual(b,batches(rows,local,42))
        with self.assertRaises(ValueError): batches(rows[:25],local,42)
