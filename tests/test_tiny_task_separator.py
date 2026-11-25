import unittest
from vlanix_tiny.task_separator import encode, generate, READOUT

class TaskSeparatorTests(unittest.TestCase):
    def test_alignment(self):
        q=bytes([1,255,1]); t=[11]*8+[9]
        a,y=encode(q,t,False); b,z=encode(q,t,True)
        self.assertEqual(len(a),27); self.assertEqual(len(b),27)
        self.assertEqual(a[:26],b[:26]); self.assertEqual((a[-1],b[-1]),(0,READOUT))
        self.assertEqual(y[:25],z[:25]); self.assertEqual(y[25:],[9,-100]); self.assertEqual(z[25:],[-100,9])
        self.assertEqual([i for i,v in enumerate(z) if v!=-100],[3,6,9,12,15,18,21,24,26])
    def test_feedback(self):
        q=bytes([1,0,0]); seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [13 if len(seen)==1 else 9]
        self.assertEqual(generate([q],predict,True),[[13]+[9]*8])
        self.assertEqual(seen[1],[1,10,9,9,13,9,9])
        self.assertEqual(seen[-1][-1],READOUT)
        self.assertEqual(len(seen[-1]),27)
    def test_control(self):
        seen=[]
        def predict(batch):
            seen.append(batch[0][:]); return [9]
        generate([bytes([7,0,0])],predict,False)
        self.assertEqual(len(seen[-1]),26)
        self.assertNotIn(READOUT,seen[-1])
    def test_invalid(self):
        with self.assertRaises(ValueError): encode(bytes([3,0,0]),[9]*9,True)
        with self.assertRaises(ValueError): encode(bytes([1,0,0]),[13]*9,True)
        with self.assertRaises(ValueError): generate([bytes([1,0,0])],lambda b:[265],True)
