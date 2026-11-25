import tempfile
import unittest
from pathlib import Path
from vlanix_tiny.learning_receipt import LearningStore,hash_object,canonical
from vlanix_tiny.executor_lease import Rejected

class ReceiptTests(unittest.TestCase):
    def test_canonical(self):
        self.assertEqual(hash_object({'a':1,'b':2}),hash_object({'b':2,'a':1}))
        self.assertNotEqual(hash_object({'a':1}),hash_object({'a':2}))
        with self.assertRaises(ValueError): canonical({'x':float('nan')})
    def test_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=LearningStore(Path(tmp)/'s.db')
            with self.assertRaises(Rejected): s.commit('x','A',1,0,'a'*64,'b'*64)
    def test_no_unverified_genesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=LearningStore(Path(tmp)/'s.db')
            with self.assertRaises(ValueError): s.register('x','a'*64,[],[])
