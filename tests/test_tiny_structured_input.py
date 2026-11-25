import unittest
from vlanix_tiny.structured_input import encode_prompt, decode_prompt

class StructuredInputTests(unittest.TestCase):
    def test_example(self):
        self.assertEqual(encode_prompt(bytes([2,1,2]),'raw'),[1,11,10,11])
        self.assertEqual(encode_prompt(bytes([2,1,2]),'planes'),[1,11,10,9,9,10]+[9]*12)
    def test_roundtrip(self):
        for a in range(256):
            for b in range(256):
                q=bytes([1,a,b])
                self.assertEqual(decode_prompt(encode_prompt(q,'planes'),'planes'),q)
    def test_invalid(self):
        for q in (b'',bytes([3,1,2]),bytes([1,2,3,4])):
            with self.assertRaises(ValueError): encode_prompt(q,'planes')
        with self.assertRaises(ValueError): encode_prompt(bytes([1,2,3]),'unknown')
        with self.assertRaises(ValueError): decode_prompt([1,10]+[11]*16,'planes')
    def test_modes(self):
        for op in (1,2,7,8):
            q=bytes([op,255,128])
            for mode,length in [('raw',4),('planes',18)]:
                p=encode_prompt(q,mode)
                self.assertEqual(len(p),length)
                self.assertEqual(decode_prompt(p,mode),q)
