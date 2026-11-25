import unittest
from vlanix_tiny.virtual_ram import VirtualRAM,BLOCK,root

class RAMTests(unittest.TestCase):
    def test_replication_and_dedup(self):
        ram=VirtualRAM(); data=b'x'*BLOCK*3+b'end'
        m,h=ram.put(data)
        self.assertEqual(len(m['blocks']),4)
        self.assertEqual(sum(len(n) for n in ram.nodes),4)
        for n in range(3):
            ram.online={0,1,2}-{n}
            self.assertEqual(ram.get(m,h),data)
    def test_corrupt_replica_and_loss(self):
        ram=VirtualRAM(); m,h=ram.put(b'hello'); key=m['blocks'][0]
        holders=[n for n in range(3) if key in ram.nodes[n]]
        ram.nodes[holders[0]][key]=b'xxxxx'
        self.assertEqual(ram.get(m,h),b'hello')
        ram.online.remove(holders[1])
        with self.assertRaises(ValueError): ram.get(m,h)
    def test_manifest_order_and_root(self):
        ram=VirtualRAM(); m,h=ram.put(b'a'*BLOCK+b'b'*BLOCK)
        bad=dict(m,blocks=m['blocks'][::-1])
        with self.assertRaises(ValueError): ram.get(bad,h)
        with self.assertRaises(ValueError): ram.get(bad,root(bad))
        with self.assertRaises(ValueError): ram.get(dict(m,size=-1),h)
    def test_empty_and_publish_guard(self):
        ram=VirtualRAM(); m,h=ram.put(b'')
        self.assertEqual(ram.get(m,h),b'')
        ram.online.remove(0)
        with self.assertRaises(RuntimeError): ram.put(b'new')
