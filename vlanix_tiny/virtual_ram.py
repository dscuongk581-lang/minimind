"""Local immutable block-store simulation; not a network or durable RAM service."""
import hashlib
import json
import re

BLOCK=65536
LIMIT=512*1024*1024
SCHEMA='ngc-ram-manifest-v1'

def sha(data): return hashlib.sha256(data).hexdigest()
def root(manifest):
    return sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())

def validate(m,expected_root):
    if not isinstance(m,dict) or set(m)!={'schema','block_size','size','blocks','sha256'}:
        raise ValueError('manifest fields')
    if m['schema']!=SCHEMA or m['block_size']!=BLOCK or type(m['size']) is not int or not 0<=m['size']<=LIMIT:
        raise ValueError('manifest bounds')
    blocks=m['blocks']
    if not isinstance(blocks,list) or len(blocks)!=(m['size']+BLOCK-1)//BLOCK:
        raise ValueError('block count')
    if not all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in [m['sha256']]+blocks):
        raise ValueError('digest syntax')
    if root(m)!=expected_root: raise ValueError('manifest root')

class VirtualRAM:
    def __init__(self):
        self.nodes=[{}, {}, {}]
        self.online={0,1,2}

    def put(self,data):
        if not isinstance(data,bytes) or len(data)>LIMIT: raise ValueError('payload bounds')
        if self.online!={0,1,2}: raise RuntimeError('publish requires all simulated nodes')
        blocks=[]
        for i,off in enumerate(range(0,len(data),BLOCK)):
            part=data[off:off+BLOCK]; h=sha(part); blocks.append(h)
            # Stable placement by content; two logical copies per unique digest.
            slot=int(h[:8],16)%3
            for n in (slot,(slot+1)%3):
                self.nodes[n][h]=part
        m=dict(schema=SCHEMA,block_size=BLOCK,size=len(data),blocks=blocks,sha256=sha(data))
        return m,root(m)

    def get(self,m,expected_root):
        validate(m,expected_root)
        out=bytearray()
        for i,h in enumerate(m['blocks']):
            length=min(BLOCK,m['size']-i*BLOCK)
            for n in sorted(self.online):
                part=self.nodes[n].get(h)
                if isinstance(part,bytes) and len(part)==length and sha(part)==h:
                    out.extend(part); break
            else: raise ValueError('no verified replica for block '+h)
        if len(out)!=m['size'] or sha(out)!=m['sha256']:
            raise ValueError('payload digest')
        return bytes(out)
