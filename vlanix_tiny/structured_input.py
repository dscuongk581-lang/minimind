"""Lossless research encoding; no arithmetic outputs or carry in prompt."""
OPS=(1,2,7,8)

def encode_prompt(q,mode):
    if not isinstance(q,bytes) or len(q)!=3 or q[0] not in OPS:
        raise ValueError('frame')
    op,a,b=q
    if mode=='raw': return [1,op+9,a+9,b+9]
    if mode=='planes':
        return [1,op+9]+[9+((v>>k)&1) for k in range(8) for v in (a,b)]
    raise ValueError('encoding')

def decode_prompt(p,mode):
    if not isinstance(p,list) or any(type(v) is not int for v in p) or len(p)<2 or p[0]!=1 or p[1]-9 not in OPS:
        raise ValueError('prompt')
    op=p[1]-9
    if mode=='raw' and len(p)==4 and all(9<=v<=264 for v in p[2:]):
        return bytes((op,p[2]-9,p[3]-9))
    if mode=='planes' and len(p)==18 and all(v in (9,10) for v in p[2:]):
        return bytes((op,*(sum((p[2+2*k+j]-9)<<k for k in range(8)) for j in (0,1))))
    raise ValueError('encoding')
