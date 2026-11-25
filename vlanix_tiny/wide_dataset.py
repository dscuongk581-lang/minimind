"""Compose predicted bit transitions and byte readout at wider widths."""
OPS=(1,2,7,8,10)


def validate(op,a,b,width):
    if type(width) is not int or width not in (8,16,32,64) or type(op) is not int or op not in OPS:
        raise ValueError('operation or width')
    if any(type(v) is not int or not 0<=v<1<<width for v in (a,b)):
        raise ValueError('operand')


def compose(op,a,b,width,local,readout):
    validate(op,a,b,width)
    carry=0; out=0
    for base in range(0,width,8):
        trace=[]
        for k in range(base,base+8):
            p=local((op,(a>>k)&1,(b>>k)&1,carry))
            if type(p) is not int or not 0<=p<=3: return None
            trace.append(p+9); carry=p>>1
        p=readout(tuple(trace))
        if type(p) is not int or not 9<=p<=264: return None
        out|=(p-9)<<base
    return out


def reference(op,a,b,width):
    validate(op,a,b,width)
    return {1:a+b,2:a^b,7:a&b,8:a|b,10:a-b}[op]&((1<<width)-1)
