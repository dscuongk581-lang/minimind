"""Bounded history and explicit-state recurrence, no oracle feedback."""


def validate(qs):
    if not qs: raise ValueError('empty batch')
    for q in qs:
        if len(q)!=4 or any(type(v) is not int for v in q): raise ValueError('frame')
        op,a,b,w=q
        if op not in (1,2,7,8) or not 1<=w<=32 or not 0<=a<1<<w or not 0<=b<1<<w:
            raise ValueError('frame')
    w=qs[0][3]
    if any(q[3]!=w for q in qs): raise ValueError('mixed widths')
    return w


def predict_ids(predict,batch):
    ids=predict([p[:] for p in batch])
    if len(ids)!=len(batch) or any(type(v) is not int or not 0<=v<265 for v in ids):
        raise ValueError('prediction contract')
    return ids


def bounded_rollout(qs,predict,window=8):
    w=validate(qs)
    if type(window) is not int or not 1<=window<=32: raise ValueError('window')
    histories=[[] for q in qs]; out=[[] for q in qs]
    for k in range(w):
        pairs=[[9+((a>>k)&1),9+((b>>k)&1)] for _,a,b,_ in qs]
        batch=[[1,q[0]+9]+(h[-3*(window-1):] if window>1 else [])+p
               for q,h,p in zip(qs,histories,pairs)]
        for h,o,p,v in zip(histories,out,pairs,predict_ids(predict,batch)):
            h.extend(p+[v]); o.append(v)
            if len(h)>3*window: del h[:-3*window]
    return out


def reset_rollout(qs,predict):
    w=validate(qs); carry=[0]*len(qs); out=[[] for q in qs]
    for k in range(w):
        active=[i for i,c in enumerate(carry) if c is not None]
        batch=[[1,qs[i][0]+9,9+((qs[i][1]>>k)&1),9+((qs[i][2]>>k)&1),9+carry[i]] for i in active]
        ids=dict(zip(active,predict_ids(predict,batch))) if batch else {}
        for i,o in enumerate(out):
            v=ids.get(i); o.append(v)
            carry[i]=(v-9)>>1 if v is not None and 9<=v<=12 else None
    return out
