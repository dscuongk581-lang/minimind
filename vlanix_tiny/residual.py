"""P0.8A: empirical residual diagnostics and conservative trial gate."""
import json
import math
from statistics import mean


def decompose(groups):
    if not groups: raise ValueError('empty groups')
    stats=[]; residuals=[]; raw=[]
    for values in groups.values():
        x=list(values)
        if len(x)<4 or not all(math.isfinite(v) for v in x): raise ValueError('invalid group')
        m=mean(x); r=[v-m for v in x]; v=mean(z*z for z in r)
        stats.append((len(x),m,v,mean(z**4 for z in r)-3*v*v))
        residuals.extend(r); raw.extend(x)
    n=len(raw); mu=mean(raw); v=mean(z*z for z in residuals)
    within=sum(k*c for k,_,_,c in stats)/n
    between=3*sum(k*(s-v)**2 for k,_,s,_ in stats)/n
    k4=mean(z**4 for z in residuals)-3*v*v
    return dict(n=n,variance=v,raw_variance=mean((z-mu)**2 for z in raw),
        mean_variance=sum(k*(m-mu)**2 for k,m,_,_ in stats)/n,
        within_k4=within,between_k4=between,k4=k4,closure_error=k4-within-between,
        excess=k4/v**2 if v>0 else None)


def decide(current,added,history,before,after,retention_delta):
    """Prototype gate; thresholds are hypotheses, not calibrated defaults."""
    if type(current) is not int or type(added) is not int or current<=0 or added<=0:
        raise ValueError('invalid parameter counts')
    if not all(math.isfinite(x) for x in [*history,before,after,retention_delta]):
        raise ValueError('nonfinite metric')
    if min(before,after)<0: raise ValueError('negative loss')
    if current+added>30_000_000: return 'BUDGET'
    if len(history)<3 or any(abs(x)<.05 for x in history[-3:]) or not (all(x>0 for x in history[-3:]) or all(x<0 for x in history[-3:])):
        return 'NOT_PERSISTENT'
    if retention_delta>.01: return 'RETENTION'
    if before-after<.001: return 'NO_GAIN'
    return 'PROBE_ACCEPT'


def toy_probe(kind):
    """Analytic gradients of 0.5*(w*z-y)^2 at w=0, one probe step.
    The base is zero; z=0 on retention samples. Not a Transformer test.
    """
    if kind not in ('structured','noise'): raise ValueError('kind')
    def samples(offset):
        return [(z, z if kind=='structured' else (1 if j%2==0 else -1))
                for z in (-1,1) for j in range(4+2*offset)]
    def loss(rows,w): return mean(.5*(w*z-y)**2 for z,y in rows)
    train=samples(0); val=samples(1); old=[(0,0)]*4
    history=[mean(-z*y for z,y in samples(i)) for i in range(3)]
    w=-.1*history[0]
    before,after=loss(val,0),loss(val,w)
    retention=loss(old,w)-loss(old,0)
    return dict(kind=kind,gradient_history=history,weight=w,validation_before=before,
        validation_after=after,validation_gain=before-after,retention_delta=retention,
        decision=decide(6_000_000,24576,history,before,after,retention))


if __name__=='__main__':
    print(json.dumps({'schema':'ngc-residual-toy-v1','probes':[toy_probe(k) for k in ('structured','noise')],
        'decomposition':decompose({'a':[-1,1,-1,1],'b':[-2,2,-2,2,-2,2,-2,2]})},sort_keys=True))
