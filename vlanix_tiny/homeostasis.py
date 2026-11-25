"""Bounded discrete PI controller; drift is in nats per answer distribution."""
import math

class PI:
    def __init__(self,target=.025,kp=4.,ki=1.,minimum=.05,smoothing=.5):
        if not all(math.isfinite(x) for x in (target,kp,ki,minimum,smoothing)) or not (target>0 and kp>=0 and ki>0 and 0<minimum<=1 and 0<smoothing<=1):
            raise ValueError('controller configuration')
        self.target=target; self.kp=kp; self.ki=ki
        self.minimum=minimum; self.smoothing=smoothing
        self.integral=0.; self.filtered=0.

    def update(self,drift):
        if not math.isfinite(drift) or drift<0: raise ValueError('invalid drift')
        self.filtered=(1-self.smoothing)*self.filtered+self.smoothing*drift
        error=self.filtered-self.target
        candidate=max(0.,min((1-self.minimum)/self.ki,self.integral+error))
        raw=1-self.kp*error-self.ki*candidate
        # Do not integrate further into saturation; allow recovery.
        if not ((raw<self.minimum and error>0) or (raw>1 and error<0)):
            self.integral=candidate
        elif error<0:
            self.integral=candidate
        return max(self.minimum,min(1.,1-self.kp*error-self.ki*self.integral))
