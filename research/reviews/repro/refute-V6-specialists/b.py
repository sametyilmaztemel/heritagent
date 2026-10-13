import sys
sys.path.insert(0,'/home/user/heritagent')
from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
for w,l in [(1,0),(2,0),(3,0),(4,0),(5,1),(6,1),(5,0),(3,1)]:
    d=[1.0]*w+[-1.0]*l+[0.0]*(20-w-l)
    b=paired_cluster_bootstrap(d,seed=0)
    print(w,l,b.point_estimate,b.lower_bound,stage_passes(b.point_estimate,b.lower_bound))
# proportional allocation
c={'pick':790,'look':308,'clean':650,'heat':459,'cool':533,'pick2':813}
t=sum(c.values()); q={k:20*v/t for k,v in c.items()}
a={k:int(v) for k,v in q.items()}
for k in sorted(q,key=lambda k:-(q[k]-a[k]))[:20-sum(a.values())]: a[k]+=1
print(q,a)
