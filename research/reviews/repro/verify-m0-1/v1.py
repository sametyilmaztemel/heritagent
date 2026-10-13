from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
import inspect
res={}
for pos in range(21):
    for neg in range(21-pos):
        d=[1.0]*pos+[-1.0]*neg+[0.0]*(20-pos-neg)
        b=paired_cluster_bootstrap(d,seed=0,n_resamples=10000)
        res[(pos,neg)]=(b.point_estimate,b.lower_bound)
for tc in (0.0,0.03,0.05,0.08,0.15,0.2):
    print(tc,sum(stage_passes(p,l,tau_c=tc) for p,l in res.values()))
lb=[(k,v) for k,v in res.items() if v[1]>0]
print("min point with LB>0:",min(v[0] for k,v in lb))
print("n LB>0:",len(lb))
print("diff grid:",sum(len({stage_passes(p,l,tau_c=t) for t in (0.03,0.05,0.08)})>1 for p,l in res.values()))
