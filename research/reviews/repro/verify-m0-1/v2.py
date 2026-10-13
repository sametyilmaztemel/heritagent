from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
# seed robustness: min point with LB>0 over seeds, plus point 0.15 (3/20) check
for seed in range(5):
    m=[]
    for pos in range(21):
        for neg in range(21-pos):
            d=[1.0]*pos+[-1.0]*neg+[0.0]*(20-pos-neg)
            b=paired_cluster_bootstrap(d,seed=seed,n_resamples=10000)
            if b.lower_bound>0: m.append(b.point_estimate)
    print(seed,min(m),len(m))
