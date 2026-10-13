import numpy as np
from scipy import stats
rng=np.random.default_rng(12345)

def boot_power(n,delta,d,reps=1500,B=2000,crit='lb>0'):
    """paired design: b=P(C wins,B loses), c=P(B wins, C loses); d=b+c discordance; delta=b-c.
    cluster bootstrap over tasks (percentile 95% CI) -> multinomial resampling of (+1,-1,0) outcomes."""
    if delta>d: return float("nan")
    b=(d+delta)/2; c=(d-delta)/2
    probs=[b,c,1-b-c]
    ok=0
    for _ in range(reps):
        cnt=rng.multinomial(n,probs)
        # resample counts
        bs=rng.multinomial(n,np.array(cnt)/n,size=B)
        diffs=(bs[:,0]-bs[:,1])/n
        lo=np.percentile(diffs,2.5)
        if lo>0 and (cnt[0]-cnt[1])/n>=0: ok+=1
    return ok/reps

print("== Paired power (cluster bootstrap 95% CI excludes 0), n=134 tasks, P(CI lb>0)")
print("delta  d=0.10  d=0.20  d=0.30  d=0.40")
for delta in [0.05,0.075,0.10,0.15]:
    row=[boot_power(134,delta,d,reps=1000,B=1000) for d in [0.10,0.20,0.30,0.40]]
    print(f"{delta:.3f} "+"  ".join(f"{x:.2f}" for x in row))
print("== MDE (normal approx, 80% power, two-sided .05) = 2.8*sqrt(d/n) for n=134")
for d in [0.05,0.10,0.20,0.30,0.40]:
    print(d, round(2.8*np.sqrt(d/134),3))
print("== Wald SE of single-arm success at n=134: p=.15,.3,.45 ->",[round(np.sqrt(p*(1-p)/134),3) for p in [.15,.3,.45]])
print("== n=40 regression suite, MDE 80% power 2.8*sqrt(d/40):")
for d in [0.05,0.10,0.20,0.30]:
    print(d, round(2.8*np.sqrt(d/40),3))

# Stage-2 exact: n=20 paired tasks, pass iff percentile-bootstrap LB>0 and point>=0.05
from scipy.stats import multinomial
def stage2_pass(b,c,n=20):
    if (b-c)/n<0.05-1e-12: return False
    p=[b/n,c/n,1-(b+c)/n]
    # exact distribution of b*-c* under multinomial resample
    # enumerate
    prob_le0=0.0
    for i in range(n+1):
        for j in range(n+1-i):
            if i-j<=0:
                prob_le0+=multinomial.pmf([i,j,n-i-j],n,p)
    return prob_le0<0.025
print("== Stage-2 (n=20) pass table: rows b (wins), cols c (losses)")
for b in range(0,9):
    print(b,[int(stage2_pass(b,c)) for c in range(0,5)])
# Stage-1 replay: 5 instances need >=3/5 success-with and net delta >= +2/5
print("== done")
