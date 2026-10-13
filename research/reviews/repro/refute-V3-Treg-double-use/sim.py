import numpy as np
rng=np.random.default_rng(0)
# null: all K=10 traits have identical true per-task flip prob f (no trait is truly better).
# stage-4 selection on T_reg (40 tasks, <=1 flip). Composite regression = union of flips (additive-flip assumption, U).
def run(f,K=10,n=40,reps=20000):
    out=[]
    for _ in range(reps):
        flipT=rng.random((K,n))<f      # T_reg flips per trait
        flipH=rng.random((K,n))<f      # fresh held-out previously-solved tasks
        sel=flipT.sum(1)<=1
        if sel.sum()==0: continue
        regB_T=flipT.any(0).mean(); regC_T=flipT[sel].any(0).mean()
        regB_H=flipH.any(0).mean(); regC_H=flipH[sel].any(0).mean()
        out.append((sel.sum(),regB_T,regC_T,regB_H,regC_H))
    a=np.array(out); return a.mean(0), len(out)/reps
for f in (0.02,0.03,0.05):
    m,pne=run(f)
    print(f"f={f}: P(C nonempty)={pne:.2f} E|C|={m[0]:.2f} | T_reg: B={m[1]:.3f} C={m[2]:.3f} gap={m[1]-m[2]:.3f} | heldout: B={m[3]:.3f} C={m[4]:.3f} gap={m[3]-m[4]:.3f}")
