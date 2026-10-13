from scipy.stats import binom
ns={'Pick':24,'Clean':31,'Heat':23,'Cool':21,'Look':18,'Pick2':17}
base={'Pick':75.0,'Clean':57.4,'Heat':13.0,'Cool':60.0,'Look':18.9,'Pick2':36.5}
for f,v in base.items():
    n=ns[f]; ks=[k for k in range(n+1) if abs(round(100*k/n,1)-v)<0.06]
    print(f,n,v,'integer-consistent k:',ks, 'nearest', min(range(n+1),key=lambda k:abs(100*k/n-v)), round(100*min(range(n+1),key=lambda k:abs(100*k/n-v))/n,1))
print('overall 45.5 ->',[k for k in range(135) if abs(round(100*k/134,1)-45.5)<0.06], '57.5 ->',[k for k in range(135) if abs(round(100*k/134,1)-57.5)<0.06],'59.4->',[k for k in range(135) if abs(round(100*k/134,1)-59.4)<0.06],'46.3 ->',[k for k in range(135) if abs(round(100*k/134,1)-46.3)<0.06],'48.5->',[k for k in range(135) if abs(round(100*k/134,1)-48.5)<0.06],'54.5->',[k for k in range(135) if abs(round(100*k/134,1)-54.5)<0.06])
print('GiGPO 14.8 as k/128:',[k for k in range(129) if abs(round(100*k/128,1)-14.8)<0.06],' k/140:',[k for k in range(141) if abs(round(100*k/140,1)-14.8)<0.06],' k/134:',[k for k in range(135) if abs(round(100*k/134,1)-14.8)<0.06])
for v in [31.2,42.7,48.0,60.3]:
    print(v,'k/128',[k for k in range(129) if abs(round(100*k/128,1)-v)<0.06],'k/140',[k for k in range(141) if abs(round(100*k/140,1)-v)<0.06],'k/134',[k for k in range(135) if abs(round(100*k/134,1)-v)<0.06])
print('P(G0 solves >=40 of 60) by p:')
for p in [0.15,0.30,0.45,0.55,0.65,0.70,0.80]:
    print(p, round(binom.sf(39,60,p),4), 'E solved',60*p)
# family quotas in T_mine: 60 tasks proportional to train availability with min 8
tr={'Pick':790,'Look':308,'Clean':650,'Heat':459,'Cool':533,'Pick2':813}
tot=sum(tr.values()); print(tot)
print({k:round(60*v/tot,1) for k,v in tr.items()})
need={k:8+4+10 for k in tr}  # mine min + replay + gen (excluding ablation/interaction share)
print('min per-family train need (mine8+replay4+gen10=22) vs avail',{k:(22,v,round(v/22,1)) for k,v in tr.items()})
# unseen eval counts
un={'Pick':24,'Look':18,'Clean':31,'Heat':23,'Cool':21,'Pick2':17}
print('T_eval per-family sums',sum(un.values()))
