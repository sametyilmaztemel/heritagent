"""C2: compare the study's ABSOLUTE effect model (rescue prob delta/(1-p0) among G0 failures,
mean effect = delta) with a RELATIVE reading (rescue prob = delta among failures,
mean effect = delta*(1-p0)). f=0, e=1, all five stages."""
import sys; sys.path.insert(0, "<scratch>/out/repro/verify-power")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import c2_c4_stages as m
for p0 in (0.3, 0.5):
    for d in (0.1, 0.2):
        absol = m.replay_pass(p0, d, 0) * m.ablation_pass(p0, d, 0)
        rel_d = d * (1 - p0)
        rel = m.replay_pass(p0, rel_d, 0) * m.ablation_pass(p0, rel_d, 0)
        print(f"p0={p0} effect={d}: absolute-model false-reject={1-absol:.4f} | relative-model (abs effect {rel_d:.3f}) false-reject={1-rel:.4f}")
