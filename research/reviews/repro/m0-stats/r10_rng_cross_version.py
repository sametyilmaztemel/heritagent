"""R10 - Cross-interpreter reproducibility of the bootstrap stream.

bootstrap.py:9-10 claims "explicit deterministic seed via random.Random - no
SciPy, no hidden version-dependent defaults". The CPython docs guarantee
cross-version reproducibility only for Random.random() with a compatible
seeder; randrange()/_randbelow are explicitly "subject to change". This
script prints a digest of 10k-resample outputs so the result can be compared
across interpreters (run it once per venv).

Run from /home/user/heritagent with each interpreter:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venvXY>/bin/python <this file>
"""
import hashlib
import sys

from inheritance.bootstrap import paired_cluster_bootstrap

vectors = [
    [1.0] * 10 + [-1.0] * 3 + [0.0] * 7,
    [1.0] + [0.0] * 19,
    [float(i % 3 - 1) for i in range(20)],
]
digest = hashlib.sha256()
for v in vectors:
    for seed in (0, 1, 11, 12345):
        r = paired_cluster_bootstrap(v, seed=seed)
        digest.update(repr((r.point_estimate, r.lower_bound, r.upper_bound)).encode())
print(f"{sys.version.split()[0]} digest={digest.hexdigest()[:16]} "
      f"example LB(pos10/neg3, seed0)={paired_cluster_bootstrap(vectors[0], seed=0).lower_bound}")
