import sys, json, copy
sys.path.insert(0, "/home/user/heritagent")
from genome.validation.loader import load_genome
from genome.validation.registry import TraitRegistry
from genome.validation.projection import project_skill, project_skillrl
from genome.validation.errors import GenomeValidationError
from runtime.expression import compile_runtime_config
from runtime.skills import skill_block

import tempfile, pathlib; reg = TraitRegistry(pathlib.Path(tempfile.mkdtemp(dir="<scratch>/out/repro/refute-V5-armB-representation")))
skill = {"name":"Check drawers","principle":"Search receptacles systematically","when_to_apply":"object not visible",
         "procedure":[{"id":"S1","text":"open each drawer"}],"applicability":{"task_families":["pick"]}}
uri = reg.put("skill","rejected_skill_v1",1,skill)
print("uri", uri)
def g(origin, prov):
    return {"genome_id":"G-abcdef12","schema_version":"0.1","generation":1,"lineage_id":"L1","parent":"G-0f1e2d3c",
            "genes":{"skills":[{"gene_id":"rejected_skill_v1","version":1,"type":"skill","artifact":uri,"origin":origin,"provenance":prov}]}}
cases = {
 "seed_no_cig": g("seed", {"born_generation":0}),
 "assim_reject_cig": g("assimilation", {"born_generation":1,"source_trajectory":"T-1","cig_record":"CIG-REJECTED-0001"}),
 "assim_missing_cig": g("assimilation", {"born_generation":1,"source_trajectory":"T-1"}),
 "acquired": g("acquired", {"born_generation":1,"source_trajectory":"T-1"}),
 "unrestricted": g("unrestricted_inheritance", {"born_generation":1}),
}
for k,v in cases.items():
    try:
        load_genome(copy.deepcopy(v), reg); print(k, "ACCEPTED")
    except GenomeValidationError as e:
        print(k, "REJECTED", str(e)[:160])
cfg = compile_runtime_config(cases["assim_reject_cig"], reg)
print("render:\n"+skill_block(cfg.skills))
print("skillrl:", project_skillrl(skill))
