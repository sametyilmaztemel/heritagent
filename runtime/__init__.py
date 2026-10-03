"""HeritAgent single-agent runtime.

Model-adapter-independent agent core: ModelAdapter contract + backends,
EAG expression layer, regulation evaluator, seed policies, and the
minimal execution loop with explicit budgets and event hooks (issue #7).
No benchmark logic and no trajectory persistence live here (SPEC §21, §28;
#8 records the events this loop emits).
"""

__version__ = "0.1.0"
