from inheritance.interaction import InteractionAllocation
from inheritance.runner import EvaluationTask
alloc = InteractionAllocation(tasks=tuple(EvaluationTask(task_id=f"ix-{i}", family="", payload={}) for i in range(10)))
print("bank_id=None / family='' accepted at construction:", alloc.allocation_metadata()["tasks"][0])
