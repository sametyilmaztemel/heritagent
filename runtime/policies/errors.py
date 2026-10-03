"""Policy-layer errors (fail-closed parsing)."""


class PlanParseError(Exception):
    """A planner reply did not match the required format."""
