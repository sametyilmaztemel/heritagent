"""Regulation evaluator tests (issue #7 criterion 5)."""

import pytest

from genome.validation.errors import GenomeValidationError
from runtime.regulation import Condition, Predicate


def test_operators():
    counters = {"consecutive_failures": 2}
    assert Predicate("consecutive_failures", ">=", 2).evaluate(counters)
    assert Predicate("consecutive_failures", ">", 1).evaluate(counters)
    assert Predicate("consecutive_failures", "==", 2).evaluate(counters)
    assert not Predicate("consecutive_failures", "<", 2).evaluate(counters)
    assert Predicate("consecutive_failures", "<=", 2).evaluate(counters)


def test_missing_counter_is_unsatisfied_deterministically():
    assert not Predicate("steps_without_progress", ">=", 0).evaluate({})


def test_unknown_metric_and_operator_fail_closed():
    with pytest.raises(GenomeValidationError, match="non-observable metric"):
        Predicate("task_type", "==", "x").evaluate({})
    with pytest.raises(GenomeValidationError, match="unknown operator"):
        Predicate("failing_tools", "~=", 1).evaluate({"failing_tools": 1})


def test_all_of_requires_every_predicate():
    condition = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),
                                  Predicate("failing_tools", ">=", 5)))
    assert condition.evaluate({"consecutive_failures": 2, "failing_tools": 5})
    assert not condition.evaluate({"consecutive_failures": 2, "failing_tools": 4})


def test_any_of_requires_at_least_one():
    condition = Condition(any_of=(Predicate("consecutive_failures", ">=", 3),
                                  Predicate("repeated_tool_error", ">=", 2)))
    assert condition.evaluate({"consecutive_failures": 0, "repeated_tool_error": 2})
    assert not condition.evaluate({"consecutive_failures": 1, "repeated_tool_error": 1})


def test_combined_all_and_any_semantics():
    condition = Condition(all_of=(Predicate("steps_without_progress", ">=", 1),),
                          any_of=(Predicate("consecutive_failures", ">=", 2),
                                  Predicate("failing_tools", ">=", 9)))
    assert condition.evaluate({"steps_without_progress": 1, "consecutive_failures": 2})
    assert not condition.evaluate({"steps_without_progress": 1, "consecutive_failures": 1,
                                   "failing_tools": 3})


def test_from_genome_and_constitutive():
    condition = Condition.from_genome(
        {"express_when": {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}})
    assert not condition.is_constitutive()
    assert Condition.from_genome({}).is_constitutive()
    assert Condition.from_genome({"express_when": {}}).is_constitutive()


def test_malformed_genome_rule_fails_closed():
    with pytest.raises(GenomeValidationError, match="malformed regulation predicate"):
        Condition.from_genome({"express_when": {"all_of": [{"metric": "consecutive_failures"}]}})


def test_evaluation_is_deterministic():
    counters = {"consecutive_failures": 2, "failing_tools": 3}
    condition = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),))
    assert condition.evaluate(counters) == condition.evaluate(dict(counters))
