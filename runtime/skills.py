"""Runtime skill expression (issue #7 criterion 8).

At each decision point the loop evaluates regulation against current
agent-observable counters and exposes only currently expressed skills.
Stable gene/procedure ids are preserved on RuntimeSkill for future
attribution; evaluation-only `applicability` cannot participate because the
compiled RuntimeSkill carries no such field.
"""

from __future__ import annotations

from runtime.expression import RuntimeConfig, RuntimeSkill


def expressed_skills(config: RuntimeConfig, counters: dict) -> tuple[RuntimeSkill, ...]:
    """Deterministic for a given counter state: constitutive skills always
    express; regulated skills follow their `express_when` condition."""
    return tuple(skill for skill in config.skills
                 if config.is_active(skill.gene_id, counters))


def skill_block(skills) -> str:
    """Prompt-facing rendering of expressed skills (projected fields only)."""
    if not skills:
        return "No skills are currently expressed."
    lines = []
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.principle} (apply when: {skill.when_to_apply})")
        for step in skill.procedure:
            lines.append(f"    {step['id']}: {step['text']}")
    return "\n".join(lines)


__all__ = ["expressed_skills", "skill_block"]
