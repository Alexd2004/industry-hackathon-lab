"""Side-by-side decisions (step 15): what A2 decided against what the rule would have decided.

diff(a2_block, rule_decision) is computed here, in code, never by the model. It returns
{field: [rule_value, a2_value]} for the fields both sides state and that differ, the shape the Loop tab
already reads (ui_loop.diff_table, log_line). It is empty when A2 did not produce a decision of its own
(not run, FALLBACK, no output): on a FALLBACK A2's output is the rule's decision, so there is nothing to
compare. A REPLAY block holds a recorded live output and is compared like a LIVE one.

format_diff(record) is the plain-English sentence for the Loop tab: the agent decided X because Y, the rule
would have decided Z.
"""
import math

from softsignal.agents.base import FALLBACK, LIVE, REPLAY

DIFF_FIELDS = ("blend_w", "cutoff", "cap", "cap_margin", "action")  # the Loop tab's rows (ui_loop.DIFF_ROWS)
COMPARED = (LIVE, REPLAY)
TOLERANCE = 1e-9  # floats within this are the same decision (0.15 against 0.15000000000000002)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(a, b, rel_tol=0, abs_tol=TOLERANCE)
    return type(a) is type(b) and a == b  # True is not 1


def a2_decision(a2_block: dict | None) -> dict | None:
    """A2's own decision (its output dict), or None when A2 has none to compare."""
    if not isinstance(a2_block, dict) or a2_block.get("status") not in COMPARED:
        return None
    out = a2_block.get("output")
    return out if isinstance(out, dict) else None


def diff(a2_block: dict | None, rule_decision: dict) -> dict:
    """{field: [rule_value, a2_value]} for each field both state (not None) and that differs. Empty when A2 has
    no decision of its own."""
    a2 = a2_decision(a2_block)
    if a2 is None:
        return {}
    return {k: [rule_decision[k], a2[k]] for k in DIFF_FIELDS
            if rule_decision.get(k) is not None and a2.get(k) is not None and not _same(rule_decision[k], a2[k])}


def diff_count(a2_block: dict | None, rule_decision: dict) -> int:
    return len(diff(a2_block, rule_decision))


def _fmt(k: str, v) -> str:
    if k == "cap" and _is_number(v):
        return f"cap {v:.0%}"
    if k == "cap_margin" and _is_number(v):
        return f"margin {v:.1%}"
    return f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"


def _decision_text(d: dict) -> str:
    return ", ".join(_fmt(k, d[k]) for k in ("action", "cap", "cap_margin", "cutoff") if d.get(k) is not None) or "no decision"


def format_diff(record: dict) -> str:
    """One sentence on A2 against the rule for a decisions.jsonl record. Never raises on a partial record."""
    rule = record.get("rule_decision") or {}
    applied = record.get("applied") or {}
    rule_text = _decision_text(rule)
    block = record.get("a2")
    a2 = a2_decision(block)
    if a2 is None:
        status = block.get("status") if isinstance(block, dict) else None
        if status == FALLBACK:
            why = block.get("fallback_reason") or "no reason given"
            return f"A2 fell back ({why}), so the rule decided: {rule_text}."
        return f"A2 did not decide this round, the rule decided: {rule_text}."
    a2_text = _decision_text(a2)
    reason = str(a2.get("reason") or "no reason given").strip()
    changed = diff(block, rule)
    if not changed:
        text = f"A2 decided {a2_text} because {reason} The rule would have decided the same."
    else:
        text = (f"A2 decided {a2_text} because {reason} The rule would have decided {rule_text} "
                f"(differs on {', '.join(changed)}).")
    if applied.get("source"):
        text += f" Applied: {applied['source']}."
    return text
