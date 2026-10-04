"""A1 drift watcher (Tier 3, step 16; Crew Plan section 3.1).

Question: is the drift in this batch real, or noise? A1 reads the PSI of this batch against the batches
seen before it (per feature group and for the live scores), the audit counts revealed so far, and the
earlier rounds' PSI. It informs A2 (stage 1, in parallel with A3) and changes nothing itself: the hold rule
(min_audit_adults before any re-tune) is enforced in loop.py, not left to A1.

Input: contracts.a1_input() (counts and PSI only: no rows, no ids, no labels). Output: schemas.A1Output
{drift: real / not_real / insufficient_data, evidence: [{field, value}], reason}. Checks: the schema; every
evidence item names an input field and copies its value exactly; every number in the reason is in the
input; no numeric age. Fallback: the policy.yaml threshold (psi_drift): drift is real when the largest
group or score PSI reaches it, badged FALLBACK. No reference rows yet (round 0, round 1) or no threshold
returns insufficient_data without a model call.

What to expect (Crew Plan): batches are random draws from one population, so most rounds read not_real.
calibrate() measures that: python -m softsignal.agents.a1_drift --calibrate. Nothing stages drift.

A1 runs on the decision path, so it keeps the 4 s timeout (base.TIMEOUT_S).
"""
import argparse
import json

from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import (
    AGE_CLAIM, FALLBACK, INSUFFICIENT, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, UNKNOWN_FIELD, UNSUPPORTED, AgentResult,
    age_claims, call_model, input_hash, numbers_not_in_input,
)
from softsignal.agents.contracts import PSI_GROUPS, a1_fields, floor_psi
from softsignal.agents.schemas import A1_MAX_REASON_CHARS, A1Output

AGENT = "A1"
REAL, NOT_REAL = "real", "not_real"
PSI_KEYS = ("score", "activity_max", "text_max")  # what the fallback compares with the threshold
SYSTEM = f"""You are A1, the drift watcher in SoftSignal, a system that estimates whether an account belongs to \
a teen or an adult from writing style and app activity. Each round a new batch of accounts arrives. Decide \
whether the drift in this batch is real or noise. You inform the loop controller; you change nothing.

The input is JSON:
- psi: population stability index of this batch against the batches seen in earlier rounds, floored to 3 \
decimals. score is the PSI of the live rule's scores (null when there is no earlier score history, or once the \
live model changes every round). For each feature group (n_features gives the number of app-activity and \
writing-style columns): the largest feature PSI (_max), the mean (_mean) and the feature with the largest PSI \
(_top_feature).
- psi_drift: the threshold the fallback rule uses (drift is real at or above it).
- psi_conventions: the usual PSI levels (under stable: stable; stable to large: moderate; over large: large).
- audit: audit accounts revealed so far (adults, teens).
- history: the earlier rounds' PSI (score, activity_max, text_max).
- n_reference, n_batch: accounts in the earlier batches and in this batch.
- fields: every numeric field you may cite as evidence, as path -> value.

Small batches are noisy, so a single moderate value is usually noise. A shift that persists across rounds in \
history, or a value at or above psi_drift, is more likely real.

Rules:
- Use only the input. Batches are random draws from one population, so not_real is the expected answer most \
rounds. Say so plainly; never invent a drift to look busy.
- evidence: up to 4 items, each a path copied exactly from fields with its value copied exactly. A real \
verdict must cite at least one psi field.
- Every number in reason must appear in the input exactly as written there. Do not compute new numbers.
- insufficient_data only if the input lacks what you need.
- reason: one or two plain sentences, at most {A1_MAX_REASON_CHARS} characters, no markdown."""


def user_message(payload: dict) -> str:
    """The fresh per-call prompt: the input (plus its citable fields) in a tagged block, nothing else."""
    body = {**payload, "fields": a1_fields(payload)}
    return (f"Batch for round {payload['round']}. Is the drift real?\n"
            f"<input>\n{json.dumps(body, separators=(',', ':'), default=str)}\n</input>")


def _insufficient(payload: dict) -> bool:
    return payload["n_reference"] == 0 or payload["n_batch"] == 0 or payload["psi_drift"] is None


def fallback_output(payload: dict) -> dict:
    """The fixed-threshold rule: real if the largest of the score and feature-group PSIs reaches psi_drift.

    Built from the input's own values, so it passes validate_output (tested).
    """
    if _insufficient(payload):
        why = ("no earlier batch to compare with yet" if payload["n_reference"] == 0 or payload["n_batch"] == 0
               else "no psi_drift threshold in policy.yaml")
        return {"drift": INSUFFICIENT, "evidence": [], "reason": f"Insufficient data: {why}."}
    p, threshold = payload["psi"], payload["psi_drift"]
    key = max((k for k in PSI_KEYS if p.get(k) is not None), key=lambda k: (p[k], k))
    top = p[key]
    drift = REAL if top >= threshold else NOT_REAL
    where = f"{p[key.replace('_max', '_top_feature')]}, " if key != "score" else ""
    reason = (f"Largest PSI is {top} ({where}psi.{key}), {'at or above' if drift == REAL else 'below'} the "
              f"{threshold} threshold.")
    return {"drift": drift, "evidence": [{"field": f"psi.{key}", "value": top},
                                         {"field": "psi_drift", "value": threshold}], "reason": reason}


def validate_output(output: dict, payload: dict) -> tuple[str | None, list[str]]:
    """(fallback reason or None, errors): evidence names input fields with their exact values, a real verdict
    cites at least one PSI, every number in the reason is in the input, no numeric age."""
    if age_claims(output["reason"]):
        return AGE_CLAIM, [f"states an age: {age_claims(output['reason'])}"]
    fields = a1_fields(payload)
    for ev in output["evidence"]:
        if ev["field"] not in fields or abs(fields[ev["field"]] - ev["value"]) > 1e-9:
            return UNKNOWN_FIELD, [f"evidence {ev} does not match an input field and its value"]
    if output["drift"] == REAL and not any(ev["field"].startswith("psi.") for ev in output["evidence"]):
        return UNSUPPORTED, ["drift is real but no psi value is cited as evidence"]
    invented = numbers_not_in_input(output["reason"], payload)
    if invented:
        return NUMBER_NOT_IN_INPUT, [f"numbers not in the input: {invented}"]
    return None, []


def _fallback(payload: dict, h: str, reason: str, errors: list[str], timer: AgentTimer | None, round_id,
              rejected: str | None = None) -> AgentResult:
    """The deterministic path, timed as a tool call so the round summary shows A1 as FALLBACK."""
    rnd = {} if round_id is None else {"round_id": round_id}
    if timer is None:
        output = fallback_output(payload)
    else:
        with timer.call(AGENT, "fallback", "tool", status=FALLBACK, **rnd):
            output = fallback_output(payload)
    return AgentResult(AGENT, FALLBACK, output, reason, h, errors, rejected)


def run_a1(payload: dict, client=None, timer: AgentTimer | None = None, round_id=None) -> AgentResult:
    """A1 on one round's input. client: base.make_client() (None = offline: the threshold rule, FALLBACK)."""
    h = input_hash(payload)
    if _insufficient(payload):  # nothing to compare: no model call
        return _fallback(payload, h, INSUFFICIENT, [], timer, round_id)
    if client is None:
        return _fallback(payload, h, OFFLINE, [], timer, round_id)
    reply = call_model(client, agent=AGENT, step="drift", system=SYSTEM, user=user_message(payload),
                       schema=A1Output, check=lambda out: validate_output(out, payload), timer=timer,
                       round_id=round_id)
    if reply.fallback_reason is not None:
        return _fallback(payload, h, reply.fallback_reason, reply.errors, timer, round_id, reply.raw)
    return AgentResult(AGENT, LIVE, reply.output, None, h)


def calibrate(seeds=range(10)) -> "pd.DataFrame":
    """The measurement behind policy.yaml psi_drift: each oracle batch's PSI against the earlier batches, for
    every seed (stratified random draws from one population, so no drift by construction). One row per
    (seed, round >= 2) with the group maxima and the starter-score PSI, floored like A1's input."""
    import pandas as pd

    from softsignal.data import load_data
    from softsignal.features import ID_COL
    from softsignal.loop import starter_score
    from softsignal.metrics import psi
    from softsignal.oracle import Oracle
    from softsignal.agents.contracts import group_psi

    train, test = load_data(on_param_mismatch="error")
    rows = []
    for k in seeds:
        seen = []
        for b in Oracle(train, test[ID_COL].tolist(), seed=42 + k):
            if seen:
                ref = pd.concat(seen, ignore_index=True)
                g = group_psi(ref, b.rows)
                rows.append({"seed": 42 + k, "round": b.round, **{f"{grp}_max": g[f"{grp}_max"] for grp in PSI_GROUPS},
                             "score": floor_psi(psi(starter_score(ref), starter_score(b.rows)))})
            seen.append(b.rows)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="A1 drift watcher tools")
    ap.add_argument("--calibrate", action="store_true", help="measure no-drift PSI against policy.yaml psi_drift")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args(argv)
    if not args.calibrate:
        ap.print_help()
        return
    from softsignal.policy import load_policy

    threshold = load_policy()["psi_drift"]
    d = calibrate(range(args.seeds))
    top = d[[c for c in d.columns if c not in ("seed", "round")]].max(axis=1)
    alarms = int((top >= threshold).sum()) if threshold is not None else None
    print(d.describe(percentiles=[0.5, 0.95, 0.99]).drop(columns=["seed", "round"]).round(3).to_string())
    print(f"\n{len(d)} no-drift batches; largest PSI {top.max():.3f}; psi_drift {threshold}; false alarms {alarms}")
    if alarms == 0:  # rule of three: 0 of n bounds the per-batch rate at about 3/n with 95% confidence
        print(f"0 false alarms in {len(d)} batches bounds the per-batch false-alarm rate to about {3 / len(d):.0%} (95%)")


if __name__ == "__main__":
    main()
