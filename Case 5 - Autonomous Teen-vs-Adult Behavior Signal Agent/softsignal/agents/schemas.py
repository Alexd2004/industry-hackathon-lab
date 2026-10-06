"""Output schemas for the agents (Combined Plan section 7a, per-agent contract). One Pydantic model per agent,
sent as output_config.format and checked again in code.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

A4_MAX_NOTE_CHARS = 600
A4_MAX_CITES = 5


class A4Output(BaseModel):
    """A4 verify-band triager: one short note for the human reviewer of this batch's verify band."""

    model_config = ConfigDict(extra="forbid")

    batch_reason: str = Field(
        min_length=1, max_length=A4_MAX_NOTE_CHARS,
        description="Plain-language note for the reviewer, under 600 characters. Every number in it must "
                    "appear in the input exactly as written there.",
    )
    based_on: list[str] = Field(
        min_length=1, max_length=A4_MAX_CITES,
        description="1 to 5 feature keys, copied exactly from the input's signals[].feature, that the note "
                    "relies on, most important first.",
    )


A1_MAX_REASON_CHARS = 300
A1_MAX_EVIDENCE = 4


class A1Evidence(BaseModel):
    """One input value A1 relies on: field is a path from contracts.a1_fields() ("psi.activity_max")."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="A field path copied exactly from the input's fields list.")
    value: float = Field(description="That field's value, copied exactly from the input.")


class A1Output(BaseModel):
    """A1 drift watcher: is the drift in this batch real, or noise?"""

    model_config = ConfigDict(extra="forbid")

    drift: Literal["real", "not_real", "insufficient_data"]
    evidence: list[A1Evidence] = Field(max_length=A1_MAX_EVIDENCE,
                                       description="Up to 4 input values the verdict rests on.")
    reason: str = Field(min_length=1, max_length=A1_MAX_REASON_CHARS,
                        description="One or two plain sentences, under 300 characters. Every number in it must "
                                    "appear in the input exactly as written there.")


A2_MAX_REASON_CHARS = 400
A2_MAX_CITES = 5


class A2Output(BaseModel):
    """A2 loop controller: this round's action, false-teen cap and cap margin. The stack has no blend_w and the plan
    gives no mapping from a cutoff to t_verify / t_soft (Combined Plan 5b, open decision 5), so neither is in the
    schema. cap_margin is the one model lever: how far below the cap the verify cutoff aims. An out-of-range cap
    or margin is clamped in code, not rejected."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["hold", "re-tune", "promote"] = Field(
        description="hold keeps the current thresholds, re-tune refits and re-thresholds, promote moves "
                    "SHADOW to ACTIVE. Only choose what input.guards allows.",
    )
    cap: float = Field(
        strict=True, allow_inf_nan=False,  # a bool or a string is not a cap (lax mode would take true as 1.0)
        description="Cap on the false-teen rate (share of adults sent to verification) for this round, "
                    "as a fraction, e.g. 0.15. Stay within input.bounds.",
    )
    cap_margin: float | None = Field(
        default=None, strict=True, allow_inf_nan=False,
        description="How far below the cap the verify cutoff aims, as a fraction between 0 and 0.05, e.g. 0.02. "
                    "A larger margin lowers false-teen overshoot and costs some recall. null keeps the policy value.",
    )
    refit_window: int | None = Field(
        default=None, strict=True,
        description="Refit on the labels of only the last this-many rounds (at least 2), to drop data from before "
                    "a drift. null refits on all rounds. Leave null unless A1 reports real drift.",
    )
    reason: str = Field(
        min_length=1, max_length=A2_MAX_REASON_CHARS,
        description="Why this action and cap, under 400 characters. Every number in it must appear in the "
                    "input exactly as written there.",
    )
    cites: list[str] = Field(
        min_length=1, max_length=A2_MAX_CITES,
        description="1 to 5 dotted field paths copied exactly from the input (for example audit.audit_adults) "
                    "that the reason relies on, most important first.",
    )


A3_MAX_DESC_CHARS = 250
A3_MAX_REASON_CHARS = 200
# Kept small: every A3 call that reached the model timed out at 4 s, asked for up to 4 x 4 evidence items and 3
# changes (about 700-1000 output tokens against A1's 125). Two of each still cover a count and a signal per pattern.
A3_MAX_PATTERNS = 2
A3_MAX_EVIDENCE = 2
A3_MAX_CHANGES = 2
A3_PARAMS = ("cap", "cutoff", "blend_w")  # the plan's enum; advisory only, A2 acts on action and cap alone


class A3Evidence(BaseModel):
    """One input value a pattern rests on: field is a path from contracts.a3_fields() ("false_teen.n_accounts")."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="A field path copied exactly from the input's fields list.")
    value: float = Field(description="That field's value, copied exactly from the input.")


class A3Pattern(BaseModel):
    """One recurring trait among the audit accounts the live model got wrong."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal["false_teen", "missed_teen"]
    description: str = Field(min_length=1, max_length=A3_MAX_DESC_CHARS,
                             description="The trait the errors share, under 250 characters. Every number in it "
                                         "must appear in the input exactly as written there.")
    n_accounts: int = Field(strict=True, ge=1, description="Accounts of this error type that show the trait, "
                                                           "copied from the input; never above that type's total.")
    evidence: list[A3Evidence] = Field(min_length=1, max_length=A3_MAX_EVIDENCE)


class A3Change(BaseModel):
    """An advisory parameter change. A2 reads it as data and decides alone."""

    model_config = ConfigDict(extra="forbid")

    param: Literal["cap", "cutoff", "blend_w"]
    direction: Literal["up", "down"]
    reason: str = Field(min_length=1, max_length=A3_MAX_REASON_CHARS,
                        description="Why, under 200 characters. Every number in it must appear in the input "
                                    "exactly as written there.")


class A3Output(BaseModel):
    """A3 error analyst: patterns among the audit-slice errors of earlier rounds, and advisory changes."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "insufficient_data"]
    patterns: list[A3Pattern] = Field(max_length=A3_MAX_PATTERNS)
    suggested_param_changes: list[A3Change] = Field(max_length=A3_MAX_CHANGES)


A5_VERDICTS = ("supported", "unsupported", "projected", "cannot_check")
A5_MAX_NOTE_CHARS = 200
A5_MAX_RISKS = 4
A5_MAX_CLAIMS = 40


class A5Verdict(BaseModel):
    """A5's verdict on one claim (Combined Plan 7a: {claim, verdict, source: file + row}, plus the risk tags)."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(description="The claim's id, copied exactly from input.claims.")
    verdict: Literal["supported", "unsupported", "projected", "cannot_check"]
    source: str | None = Field(description="supported / projected: the id of the row that holds every number of "
                                           "the claim. unsupported: that row id, a file name, or null. "
                                           "cannot_check: a file name or null.")
    risks: list[str] = Field(max_length=A5_MAX_RISKS, description="Ids from input.checklist that apply, or [].")
    note: str = Field(max_length=A5_MAX_NOTE_CHARS, description="Why, under 200 characters. Every number in it "
                                                                "must appear in the input exactly as written.")


class A5Output(BaseModel):
    """A5 honesty auditor: one verdict per input claim."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[A5Verdict] = Field(min_length=1, max_length=A5_MAX_CLAIMS)
