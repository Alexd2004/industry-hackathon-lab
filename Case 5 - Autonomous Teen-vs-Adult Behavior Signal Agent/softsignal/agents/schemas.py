"""Output schemas for the agents (Combined Plan section 7a, per-agent contract). One Pydantic model per agent,
passed to client.messages.parse() as output_format and checked again in code. A2 and A4 are built so far; each
owner adds theirs here (A1 drift, A3 patterns, A5 verdicts).
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


A2_MAX_REASON_CHARS = 400
A2_MAX_CITES = 5


class A2Output(BaseModel):
    """A2 loop controller: this round's action and false-teen cap. Only these two are A2's to choose: the
    stack has no blend_w and the plan gives no mapping from a cutoff to t_verify / t_soft (Combined Plan 5b,
    open decision 5), so neither is in the schema. An out-of-range cap is clamped in code, not rejected."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["hold", "re-tune", "promote"] = Field(
        description="hold keeps the current thresholds, re-tune refits and re-thresholds, promote moves "
                    "SHADOW to ACTIVE. Only choose what input.guards allows.",
    )
    cap: float = Field(
        allow_inf_nan=False,
        description="Cap on the false-teen rate (share of adults sent to verification) for this round, "
                    "as a fraction, e.g. 0.15. Stay within input.bounds.",
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
