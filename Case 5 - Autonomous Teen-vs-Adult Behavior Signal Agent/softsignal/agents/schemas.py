"""Output schemas for the agents (Combined Plan section 7a, per-agent contract). One Pydantic model per agent,
sent as output_config.format and checked again in code. A1 and A4 are built; each owner adds theirs here
(A2 decision, A3 patterns, A5 verdicts).
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
