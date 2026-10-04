"""Output schemas for the agents (Combined Plan section 7a, per-agent contract). One Pydantic model per agent,
passed to client.messages.parse() as output_format and checked again in code. Only A4 is built so far; each
owner adds theirs here (A1 drift, A2 decision, A3 patterns, A5 verdicts).
"""
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
