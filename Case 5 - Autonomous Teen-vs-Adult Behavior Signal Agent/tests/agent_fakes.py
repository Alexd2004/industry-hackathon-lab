"""Test doubles for the agents' model calls (no network): a fake Anthropic client and Message-like replies."""
import json
from types import SimpleNamespace

import httpx2

REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeClient:
    """Stands in for anthropic.Anthropic: messages.create returns `reply` (or raises it); calls and the options
    the agent set (timeout, retries) are recorded."""

    def __init__(self, reply=None, raises=None):
        self.calls, self.options, self.reply, self.raises = [], [], reply, raises
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raises is not None:
            raise self.raises
        return self.reply


def reply(output=None, stop_reason="end_turn", text=None):
    """A Message-like reply: one text block holding output as JSON (or the given raw text)."""
    body = text if text is not None else (json.dumps(output) if output is not None else "")
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=[SimpleNamespace(type="text", text=body)] if body else [],
                           usage=SimpleNamespace(input_tokens=1200, output_tokens=80, cache_creation_input_tokens=None,
                                                 cache_read_input_tokens=None))
