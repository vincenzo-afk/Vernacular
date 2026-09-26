"""
Fake LLM client for register detection / translation tests. Scripted,
deterministic — no network calls. See TESTING.md.
"""

from dataclasses import dataclass


@dataclass
class ScriptedResponse:
    """One canned response the fake will return, in order."""

    content: str
    should_raise: bool = False


class FakeLLMClient:
    """
    Records every call made to it and returns scripted responses in
    order. Raises RuntimeError if more calls are made than responses
    were scripted (fail loudly in tests rather than silently returning
    None).
    """

    def __init__(self, responses: list[ScriptedResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def queue(self, response: ScriptedResponse) -> None:
        self._responses.append(response)

    async def complete(self, *, system: str, user: str, **kwargs) -> str:
        self.calls.append({"system": system, "user": user, **kwargs})
        if not self._responses:
            raise RuntimeError(
                "FakeLLMClient.complete() called with no scripted responses "
                "remaining — add more ScriptedResponse entries in your test."
            )
        response = self._responses.pop(0)
        if response.should_raise:
            raise RuntimeError("Scripted failure for test")
        return response.content
