"""Scripted LLM for unit tests: replays canned responses and records every call."""

from collections import deque
from collections.abc import Iterable
from typing import Any


class FakeLLM:
    """Returns ``responses`` in order; :attr:`calls` keeps system/user/json_schema."""

    def __init__(self, responses: Iterable[str], model_name: str = "fake") -> None:
        self.model_name = model_name
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if not self._responses:
            raise RuntimeError("FakeLLM has no scripted responses left")
        return self._responses.popleft()
