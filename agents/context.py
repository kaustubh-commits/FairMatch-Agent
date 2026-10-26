"""
Shared state for one agent run: the trace and the (optional) LLM.

`ask_llm` is the ONLY way agents talk to the LLM. If a call fails, it
switches the whole run to demo-safe mode, records that in the trace, and
returns None. The agent then uses its rule-based path instead.
"""

from __future__ import annotations

import json
from typing import Optional

from agents.trace import Trace
from llm import DEMO_SAFE, LLMClient, LLMError


class RunContext:
    def __init__(self, llm: Optional[LLMClient], trace: Trace):
        self.llm = llm
        self.trace = trace

    @property
    def mode(self) -> str:
        """The LLM in use (e.g. 'gemini:gemini-2.5-flash -> ollama:llama3.2') or 'demo-safe'."""
        return self.llm.name if self.llm else DEMO_SAFE

    def ask_llm(self, agent: str, purpose: str, prompt: str,
                json_mode: bool = False) -> Optional[str | dict]:
        """
        Ask the LLM. Returns its text (or parsed JSON when json_mode=True),
        or None in demo-safe mode / after a failure.
        """
        if self.llm is None:
            return None
        with self.trace.step(agent, f"LLM: {purpose}", tool="llm", provider=self.llm.name) as ev:
            try:
                text = self.llm.complete(prompt, json_mode=json_mode)
                result = json.loads(text) if json_mode else text
                used = getattr(self.llm, "last_used", None) or self.llm.name
                ev.tool = f"llm ({used})"
                ev.result_summary = (json.dumps(result) if json_mode else text)[:160]
                return result
            except (LLMError, json.JSONDecodeError) as e:
                ev.result_summary = f"FAILED: {str(e)[:160]}"
                failed = self.llm.name
        # Outside the `with` so this event comes after the failed call in the trace.
        self.llm = None
        self.trace.note("System", "fallback to demo-safe",
                        f"{failed} failed; continuing in demo-safe mode (rule-based, no LLM).")
        return None
