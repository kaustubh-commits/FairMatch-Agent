"""
Shared state for one agent run: the trace and the (optional) LLM.

`ask_llm` is the ONLY way agents talk to the LLM. If a call fails, it
switches the whole run to demo-safe mode, records that in the trace with a
friendly line (raw details kept in the event's args for an expander), and
returns None. The agent then uses its rule-based path instead.
"""

from __future__ import annotations

import json
from typing import Optional

from agents.trace import Trace, shorten
from llm import DEMO_SAFE, LLMClient, LLMError
from llm.client import describe, headline


class RunContext:
    def __init__(self, llm: Optional[LLMClient], trace: Trace):
        self.llm = llm
        self.trace = trace

    @property
    def mode(self) -> str:
        """The LLM in use (e.g. 'gemini:gemini-3.5-flash (fallbacks: ...)') or 'demo-safe'."""
        return self.llm.name if self.llm else DEMO_SAFE

    def ask_llm(self, agent: str, purpose: str, prompt: str,
                json_mode: bool = False) -> Optional[str | dict]:
        """
        Ask the LLM. Returns its text (or parsed JSON when json_mode=True),
        or None in demo-safe mode / after a failure.
        """
        if self.llm is None:
            return None
        failure: Optional[LLMError] = None
        with self.trace.step(agent, f"LLM: {purpose}", tool="llm") as ev:
            try:
                text = self.llm.complete(prompt, json_mode=json_mode)
                result = json.loads(text) if json_mode else text
                used = getattr(self.llm, "last_used", None) or self.llm.name
                ev.tool = f"llm ({used})"
                raw = json.dumps(result) if json_mode else text
                ev.result_summary = shorten(raw, 400)
                if len(raw) > 400:
                    ev.args["raw_output"] = raw          # full text for the developer view
            except LLMError as e:
                failure = e
                ev.result_summary = f"Failed: {describe(e)}"
                ev.args["error_detail"] = str(e)[:2000]
            except json.JSONDecodeError as e:
                failure = LLMError(f"reply was not valid JSON: {e}", "bad_reply", self.llm.name)
                ev.result_summary = "Failed: the LLM's reply was not valid JSON"
                ev.args["error_detail"] = str(e)

        if failure is None:
            # A backup model answered after the main one failed: say so, briefly.
            earlier = getattr(self.llm, "last_errors", [])
            if earlier:
                summary = LLMError("", earlier[0].kind, earlier[0].provider)
                summary.errors = earlier
                backup = self.llm.last_used.split(":", 1)[-1]
                self.trace.note("System", "LLM fallback", f"{describe(summary)}; answered by {backup}.",
                                error_detail="; ".join(f"{e.provider}: {e}" for e in earlier)[:2000])
                self.trace.headline("System", f"{headline(summary)}; switched to {backup}.")
            return result

        # Outside the `with`, so this event comes after the failed call in the trace.
        self.llm = None
        self.trace.note("System", "fallback to demo-safe",
                        f"{headline(failure)}, continuing rule-based (demo-safe).")
        self.trace.headline("System", f"{headline(failure)}, continuing rule-based.")
        return None
