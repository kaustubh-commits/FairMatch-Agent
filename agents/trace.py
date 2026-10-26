"""
The live agent trace.

Every step an agent takes (planning, a tool call, an LLM call, a fallback)
becomes a TraceEvent:

    {agent, action, tool, args, result_summary, duration_ms}

Events are handed to an `on_event` callback the moment they happen, so a UI
can draw them step by step while the run is still going (PLAN.md section 5).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

# One colour per agent, used by the CLI now and the Streamlit UI in Phase 3.
AGENT_COLORS = {
    "System": "#6b7280",          # grey
    "Planner": "#7c3aed",         # violet
    "Matcher": "#2563eb",         # blue
    "Equity Analyst": "#d97706",  # amber
    "Auditor": "#059669",         # green
    "Explainer": "#db2777",       # pink
}


@dataclass
class TraceEvent:
    agent: str
    action: str
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    result_summary: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Trace:
    """Collects events and forwards each one to `on_event` immediately."""

    def __init__(self, on_event: Optional[Callable[[TraceEvent], None]] = None):
        self.events: list[TraceEvent] = []
        self.on_event = on_event

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def note(self, agent: str, action: str, summary: str, **args) -> None:
        """An instant event with no tool call (a decision, a warning, a fallback)."""
        self.emit(TraceEvent(agent=agent, action=action, args=args, result_summary=summary))

    @contextmanager
    def step(self, agent: str, action: str, tool: Optional[str] = None, **args):
        """
        Time a block of work and emit ONE event when it finishes:

            with trace.step("Matcher", "rank", tool="rank_candidates", k=3) as ev:
                result = rank_candidates(...)
                ev.result_summary = "top: S025 (0.94)"
        """
        event = TraceEvent(agent=agent, action=action, tool=tool, args=args)
        start = time.perf_counter()
        try:
            yield event
        finally:
            event.duration_ms = round((time.perf_counter() - start) * 1000)
            self.emit(event)
