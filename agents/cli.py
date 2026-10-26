"""
Command-line runner for the agents (Phase 2 checkpoint).

    python -m agents.cli "Find the best 3 candidates for ML internships in Koramangala"
    python -m agents.cli --mode demo-safe "Compare the allocation policies"
    python -m agents.cli "Explain decision D0001"
    python -m agents.cli "Appeal D0001 for S011: I have strong ML project experience"

Trace lines are printed live, one per step, coloured by agent.
"""

from __future__ import annotations

import argparse
import sys

from agents.orchestrator import run
from agents.trace import AGENT_COLORS, TraceEvent

RESET = "\033[0m"


def _ansi(hex_color: str) -> str:
    """Turn '#2563eb' into a 24-bit terminal colour code."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"\033[1;38;2;{r};{g};{b}m"


def print_event(event: TraceEvent, color: bool = True) -> None:
    agent = f"{event.agent:<15}"
    if color:
        agent = _ansi(AGENT_COLORS.get(event.agent, "#000000")) + agent + RESET
    tool = f" [{event.tool}]" if event.tool else ""
    timing = f" ({event.duration_ms} ms)" if event.duration_ms else ""
    print(f"  {agent} {event.action}{tool}{timing}")
    if event.result_summary:
        print(f"  {'':<15}   -> {event.result_summary}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FairMatch agents on one request.")
    parser.add_argument("request", help="what you want, in plain English")
    parser.add_argument("--mode", choices=["gemini", "ollama", "demo-safe"], default=None,
                        help="LLM provider (default: LLM_PROVIDER in .env, else gemini)")
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    args = parser.parse_args()

    print("TRACE")
    result = run(args.request, provider=args.mode,
                 on_event=lambda e: print_event(e, color=not args.no_color))
    print(f"\nANSWER  (intent: {result.intent}; mode: {result.mode_start}"
          + (f" -> {result.mode_end}" if result.mode_end != result.mode_start else "") + ")")
    print(result.answer)


if __name__ == "__main__":
    main()
