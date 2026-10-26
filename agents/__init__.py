"""
The five FairMatch agents (PLAN.md section 5) and the orchestrator that runs them.

Agents only call the deterministic tools in tools/. The LLM is used for two
things only: reading the request (Planner) and rephrasing tool output
(Equity takeaway, Explainer summary), and both are checked before use.
"""
