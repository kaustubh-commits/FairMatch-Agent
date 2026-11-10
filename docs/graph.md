# The FairMatch agent graph (LangGraph)

Generated from the compiled graph by `python -m agents.graph`; do not edit by hand.
Rendered image: [screenshots/langgraph.png](screenshots/langgraph.png).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	planner(planner)
	matcher(matcher)
	equity_analyst(equity_analyst)
	auditor(auditor)
	explainer(explainer)
	respond(respond)
	human_approval(human_approval)
	__end__([<p>__end__</p>]):::last
	__start__ --> planner;
	auditor -. &nbsp;checked&nbsp; .-> explainer;
	auditor -. &nbsp;no such decision&nbsp; .-> respond;
	equity_analyst --> auditor;
	explainer -. &nbsp;done&nbsp; .-> __end__;
	explainer -. &nbsp;rank: decisions proposed&nbsp; .-> human_approval;
	human_approval -. &nbsp;all approved&nbsp; .-> __end__;
	matcher -. &nbsp;rank / compare&nbsp; .-> equity_analyst;
	matcher -. &nbsp;recommend&nbsp; .-> explainer;
	matcher -. &nbsp;no listings found&nbsp; .-> respond;
	planner -. &nbsp;explain / appeal&nbsp; .-> auditor;
	planner -. &nbsp;rank / compare / recommend&nbsp; .-> matcher;
	planner -. &nbsp;help, or no skills given&nbsp; .-> respond;
	respond --> __end__;
	human_approval -. &nbsp;more to approve&nbsp; .-> human_approval;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| Node | Calls (unchanged) |
|---|---|
| `planner` | `agents.planner.plan` (rules, or a checked LLM reading of the request) |
| `matcher` | `agents.matcher.rank` / `compare` / `recommend` |
| `equity_analyst` | `agents.equity.check_proposals` / `compare_policies` |
| `auditor` | `agents.auditor.verify_and_log` / `check_allocations` / `fetch` / `appeal` |
| `explainer` | `agents.explainer.explain_all` / `explain_matches` / `explain` + shared answer formatting |
| `respond` | short answers: the help text, "tell me your skills", "no such decision" |
| `human_approval` | `interrupt()`: pauses until a person presses **Approve**, then `tools.decisions.approve_decision` |

Solid arrows always run; dotted arrows are conditional routes, labelled with the intent
or outcome that takes them. The routes match the fixed steps in `agents/planner.py` (`STEPS`).
