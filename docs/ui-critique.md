# UI critique and fixes

A critique of the Streamlit UI (`ui/`), done by hand. The `/impeccable` and `emil-design-eng` skills weren't available, so I followed the same steps myself: **critique, then distill, then light motion**. I reviewed every page in Chromium with Playwright, plus the screenshots in `docs/screenshots/`, from the point of view of a first-time visitor. That visitor is either a student or a placement coordinator.

Severity: **High** = blocks understanding · **Med** = slows people down · **Low** = polish.

## 1. Critique: what was wrong

| # | Issue | Where | Severity | Fix |
|---|---|---|---|---|
| 1 | **No clear entry point.** The app opened on a coordinator chat. A student had no obvious path, and nothing said who each page is for. | App | High | A new **Home** page with two paths: *I'm a student* and *I'm a coordinator*. The navigation is grouped into the same two sections. |
| 2 | **The trace was written for developers.** It showed tool names (`rank_candidates`), raw JSON, millisecond timings, and 15–25 lines per request. | Ask, Students, Appeal dialog | High | By default it shows **one plain sentence per agent**, e.g. "Ranked the eligible students (72, 42) for 2 ML listings in MG Road…". A **Developer details** toggle shows everything else. |
| 3 | **Duplicate trace lines.** "LLM: summarise D0001" was followed by "summary (llm)" with the same text. | Trace | Med | Merged. Accepted LLM text appears only in the LLM step; a separate line appears only when the template was used. |
| 4 | **Text cut off mid-word**, e.g. "ML 3/3 · Pythoı" and summaries truncated at 160 characters. | Trace, tables | Med | `shorten()` cuts at a word boundary and adds "…". The table columns are wider. |
| 5 | **Students shown as bare ids** ("S025"). Hard to follow across tables, cards and appeals. | Everywhere | High | Synthetic names are shown as **"Ananya R. (S025)"**; the README says they are synthetic. |
| 6 | **Jargon**: "past", "fit", "Intent: rank · Mode: demo-safe", `skill_first`. | Tables, captions | Med | Plain words: "past opportunities", "match score" vs "skill fit", "commute 4.9 km". The intent/mode caption is only shown with developer details on. |
| 7 | **Slow answers in Gemini mode.** The Explainer made one LLM call per decision, and nothing appeared until the end. | Ask | Med | **One LLM call per request.** The tables render as soon as the Matcher and Auditor finish, and the verified explanation is revealed afterwards. |
| 8 | **Sidebar clutter**: a repeated brand block, a large data pill, and the LLM toggle given the same weight as navigation. | Sidebar | Med | The navigation is grouped by audience. The data source is one quiet line. The LLM mode lives in a collapsed **Settings** section. |
| 9 | **The Decision Log opened with five huge numbers** that repeated the filter chips. | Decision Log | Low | The counts are shown on the filter chips ("Proposed · 4"), and the metric row is gone. |
| 10 | **The Policy Lab was long and dense**: an explainer, a duplicate table and the charts all fully expanded. | Policy Lab | Low | The charts stay. "How to read this" and the table view move into expanders. The commute limit is one control above the charts. |
| 11 | **Inconsistent type and spacing**: 11 different font sizes (0.70–0.95rem) and ad-hoc margins. | Global | Low | CSS tokens: type scale 0.78 / 0.875 / 1 / 1.25 / 1.75 rem and spacing 4 / 8 / 12 / 16 / 24 px, used everywhere. |
| 12 | **Distracting motion**: the active pipeline chip glowed continuously, and every redraw re-rendered the whole list. | Trace | Low | No glow. Only the **newest** trace line fades in, in 160 ms. A single small pulsing dot marks the agent at work. |

## 2. Distill: what was removed

- The brand and tagline block in the sidebar (the Home page carries the name).
- The "Intent · Mode" caption under every answer (it's in developer details).
- The Decision Log metric row (its counts are on the chips).
- The "How it works" card text on Ask, shortened to one line (Home explains the two paths).
- The Policy Lab's always-open explainer and table view (now expanders, so a table view still exists).
- The 160-character mid-word truncation everywhere.

## 3. Motion (light, only where it helps)

- Buttons and path cards: 120 ms ease-out on the border and background when hovered; when pressed, they scale to 0.98 so the click feels acknowledged.
- The trace: only the newest line fades and rises 4 px, so earlier lines don't replay on every redraw.
- The explanation is revealed word by word once it has passed the number check, so nothing unverified is ever shown.
- Everything is turned off under `prefers-reduced-motion`.

## 4. Mechanism notes (from the same pass)

**Commute versus the outskirts.** Shorter commutes make an internship more practical, but penalising distance pushes out students who live on the city's edge. So the commute weight is a per-policy choice: 0.2 for skill-first, 0.1 for hybrid and 0 for equal access. The Equity Analyst then measures the *outskirts selection gap* (how much less often students living more than 12 km from MG Road get picked) and flags any gap above 10 points.
