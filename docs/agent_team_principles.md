# Design principles: ATOM (junyyang-amd/ATOM, kimi-k3-tpot line)

A designer, a design reviewer and an agent ruling on an escalation weigh a
decision against the agent-team skill's default principles
(`principles.md`) and the project principles below. The project principles
win a conflict. Both decide how to investigate, never what: goals and
acceptance belong to the requirement in the umbrella issue.

## Project principles

1. **Compare like with like.** Every A/B runs both images on the same node,
   with the same launch settings, model, dataset and concurrency, and
   changes one thing at a time. Why: the earlier comparison mixed nodes,
   dates and ATOM commits, which is why its cause is unknown. Source:
   `logs/perf-cmp/rocm10.1-vs-7.2.4-tables.md` (Limits).
2. **A number has a spread.** Each measured point is repeated (at least 3
   runs) and reported as mean and min-max; a difference inside the spread is
   not a finding. Why: TW nodes vary by several percent run to run.
   Source: LLM-Booster#351, the 11-09 tables.
3. **Run only on the node the owner named.** GPU work uses pit2-p03-g07
   alone, one GPU job at a time; another node needs a ruling. Why: the
   owner holds it; others belong to other users or are reserved. Source:
   the owner's decision of 2026-11-16 on #62, junyyang-amd/ATOM#30
   (two Rulings on node placement).
4. **Settings follow the CI case.** A reproduction uses the launch settings
   of the ATOMesh case it reproduces (`pd_matrix.py` / the case config), and
   the report states every deviation. Why: the regression is defined by
   those cases. Source: LLM-Booster#351.
5. **Unit tests run without a GPU.** Any script logic that parses traces or
   computes the breakdown gets a CPU test under `tests/`. Why: the gate is
   CPU-only. Source: `CLAUDE.md` (Build & Test & Lint).
