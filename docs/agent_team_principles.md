# Design principles: ATOM (junyyang-amd/ATOM, RL line)

A designer, a design reviewer and an agent ruling on an escalation weigh a
decision against the agent-team skill's default principles (`principles.md`
in the agent-team skill: the simplest solution that meets acceptance,
measure before deciding, reuse first, no speculative generality, fix shared
code, prefer reversible choices, follow conventions, change public
interfaces only when the requirement does) and the project principles
below. The project principles win a conflict. Both decide how to build,
never what: goals and acceptance belong to the requirement in the umbrella
issue.

## Project principles

1. **Unit tests run without a GPU.** New logic gets a CPU test under
   `tests/` (or a Rust test in `atom/mesh`) that mocks AITER and
   `torch.cuda` the way the suite already does; GPU behaviour is proven by
   `tools/rl/accept.py`, not the gate. Why: the PR gate is CPU-only.
   Source: `CLAUDE.md` (Build & Test & Lint).
2. **With RL off, nothing changes.** The control plane and the engine
   endpoints are inert unless enabled (`--enable-rl` on ATOMesh); inference
   requests take the path they took before. Why: production inference
   (AgentX) uses neither. Source: LLM-Booster#456 (SMG's `--enable-rl`),
   LLM-Booster#455.
3. **SMG's protocol is the spec.** Where ATOMesh's RL routes, selectors,
   bodies or responses differ from smg@6c7bda1, the difference is a bug
   unless the design names it and why. Why: frameworks on `smg.rl` must
   need no change. Source: LLM-Booster#456 (Client).
4. **Correctness is compared against a server started fresh on the same
   checkpoint**, with the same settings and prompts, never against a
   remembered number. Why: it isolates what the refit changes. Source:
   LLM-Booster#456 (Acceptance).
5. **GPU work runs only on the nodes the owner names**, and a node that
   becomes unavailable is the owner's to replace. Source: Rulings
   [#30](https://github.com/junyyang-amd/ATOM/issues/30#issuecomment-6826722899),
   [#30](https://github.com/junyyang-amd/ATOM/issues/30#issuecomment-6867816403),
   [#63](https://github.com/junyyang-amd/ATOM/issues/63#issuecomment-6909252785).
6. **Every `ATOM_*` environment variable is defined in
   `atom/utils/envs.py`.** Why: one place documents and defaults them.
   Source: `CLAUDE.md` (Key entry points: Env vars).
