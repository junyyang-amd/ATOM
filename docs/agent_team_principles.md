# Design principles: ATOM (junyyang-amd/ATOM, EPD line)

A designer, a design reviewer and an agent ruling on an escalation weigh a
decision against two lists: the agent-team skill's default principles
(`principles.md`, eight of them, each named below with how it applies
here) and the project principles. The project principles win a conflict.
Both decide how to build, never what: goals and acceptance belong to the
requirement in the umbrella issue.

No human Rulings exist on this repository yet, so each project principle
below comes from the project's own documents.

## The defaults, and where each bears on this work

- **The simplest solution that meets acceptance.** Acceptance asks for
  accuracy parity and TTFT no worse than PD. A design that adds machinery
  past that, such as an embedding cache (a non-goal), pays for code nobody
  asked for.
- **Measure before deciding.** Whether embeddings travel over the existing
  KV connector or a new channel, and whether EPD helps TTFT, turn on
  numbers. A spike measures them before the design picks a transfer.
- **Reuse existing code and dependencies first.** The PD path already moves
  KV over Mooncake / the KV connector, and the engine already hashes media
  (`multimodal_cache_seed`). Reuse both before writing a second transport
  or a second hash.
- **No generality for speculative needs.** Build for images on
  `/v1/chat/completions` with the two named models. A media-type
  abstraction for video or audio waits until those inputs are in scope.
- **Fix shared code, not callers.** Teach the shared multimodal generation
  path to return `kv_transfer_params`. Do not patch each entrypoint.
- **Prefer reversible choices.** The encoder-only role and ATOMesh's E role
  are off unless configured. Single-instance and PD deployments behave as
  before when they are off.
- **Follow the project's existing conventions.** ATOMesh's E role follows
  how the P and D roles are declared and discovered today. Engine flags
  follow `atom/config.py` and `atom/utils/envs.py`.
- **Change public interfaces only when the requirement does.** The
  OpenAI-compatible request and response shapes stay as they are. New
  flags for the encoder role extend the existing CLI.

## Project principles

1. **Unit tests run without a GPU.** New logic gets a CPU test under
   `tests/` (or a Rust test in `atom/mesh`) that mocks AITER and
   `torch.cuda` the way the suite already does. GPU behaviour is proven by
   the acceptance harness, not the gate. Why: the PR gate is CPU-only.
   Source: `CLAUDE.md` (Build & Test & Lint),
   `.github/scripts/run_unit_tests.sh`.
2. **A text-only request takes the code path it took before.**
   Multimodal handling branches on the presence of media. Text requests
   keep their routing keys, payloads and scheduling unchanged. Why:
   production text traffic (AgentX) uses no images and must not move.
   Source: LLM-Booster#455 (AgentX needs no multimodal), ROCm/ATOM#3161.
3. **Every `ATOM_*` environment variable is defined in
   `atom/utils/envs.py`.** Why: one place documents and defaults them.
   Source: `CLAUDE.md` (Key entry points: Env vars).
4. **Accuracy is compared against direct ATOM with the same image,
   settings and samples.** A recipe number is a sanity check, not the
   reference. Why: it isolates what ATOMesh or disaggregation changes.
   Source: LLM-Booster#455 (Acceptance).
5. **Launch settings follow the model's recipe.** A topology deviates from
   `recipes/MiniMax-M3.md` or `recipes/Kimi-K3.1.md` only where
   disaggregation requires it, and the design says where. Source:
   LLM-Booster#455 (Task card: Environment, Topology).
