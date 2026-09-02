# Song World repository rules

## Objective

Build a local-first standalone macOS musical instrument. Phase 0 is a feasibility gate, not the final product; its proven engine and timing boundaries must remain reusable by the later application.

## Governance

- The initial empty/new-repository scaffold may be committed on `main`.
- Every subsequent implementation change uses `branch -> focused commits -> pull request -> user review/merge`.
- Never merge a pull request, force-push shared history, or implement directly on `main` after bootstrap.
- Stage explicit paths; never use broad staging commands such as `git add .` or `git add -A`.
- Run and report relevant builds, tests, and benchmarks before opening a pull request.

## Real-time guardrails

- The audio callback must not wait on UI, filesystem, network, model inference, allocation-heavy work, or contended locks.
- Deterministic transport owns source musical position; the model never owns chronology.
- Model-specific code stays behind `GenerativeEngine`; provide a deterministic mock implementation.
- Use a non-real-time inference thread and a bounded lock-free audio buffer.
- Home playback eventually means the untouched source master at a coherent source-musical position.
- Keep the UI in a separate top-level `frontend/` directory when UI work begins.

## Scope control

- Execute one approved phase at a time and stop at its gate.
- Do not commit commercial music, secrets, model weights, generated benchmark audio, or user-local paths.
- Optional AI failure must degrade to a usable deterministic audio path.
- Document measured evidence and exact blockers; never replace missing measurements with estimates.
