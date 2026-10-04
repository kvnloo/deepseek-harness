# @deepseek-ai/dsh-z0-memory-context

Opt-in bridge from DSH's native `agent/pre-step` seam to
`z0int context resolve`.

The bridge is deliberately narrow:

- `mode: off` by default.
- `shadow` retrieves but does not change the model request.
- `inject` appends one durable, source-attributed context message on step 1.
- ctx remains read-only: z0 invokes it with `--refresh off`.
- lexical ctx is the default; semantic/hybrid retrieval is not enabled here.
- retrieved memory is rendered as untrusted data, never instruction/authority.
- exact model-visible text is SHA-256 stamped in the message source metadata.

The local `z0int` executable and ctx index are external prerequisites.
Use the example patch at
`apps/cli/config/examples/z0-memory-context.cordis.yml`.
