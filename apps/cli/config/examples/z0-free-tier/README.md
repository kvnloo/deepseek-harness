# Zer0 free-tier provider overlay

This fork-local overlay exposes the built-in pi-ai `groq` and `cerebras` routes to DeepSeek Harness without storing provider credentials in DSH configuration.

## Secrets

Use a dedicated Bitwarden Secrets Manager project with exactly these secret keys:

- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`

The project id is supplied as `BWS_DSH_PROJECT_ID`. The BWS machine-account token stays in the existing OS-keyring/bootstrap path and is never committed here.

Launch from the repository root:

```sh
scripts/z0-dsh-bws-free-tier.sh
```

The wrapper uses `bws run --project-id`, verifies both injected variables are non-empty without printing them, and starts DSH with `cordis.patch.yml`.

## Provider catalog

Do not pin a `models:` list in this overlay. Groq and Cerebras are built-in pi-ai providers, so DSH should use the installed pi-ai catalog and its normal model-discovery/settings surfaces. This prevents the local overlay from becoming a second stale model registry.

Quota-aware model allocation belongs to z0intelligence + Kerdoios. DSH executes the selected provider/model and should not maintain a duplicate daily-quota scheduler.
