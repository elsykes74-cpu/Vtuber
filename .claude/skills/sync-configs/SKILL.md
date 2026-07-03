---
name: sync-configs
description: Keep conf.default.yaml and conf.ZH.default.yaml structurally in sync when changing config_manager classes, adding engines, or touching config templates. Use whenever editing files under src/open_llm_vtuber/config_manager/ or config_templates/.
---

# Config template synchronization

There are two default config templates that must stay structurally
identical:

- `config_templates/conf.default.yaml` (English comments)
- `config_templates/conf.ZH.default.yaml` (Chinese comments)

## Rules

1. Any field added, renamed, or removed in a `config_manager` class must be
   reflected in **both** templates in the same location. Localize the
   comments in the ZH template; keys and values stay identical.
2. New engine options (ASR/TTS/LLM/VAD) go into both templates, and the
   corresponding config class gets the new block
   (see CLAUDE.md "Adding New Engines").
3. If the change would break existing user `conf.yaml` files, add an
   upgrade step under `upgrade_codes/` rather than silently changing
   semantics.

## Verify

```bash
uv run python tests/smoke_test.py
```

The test validates both templates against the pydantic models and diffs
their full key structure — any key present in only one template fails the
run and is listed by dotted path.
