---
name: smoke-test
description: Run the offline config smoke test (validates config templates, character configs, and EN/ZH template sync) plus a dependency lock check. Use before committing any change to src/open_llm_vtuber/config_manager/, config_templates/, characters/, or pyproject.toml, and before opening or merging any PR.
---

# Config smoke test

This project has no unit test suite. This smoke test is the fastest real
verification available — it exercises the same code path the server uses at
startup (`read_yaml` → `validate_config`) and on character switching
(`deep_merge`), entirely offline.

## Run it

```bash
uv run python tests/smoke_test.py
```

If the full `uv sync` is impractical (fresh ephemeral container, heavy torch
deps), a minimal environment is enough:

```bash
pip install pydantic pyyaml loguru chardet
python tests/smoke_test.py
```

## When pyproject.toml changed, also run

```bash
uv lock --check
```

This catches unresolvable or lock-drifted dependencies before they merge.
(In June 2026, a dependency added to pyproject.toml without a lock update
made `uv sync` fail for every user — discovered only after merge. This
check is how that class of breakage gets caught pre-merge.)

## Interpreting failures

- A pydantic validation error means a config file and the config_manager
  models disagree. Fix the config or update the model — do not loosen the
  model to make the error go away.
- An EN/ZH sync failure lists the dotted key paths present in only one
  template. Add the missing keys to the other template (with localized
  comments in conf.ZH.default.yaml).

All checks must pass before a PR is merged. CI runs this same script in
`.github/workflows/smoke.yml`.
