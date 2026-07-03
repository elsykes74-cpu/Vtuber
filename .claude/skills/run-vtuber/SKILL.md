---
name: run-vtuber
description: Boot the Open-LLM-VTuber server locally to verify a change end-to-end (server startup, config loading, WebSocket routes). Use when asked to run the app, or to confirm a nontrivial backend change actually works rather than just passing lint.
---

# Running the server

## What a fresh checkout is missing (and what fixes itself)

- `frontend/` is a git submodule that arrives **empty**. Run
  `git submodule update --init frontend` or the web UI will return 404.
  Backend endpoints and `/client-ws` still work without it.
- `conf.yaml` is gitignored and absent — this is fine. On first boot the
  server auto-creates it from `config_templates/conf.default.yaml`
  (`sync_user_config`). Don't hand-create it unless you need overrides.
- `chat_history/` and `cache/` are created at runtime.

## Steps

1. `uv sync` — heavy (torch etc.); several minutes on first run in a fresh
   container. Needs network.
2. `git submodule update --init frontend`
3. `uv run run_server.py --verbose`
   - First boot downloads ASR/TTS models (Hugging Face / ModelScope).
     Offline, `server.initialize()` will fail and the process exits 1.
   - The default LLM is `ollama_llm` at `localhost:11434`. The server
     **boots** without Ollama, but conversations fail at runtime. To test
     conversation flow without Ollama, point
     `character_config.agent_config.llm_configs.openai_compatible_llm` in
     `conf.yaml` at any reachable OpenAI-compatible endpoint and set
     `agent_settings.basic_memory_agent.llm_provider` accordingly.

## Verifying it's up

- Success looks like uvicorn serving on `http://localhost:12393` with no
  exit during initialization.
- `curl -s -o /dev/null -w '%{http_code}' http://localhost:12393/` →
  `200` with the submodule initialized (`404` means frontend is missing,
  not that the server is broken).
- WebSocket endpoint for clients: `ws://localhost:12393/client-ws`.

## If a full boot is impossible (offline container)

Fall back to the offline smoke test (`/smoke-test` skill,
`tests/smoke_test.py`) and say plainly in your report that the change was
not verified against a running server.
