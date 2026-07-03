# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-LLM-VTuber is a voice-interactive AI companion with Live2D avatar support that runs completely offline. It's a cross-platform Python application supporting real-time voice conversations, visual perception, and Live2D character animations. The project features modular architecture for LLM, ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and other components.

This repository is a fork of Open-LLM-VTuber. Branches such as `v2`, `v0.3.0`, `v1-release`, and `stream_tts` are inherited upstream history — do not base work on them. All work branches from and merges to `main`; before starting, confirm your branch's merge-base is current with `origin/main` (`git fetch origin main && git merge-base HEAD origin/main`).

## Essential Commands

### Development Setup
- **Install dependencies**: `uv sync` (uses uv package manager)
- **Run server**: `uv run run_server.py`
- **Run with verbose logging**: `uv run run_server.py --verbose`
- **Update project**: `uv run upgrade.py`

### Code Quality
- **Lint code**: `ruff check .`
- **Format code**: `ruff format .`
- **Run pre-commit hooks**: `pre-commit run --all-files`
- **Config smoke test**: `uv run python tests/smoke_test.py` (offline; validates config templates, character configs, and EN/ZH template sync)
- **Dependency check**: `uv lock --check` (run whenever `pyproject.toml` changes)

### Fresh Checkout Gotchas
- **`frontend/` is an empty git submodule** in fresh clones. Run `git submodule update --init frontend` or the web UI returns 404 (backend endpoints still work).
- **`conf.yaml` does not exist and is gitignored.** The server auto-creates it from `config_templates/conf.default.yaml` on first boot — don't hand-create it unless you need overrides.
- **`chat_history/` and `cache/` are created at runtime**; their absence is normal.
- **First boot needs network**: `server.initialize()` downloads ASR/TTS models, and the default LLM config expects Ollama at `localhost:11434` (the server boots without it, but conversations fail).
- When you cannot boot the server (offline container), verify changes with `uv run python tests/smoke_test.py` and say explicitly that the change was not verified against a running server.

### Server Configuration
- **Main config file**: `conf.yaml` (user configuration)
- **Default configs**: `config_templates/conf.default.yaml` and `config_templates/conf.ZH.default.yaml`
- **Character configs**: `characters/` directory (YAML files)

## Architecture Overview

### Core Components

**WebSocket Server** (`src/open_llm_vtuber/server.py`):
- FastAPI-based server handling WebSocket connections
- Serves frontend, Live2D models, and static assets
- Supports both main client and proxy WebSocket endpoints

**Service Context** (`src/open_llm_vtuber/service_context.py`):
- Central dependency injection container
- Manages all engines (LLM, ASR, TTS, VAD, etc.)
- Each WebSocket connection gets its own service context instance

**WebSocket Handler** (`src/open_llm_vtuber/websocket_handler.py`):
- Routes WebSocket messages to appropriate handlers
- Manages client connections, groups, and conversation state
- Handles audio data, conversation triggers, and Live2D interactions

### Modular Engine System

The project uses a factory pattern for all AI engines:

**Agent System** (`src/open_llm_vtuber/agent/`):
- `agent_factory.py` - Factory for creating different agent types
- `agents/` - Various agent implementations (basic_memory, hume_ai, letta, mem0)
- `stateless_llm/` - Stateless LLM implementations (Claude, OpenAI, Ollama, etc.)

**ASR Engines** (`src/open_llm_vtuber/asr/`):
- Support for multiple ASR backends: Sherpa-ONNX, FunASR, Faster-Whisper, OpenAI Whisper, etc.
- Factory pattern for engine selection based on configuration

**TTS Engines** (`src/open_llm_vtuber/tts/`):
- Multiple TTS options: Azure TTS, Edge TTS, MeloTTS, CosyVoice, GPT-SoVITS, etc.
- Configurable voice cloning and multi-language support

**VAD (Voice Activity Detection)** (`src/open_llm_vtuber/vad/`):
- Silero VAD for detecting speech activity
- Essential for voice interruption without feedback loops

### Configuration Management

**Config System** (`src/open_llm_vtuber/config_manager/`):
- Type-safe configuration classes for each component
- Automatic validation and loading from YAML files
- Support for multiple character configurations and config switching

### Conversation System

**Conversation Handling** (`src/open_llm_vtuber/conversations/`):
- `conversation_handler.py` - Main conversation orchestration
- `single_conversation.py` - Individual user conversations
- `group_conversation.py` - Multi-user group conversations
- `tts_manager.py` - Audio streaming and TTS management

### MCP (Model Context Protocol) Integration

**MCP System** (`src/open_llm_vtuber/mcpp/`):
- Tool execution and server registry
- JSON detection and parameter extraction
- Integration with various MCP servers for extended functionality

## Key Development Patterns

### Error Handling
When a WebSocket connection fails during setup, it is cleaned up via `_cleanup_failed_connection` in `websocket_handler.py`. New connection-handling code must implement equivalent cleanup so failed connections don't leak service contexts.

### Live2D Integration
- Models stored in `live2d-models/` directory
- Each model has its own `.model3.json` configuration
- Expression and motion control through WebSocket messages

### Audio Processing
- Real-time audio streaming through WebSocket
- Voice interruption support without headphones
- Multi-format audio support with proper codec handling

### Multi-language Support
- Character configurations support multiple languages
- TTS translation capabilities (speak in different language than input)
- I18n system for UI elements

## Important File Locations

- **Entry point**: `run_server.py`
- **Main server**: `src/open_llm_vtuber/server.py`
- **WebSocket routing**: `src/open_llm_vtuber/routes.py`
- **Configuration**: `conf.yaml` (user; gitignored, auto-created on first boot), `config_templates/` (defaults)
- **Frontend**: `frontend/` (Git submodule — empty until `git submodule update --init frontend`)
- **Live2D models**: `live2d-models/`
- **Character definitions**: `characters/`
- **Chat history**: `chat_history/` (created at runtime)
- **Cache**: `cache/` (audio files, temporary data; created at runtime)

## Development Guidelines

### Adding New Engines
1. Create interface in appropriate directory (e.g., `asr_interface.py`)
2. Implement concrete class following existing patterns
3. Add to factory class (e.g., `asr_factory.py`)
4. Update configuration classes in `config_manager/`
5. Add configuration options to default YAML files

### WebSocket Message Handling
1. Add message type to `MessageType` enum in `websocket_handler.py`
2. Create handler method following `_handle_*` pattern
3. Register in `_init_message_handlers()` dictionary
4. Ensure proper error handling and client response

### Configuration Changes
- Always update both default config templates
- Maintain backward compatibility when possible
- Use the upgrade system for breaking changes
- Validate configurations in respective config manager classes

## Testing and Quality Assurance

**There is no unit test suite.** Verification for this project means:
- **Ruff** for linting and formatting (configured in `pyproject.toml`; CI runs it on every push/PR)
- **Config smoke test**: `uv run python tests/smoke_test.py` — validates all config templates and character configs through the server's own load path, and checks EN/ZH template sync (CI runs it via `.github/workflows/smoke.yml`)
- **`uv lock --check`** on any dependency change (CI runs it; an out-of-sync or unresolvable lockfile breaks `uv sync` for every user)
- **Pre-commit hooks** for automated quality checks
- Manual testing through the web interface and desktop client

Never merge a PR while its checks are still running, and never claim a change works if it was only linted.

## Package Management

Uses **uv** (modern Python package manager):
- Dependencies defined in `pyproject.toml`
- Lock file: `uv.lock`
- Generated requirements: `requirements.txt` (auto-generated)
- Optional dependencies for specific features (e.g., `bilibili` extra)