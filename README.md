# Nexus AGI Agent

A proof-of-concept AGI agent from the Unified AI Research Lab combining a frozen LLM backbone with an Adaptive Modular Memory (AMM) system. The LLM provides language understanding and generation; the AMM provides persistent, searchable, semantically-indexed long-term memory that survives context window overflows and process restarts.

---

## Architecture Overview

```text
+--------+     +--------------------------------------+     +----------+
| User   | --> | agent.py / AGIAgent (interact loop) | --> | Response |
+--------+     +--------------------------------------+     +----------+
                    | 1) extract personal facts
                    | 2) hop-1 AMM retrieval
                    | 3) hop-2 AMM retrieval (optional)
                    | 4) format memory context
                    | 5) LLM generation (+ tool loop)
                    | 6) finalize turn (history + AMM writes)
                    |
                    | retrieve/add
                    v
          +----------------------------------------+
          | memory.py / AdaptiveModularMemory      |
          +----------------------------------------+
                    | load/save
                    v
          +----------------------------------------+
          | nexus_memory.json + nexus_memory.json.pt |
          +----------------------------------------+

agent.py -> llm.py / LLMEngine:
  - hop-2 query generation
  - response generation
  - reflection generation

agent.py <-> tools.py / Tool Registry:
  - LLM emits [TOOL_CALL: ...]
  - agent executes tool and feeds result back to LLM

Background thread (in agent.py):
  - every 30s idle: generate reflection -> store in AMM
  - every 60s: flush AMM to disk
```

## Components

### `agent.py` — AGIAgent

The central orchestrator. Owns the interaction loop, multi-hop retrieval, personal fact extraction, background reflection, and tool dispatch.

**Per-turn flow (`interact`)**

| Step | What happens |
|------|-------------|
| 1 | Regex scan for personal statements ("my dog is Bruno") → immediately stored as `type='fact'` (14-day half-life) |
| 2 | **Hop-1 retrieval** — encode the user message, cosine search AMM, return up to `top_k=5` entries above threshold `0.3`. Excludes `user_input`, `agent_response`, `reflection` types to prevent self-echo |
| 3 | **Hop-2 retrieval** (optional) — if hop-1 found something, ask the LLM if a follow-up search query would add value; run a second AMM search if so |
| 4 | Format retrieved memories into grouped sections: `[Identity]`, `[Known Facts]`, `[Learned Skills]`, `[Recent Context]` |
| 5 | Build the prompt: system instructions + rolling history + AMM context + user message; generate response via LLM; loop up to 3 times if the LLM emits a `[TOOL_CALL: …]` |
| 6 | `_finalize_turn` — append both sides to the history deque, write `user_input` and `agent_response` entries into AMM |

**Short follow-up handling** — for messages under 7 words containing pronouns like "it / that / they", the retrieval query is prefixed with the last user+assistant pair so short references resolve correctly.

**Confidence gate** — if the highest cosine score among hop-1 results is below `retrieval_confidence_threshold` (default `0.25`), a soft warning is prepended to the context asking the LLM to prefer higher-confidence entries.

**Background thread** — runs on a `threading.Event` timer. When the agent has been idle for 30 s it samples 3 random memories, generates a short insight with the LLM, and stores it as `type='reflection'`. Flushes AMM to disk every 60 s.

---

### `memory.py` — AdaptiveModularMemory

An explicit, slot-based vector store that encodes text into 384-dimensional embeddings and retrieves by cosine similarity. Unlike a neural memory, it never overwrites existing slots — it appends and relies on the `deque(maxlen=…)` to evict the oldest entry when the cap is reached.

**Data layout** — three parallel deques of the same length:
- `_keys` — pre-computed `float32` embedding tensors (one per slot)
- `_values` — raw text strings
- `_metadata` — dicts carrying `type`, `subject`, `timestamp`

**Retrieval scoring**

$$\text{score}_i = \cos(\vec{q},\, \vec{k}_i) \times 0.5^{\,t_{\text{age}} \,/\, t_{1/2}}$$

The cosine threshold is applied to the **raw** cosine (before decay) so old but relevant memories still surface. Decay only influences ranking order among survivors.

**Memory types and half-lives**

| Type | Half-life | Purpose |
|------|-----------|---------|
| `identity` | ∞ | User name and permanent identity facts |
| `fact` | 14 days | Personal facts extracted from conversation |
| `skill_ref` | 30 days | Compact pointers to canonical markdown skills on disk |
| `skill` | 30 days | Legacy in-memory skill chunks (migration fallback) |
| `document` | 14 days | Ingested file or URL content |
| `user_input` | 3 days | Rolling conversation context |
| `agent_response` | 3 days | Rolling conversation context |
| `reflection` | 7 days (default) | Background-generated insights |

**Persistence** — `flush()` uses a dirty flag so disk writes only happen when the state has changed. Embeddings are stored in a `.pt` sidecar file so they are never re-encoded on restart.

**Thread safety** — a single `threading.Lock` guards all deque mutations. Encoding happens outside the lock to avoid blocking concurrent reads.

---

### `llm.py` — LLMEngine

Thin wrapper around a HuggingFace `AutoModelForCausalLM`. Loads the model once, exposes `chat()` (chat-template) and `generate()` (raw prompt). Context window overflows are handled by left-truncating the token sequence. Temperatures below `0.15` automatically switch to greedy decoding.

**Default model** — `microsoft/Phi-3.5-mini-instruct` loaded in 4-bit NF4 quantisation via `bitsandbytes`, consuming ~2.5 GB VRAM (fits a 4 GB card like an RTX 3050 Laptop).

---

### `tools.py` — Tool Registry

Tools the LLM can call by emitting `[TOOL_CALL: tool_name | argument]` in its response. The agent's tool-dispatch loop intercepts these, runs the tool, injects the result back as a user message, and asks the LLM to continue — up to `max_tool_calls_per_turn = 3` times per turn.

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo full-text search (no API key) |
| `web_fetch` | Download and clean a web page |
| `wikipedia` | Wikipedia summary for a topic |
| `calculator` | Safe arithmetic / math expression evaluation |
| `datetime_now` | Current date, time and timezone |
| `unit_convert` | Length, weight, temperature conversions |
| `timer_delta` | Days / hours between two dates |
| `python_exec` | Run a Python snippet in a subprocess sandbox |
| `file_read` | Read a text file from disk |
| `file_write` | Write / append to a file |
| `file_list` | List directory contents |
| `file_search` | Find files by name or content |
| `ingest_document` | Parse a file or URL into searchable AMM slots |
| `create_skill` | Create a draft markdown skill from structured input and index a pointer |
| `learn_skill` | Web-search a topic and create a draft markdown skill + AMM pointer |
| `list_skills` | List skill files from the on-disk catalog |
| `show_skill` | Show a structured excerpt of one skill |
| `publish_skill` | Promote a draft skill to published and refresh AMM pointer metadata |

---

### `config.py` — AgentConfig

Single dataclass holding every tunable parameter. Key values:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model_name` | `Phi-3.5-mini-instruct` | LLM backbone |
| `use_4bit` | `True` | 4-bit NF4 quantisation |
| `memory_encoder` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `max_memory_slots` | 10 000 | Hard cap on AMM entries (deque eviction) |
| `retrieval_threshold` | 0.3 | Minimum cosine to surface a memory |
| `retrieval_top_k` | 5 | Max results per retrieval hop |
| `retrieval_confidence_threshold` | 0.25 | Score below which the confidence gate fires |
| `retrieval_hop2_enabled` | `True` | Toggle second retrieval hop |
| `retrieval_hop2_strategy` | `hybrid` | Hop-2 query strategy: `pattern`, `llm`, or `hybrid` |
| `max_history_turns` | 10 | Rolling chat window (20 message slots) |
| `autonomous_learning` | `True` | Auto-trigger `learn_skill` on uncertain answers |
| `tool_routing_backend` | `pattern` | Intent-to-tool routing backend (`pattern` or `hf_zero_shot`) |
| `tool_routing_hf_model` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | HF zero-shot model used when backend is `hf_zero_shot` |

---

## Data Flow — Single Turn

```
User types: "What is my dog called?"
                │
                ▼
        _extract_personal_facts()   → no match
                │
                ▼
        AMM.retrieve("What is my dog called?", threshold=0.3)
        → [("User personal fact: my dog is called Bruno", {type:'fact'}, 0.61)]
                │
          [hop-2 enabled?]──No──▶  skip
                │
                ▼
        _format_memory_context()
        → "[Known Facts]\n  User personal fact: my dog is called Bruno\n"
                │
                ▼
        LLM.chat([
          system:  "You are Nexus …",
          history: […last 10 turns…],
          user:    "Context:\n[Known Facts]\n  …Bruno\n\nWhat is my dog called?"
        ])
        → "Bruno"
                │
                ▼
        _finalize_turn()
          history.append(user), history.append("Bruno")
          AMM.add("What is my dog called?", type='user_input')
          AMM.add("Assistant: Bruno",       type='agent_response')
                │
                ▼
        return "Bruno"
```

---

## Persistence

Two files written by `AMM.flush()`:

| File | Contents |
|------|----------|
| `nexus_memory.json` | JSON array of `{values, metadata}` |
| `nexus_memory.json.pt` | PyTorch tensor of shape `[n_slots, 384]` — pre-computed embeddings |

On restart, embeddings are loaded directly from the `.pt` file (no re-encoding). If only the JSON exists the embeddings are re-computed once as a one-time upgrade.

## Skill Catalog

Canonical skills are stored in markdown files under `skills/`:

- `skills/drafts/` for unverified generated skills
- `skills/published/` for reviewed/promoted skills
- `skills/archive/` for superseded published revisions
- `skills/index.json` as a rebuildable index

AMM stores only compact `skill_ref` pointer entries. At answer time, the agent can hydrate selected skill sections from disk into context.

Security policy:
- No inline secrets in markdown skill files
- Use `${ENV_VAR_NAME}` placeholders only
- Publishing is blocked when inline credential patterns are detected

Migration utility:

```bash
python scripts/migrate_legacy_skills.py --memory-file nexus_memory.json
```

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

**Hardware** — a CUDA GPU with ≥ 4 GB VRAM is recommended. CPU inference works but is much slower.

## Controls

| Input | Action |
|-------|--------|
| Any text | Chat with the agent |
| `/help` | Show command help |
| `/status` | Show voice mode + STT/TTS availability |
| `/voice` (or `voice`) | Toggle microphone input (requires `SpeechRecognition` + `PyAudio`) |
| `/quit` / `/exit` (or `quit` / `exit`) | Shut down gracefully and flush memory |

## Testing

```bash
python test_memory.py
```

Runs a three-phase noise-resilience test: seeds 16 personal facts into a fresh isolated AMM instance, asks 16 recall questions, floods 8 off-topic noise turns, then re-asks all 16 questions. Prints a before/after comparison table. Expected result: 16/16 in both phases.

```bash
python test_multihop_amm.py
```

Runs a multi-hop AMM benchmark with distractors and compares:
- `retrieval_hop2_enabled=False` vs `retrieval_hop2_enabled=True`
- answer accuracy
- hop-1 / hop-2 retrieval recall@k
- two-hop success rate
- latency (avg, p50, p95, max)
- tool call rate

## Benchmark Framework

A plugin-style benchmark framework is available under `benchmarks/`.

Run from the CLI:

```bash
python -m benchmarks.cli.bench run --name quality-run --use-4bit
python -m benchmarks.cli.bench list
python -m benchmarks.cli.bench compare <run_a> <run_b>
python -m benchmarks.cli.bench status <run_id>
python -m benchmarks.cli.bench list --json
```

`refs-sync` input sources are mutually exclusive: use either `--source-url` or `--input-json`.

Start the UI:

```bash
streamlit run benchmarks/ui/app.py
```

The framework supports:
- suite plugins (`benchmarks/suites/<suite_id>/suite.py`)
- baseline adapters (`benchmarks/baselines/*.py`)
- config-driven scoring (`benchmarks/config/scoring.yaml`)
- run history with soft delete / restore (`benchmarks/runs/index.sqlite`)
- per-run immutable artifacts in `benchmarks/runs/<run_id>/`
