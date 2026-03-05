# Code Review — `poc/nexus-1`

## Overall Structure

Clean, well-separated modularity: config, memory, LLM, tools, agent orchestrator, and entry point are each their own concern. The threading model (daemon think-thread + lock-protected AMM) is correct. Good use of dataclasses for config.

---

## agent.py

**Positives**
- Retrieval pipeline is well-thought-out: short-followup detection, correction handling, personal-fact extraction, and multi-hop all compose cleanly.
- `_finalize_turn` is correctly separated — always writes inputs/responses to memory regardless of how the turn resolved.
- Tool-call loop (`for _call_idx in range(max_tool_calls_per_turn + 1)`) correctly limits runaway chaining.
- Autonomous learning fallback (uncertainty detection → auto `learn_skill`) is a nice emergent behaviour layer.

**Issues**

1. **`_retrieve_hop2_llm` ignores `retrieval_hop2_pattern_threshold_scale`** — the LLM hop uses a hard-coded `threshold=self.config.retrieval_threshold` with no scaling, inconsistent with the pattern hop which applies `threshold_scale`. Should use the same scaled threshold.

2. **`_extract_personal_facts` is blind to negation** — `"I don't have a dog named Rex"` would be stored as a positive fact. No negation guard exists.

3. **`_is_short_followup` hard-codes `len > 7` word count** — this is a magic number with no config knob. Long pronoun-heavy questions ("why did they decide that it was a bad idea?") would miss the follow-up branch.

4. **No deduplication on `add_memory`** — every call to `interact()` blindly appends to AMM even if the exact text already exists. Memory will accumulate duplicates of personal facts on repeated mentions. A simple hash-set guard in `AdaptiveModularMemory.add_memory` would fix this.

5. **`_system_prompt` is built once in `__init__` and baked as a string** — tool descriptions are frozen at construction time. If tools are added dynamically this breaks. Minor in current code but worth noting.

6. **`_reflect` samples uniformly** — reflection quality would improve if it weighted toward high-decay-risk memories (facts near expiry), giving them a chance to be re-encoded as insights before they fade.

7. **`routed` pre-fetch always fires even if memory already has the answer** — e.g., asking "what time is it?" always calls `datetime_now` via routing before even checking whether the LLM already knows. This is intentional for time, correct, but the `web_latest` route fires `web_search` unconditionally — even on something like "latest news I told you about". A confidence gate from AMM retrieval before routing would prevent spurious web calls.

---

## memory.py

**Positives**
- Thread-safe via a single `_lock` with snapshot-before-compute pattern (copy out, release lock, do GPU work, results don't need lock). Correct.
- Dot-sidecar `.pt` approach means embeddings are never re-encoded on restart — very practical.
- `_decay_multiplier` is clean: exponential decay via `0.5^(t / t_half)`, `None` half-life means identity memories never decay.

**Issues**

1. **`delete_matching` decodes similarities under no lock** — it copies to `keys_snap`/`vals_snap` under lock, releases the lock, then computes similarities, then re-acquires lock to rebuild deques. A concurrent `add_memory` between the snapshot and the rebuild will lose the newly-added entry. Low probability but a real race.

2. **`deque(maxlen=max_slots)` silently evicts FIFO** — when the deque is full, the oldest memory drops regardless of importance. There's no LFU/LRU eviction or importance-weighted eviction. High-value identity/skill facts are at risk if max_slots is hit.

3. **`flush()` builds `keys_tensor = torch.stack(keys_snap)` after releasing the lock** — if `add_memory` fires concurrently between releasing the lock and `torch.save`, the saved `.pt` will be out of sync with the saved `.json`. Should snapshot both atomically before releasing.

4. **`retrieve()` computes cosine similarity with the full key matrix every call** — this is O(n) over all slots. At 10,000 slots with 384-dim embeddings this is fine, but at larger scales a FAISS/HNSW index would be warranted. Not a bug, just a known-scalability note.

---

## llm.py

**Positives**
- Clean separation: `generate()` for raw prompts, `chat()` for chat-template messages.
- Prompt truncation guard (`available = max_context - max_new_tokens`) is correct and defensive.
- 4-bit NF4 quantization path is properly conditioned on CUDA availability.

**Issues**

1. **`model_max_length` fallback of 2048 is often wrong** — Phi-3.5-mini has a 128k context. The `getattr(tokenizer, "model_max_length", 2048)` default silently truncates aggressively if the tokenizer doesn't set that attribute. Should use `tokenizer.model_max_length or model.config.max_position_embeddings`.

2. **No `repetition_penalty`** — open-ended generation at `temperature=0.7` without repetition penalty often loops on longer outputs. A default of `repetition_penalty=1.1` is a common fix.

3. **`generate()` is never called from `agent.py`** — only `chat()` is used. The raw `generate()` method is dead code in the current agent.

---

## config.py

**Positives**
- Single source of truth for all tunables. The multi-hop patterns are configurable as lists of regex strings, not baked in.
- Decay half-lives are per-type and in seconds, making them human-readable.

**Issues**

1. **`blocked_user_names` is a `set` in a `dataclass` field default** — technically fine with `field(default_factory=...)` but it's declared as `set = field(...)` without type annotation on the field, which mypy/pyright will flag.

2. **`hop2_bridge_patterns` only capture `[A-Z][A-Za-z'\-]+` (capitalized names)** — this will miss entities stored in sentence-case like `"project atlas is led by dana"` (all lowercase). If the stored fact was lowercased at write time, hop-2 pattern extraction silently produces zero entities, and the hop collapses to LLM-only.

3. **No validation in `__post_init__`** — fields like `retrieval_hop2_strategy` accept arbitrary strings; invalid values are silently normalized inside `_retrieve_hop2`. A validator would surface misconfiguration early.

---

## main.py

Works. Thin entry-point, appropriate. The `speak()` call happens synchronously after every response which will block input on long TTS output — `pyttsx3` is synchronous. A background thread for TTS would improve responsiveness.

---

## 2-Hop AMM: Full Technical Walkthrough

### The Problem It Solves

A query like *"Which city is the lead of Project Atlas based in?"* requires two independent memory lookups that are not semantically close to each other in the same document. The answer ("Lisbon") only exists in the **second** fact; the bridge (who leads the project) only exists in the **first** fact. Single-hop retrieval returns the first fact but not the second, so the LLM cannot answer.

---

### Configuration

All 2-hop parameters live in `config.py`:

| Field | Default | Role |
|---|---|---|
| `retrieval_hop2_enabled` | `True` | Master switch |
| `retrieval_hop2_strategy` | `"hybrid"` | `"pattern"`, `"llm"`, or `"hybrid"` |
| `retrieval_hop2_pattern_max_entities` | `4` | Max extracted bridge entities per hop-1 pass |
| `retrieval_hop2_pattern_top_k_per_query` | `2` | Top-k per each templated query |
| `retrieval_hop2_pattern_threshold_scale` | `1.0` | Multiplier on base threshold for hop-2 queries |
| `hop2_bridge_patterns` | 4 regexes | Patterns that extract the bridging entity from hop-1 text |
| `hop2_location_intent_patterns` | 1 regex | Detects whether the question is location-oriented |
| `hop2_query_templates_by_intent` | `{location: [...], default: [...]}` | Templates for constructing hop-2 search queries |

---

### Execution Flow

```
interact(user_input)
    │
    ├─ Hop 1: memory.retrieve(user_input, top_k=5, threshold=0.3,
    │          exclude_types={reflection, user_input, agent_response})
    │          -> retrieved_1 = [(text, meta, score), ...]
    │
    └─ if retrieved_1 and hop2_enabled:
           _retrieve_hop2(user_input, retrieved_1)
               │
               ├─ [pattern branch] _retrieve_hop2_pattern(user_input, hop1)
               │       │
               │       ├─ _extract_hop2_entities(hop1)
               │       │     For each (text, meta, score) in hop1:
               │       │       For each pattern in hop2_bridge_patterns:
               │       │         regex.search(text) -> extract named group "entity"
               │       │         e.g. "Project Atlas is led by Dana." -> "Dana"
               │       │         Collect up to max_entities (4) unique entities
               │       │
               │       ├─ _hop2_intent(user_input)
               │       │     Check hop2_location_intent_patterns -> "location" or "default"
               │       │
               │       ├─ _build_pattern_hop2_queries(user_input, hop1)
               │       │     For each entity x each template for the intent:
               │       │       e.g. entity="Dana", intent="location" ->
               │       │         "Dana based in", "Dana located in",
               │       │         "Dana works from", "Dana works in",
               │       │         "Dana live in", "Dana location"
               │       │
               │       └─ For each query:
               │             memory.retrieve(q, top_k=2,
               │                threshold=base_threshold * threshold_scale,
               │                exclude_types={reflection, user_input, agent_response})
               │             Aggregate with score-max deduplication
               │             Return top_k results -> pattern_rows
               │
               ├─ [llm branch] _retrieve_hop2_llm(user_input, hop1)
               │       │
               │       ├─ Build prompt:
               │       │     "User message: '...'
               │       │      Retrieved memories:
               │       │        - [MH01] Project Atlas is led by Dana.
               │       │      Is there additional related information worth
               │       │      searching for? Output a short search query if
               │       │      yes, otherwise output 'NONE'."
               │       │
               │       ├─ llm.chat(hop2_messages, max_new_tokens=50, temp=0.1)
               │       │     -> e.g. "Dana location" or "NONE"
               │       │
               │       └─ if not NONE: memory.retrieve(query, top_k=2)
               │             -> llm_rows
               │
               └─ [hybrid merge]
                     score-max deduplication of pattern_rows + llm_rows
                     sort by score descending, return top_k
```

---

### Concrete Example: Case C01

**Seeded facts in AMM:**
```
[MH01] Project Atlas is led by Dana.          (type=fact)
[MH02] Dana is based in Lisbon.               (type=fact)
+ 500 distractor facts
```

**User query:** `"Which city is the lead of Project Atlas based in?"`

**Hop 1:**
- Cosine similarity between query embedding and all key embeddings
- `[MH01]` scores highly (~0.65) because "lead of Project Atlas" has strong semantic overlap with "Project Atlas is led by"
- `[MH02]` scores low (~0.15) because "Dana is based in Lisbon" has little direct overlap with the question — the agent doesn't know Dana is relevant yet

**Pattern Hop 2 — entity extraction:**
- Text: `"[MH01] Project Atlas is led by Dana."`
- Bridge pattern: `r"\b(?:is|was|are|were)\s+(?:led|managed|owned|...)\s+by\s+(?P<entity>[A-Z][A-Za-z'\-]+...)"`
- Match: `"is led by Dana"` → `entity = "Dana"`

**Pattern Hop 2 — intent detection:**
- User input contains "city" and "based in" → matches `hop2_location_intent_patterns` → intent = `"location"`

**Pattern Hop 2 — query construction:**
Templates for `"location"` intent × entity `"Dana"`:
```
"Dana based in"
"Dana located in"
"Dana works from"
"Dana works in"
"Dana live in"
"Dana location"
```

**Pattern Hop 2 — memory retrieval per query:**
- `"Dana based in"` → cosine search → `[MH02] Dana is based in Lisbon.` scores ~0.78 → retrieved
- Other queries may also retrieve it at varying scores
- Aggregation: take max score across queries per unique text → `[MH02]` at score 0.78

**LLM Hop 2:**
- Prompt shows `[MH01]` to the LLM at low temperature (0.1)
- LLM outputs something like `"Dana location"` or `"Where is Dana based"`
- `memory.retrieve("Dana location", top_k=2)` → also retrieves `[MH02]`

**Hybrid merge:**
- Both pattern and LLM branches retrieved `[MH02]`; score-max dedup keeps the higher score
- Final `retrieved_2 = [([MH02] Dana is based in Lisbon., meta, 0.78)]`

**Context assembly (`_format_memory_context`):**
```
[Known Facts]
  [MH01] Project Atlas is led by Dana.
  [MH02] Dana is based in Lisbon.
```

**LLM receives:**
```
System: <system_prompt with identity rules>
User: Context:
[Known Facts]
  [MH01] Project Atlas is led by Dana.
  [MH02] Dana is based in Lisbon.

Which city is the lead of Project Atlas based in?
```

**Answer:** "Lisbon" ✓

---

### Why Pattern + LLM Hybrid

The pattern approach is fast and deterministic but brittle — it fails if entity casing doesn't match the regex (`[A-Z]` anchors). The LLM approach is flexible but slow and consumes tokens. Hybrid gives you the pattern approach's speed as the primary path with LLM as fallback for anything the patterns miss. The merge uses score-max deduplication so both can agree on the same fact without double-counting.

---

### Known Limitations of the Current 2-Hop Implementation

1. **Only 1 level of bridge** — it's 2-hop, not N-hop. Queries requiring 3 intermediate facts (Atlas → Dana → Dana's team → team city) are not supported.
2. **Entity regex requires capitalization** — entities stored in lowercase (e.g., from extracted fact strings like `"user personal fact: my boss is sara"`) will silently produce zero entities in the pattern branch.
3. **LLM hop uses the same `llm` instance as the main response** — during hop-2, the LLM is called synchronously mid-`interact()` with no budget awareness. On a slow machine this can add significant latency per turn.
4. **`threshold_scale = 1.0` by default** — means hop-2 retrieval uses the same threshold as hop-1, but hop-2 queries are more specific (e.g., `"Dana based in"`), so a *lower* threshold would be appropriate here since the entity-prefixed query is more targeted. The scale config exists but its default doesn't exploit this.
