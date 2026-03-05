# AMM Scaling & Hybrid Architecture Ideas

Research directions for improving the Adaptive Modular Memory system,
adding reasoning capability, and scaling the hybrid LLM+AMM architecture.

---

## 1. Memory Consolidation (highest-value, lowest competition)

The current `_reflect()` generates insights but **never removes the source memories**.
Memory grows monotonically. Human memory doesn't work this way -- sleep consolidation
compresses episodic memories into semantic memories and discards the originals.

### Proposed implementation

```
Every N interactions (or on idle):
  1. Cluster similar slots using the existing embeddings (agglomerative, threshold=0.7)
  2. For each cluster of 3+ slots:
     - Generate a single compressed summary via the LLM
     - Score the summary against each original's key -- if avg similarity > 0.6,
       replace the cluster with the summary
     - New slot gets type="consolidated", decay=None (permanent)
  3. Memory shrinks instead of growing
```

### Why it matters

- The single change that would make AMM scale
- At 10k max_slots with FIFO eviction, critical facts get silently dropped
- With consolidation, the system gets *denser* over time -- fewer slots but each one
  carries more information
- The original retrieval interface is unchanged
- Not yet published as a working system for agent-scale memory (CLS theory exists for
  neural nets but not in this hybrid retrieval-augmented architecture)

---

## 2. Retrieval-Weighted Encoder Tuning (the "make it learn" move)

The frozen encoder (`all-MiniLM-L6-v2`, 22M params) is the ceiling. This makes it
adaptive without the cost of retraining from scratch.

### Proposed implementation

```
After each turn:
  - If the user's NEXT turn is not a correction -> positive signal
  - If the user corrects -> negative signal

  Positive: pull the query embedding and the retrieved fact embeddings
            closer via a contrastive LoRA adapter on the encoder
  Negative: push them apart

  Train the adapter every K interactions (micro-batch of recent turns)
  Takes seconds on CPU with a 22M-param encoder
```

### Why it matters

- Over time the encoder **specialises for this user's semantic space**
- "My dog" and "Rex" would converge in embedding space for a user who always
  associates them
- Genuinely personal -- the encoder becomes a fingerprint of how a person thinks
- The closest thing to "making AMM generalise" -- bending the representation
  space based on usage patterns rather than adding more data
- The retrieval analogue of what happens inside LLM weights during training,
  at orders of magnitude less compute

---

## 3. Confidence-Gated Hybrid (the scaling architecture)

The current system hard-injects all retrieved memory into the prompt. The LLM has
no mechanism to say "I know this better than memory does." This fails both ways:

- Memory has a stale fact -> LLM regurgitates it despite knowing better from training
- Memory is empty -> LLM hallucinates instead of admitting it doesn't know

### Proposed implementation

```
                           +---------------+
       query --------------| Gate MLP      |--- alpha (0-1)
       top_retrieval_score | (tiny,        |
       retrieval_count ----| 3 layers)     |
       query_type_embed ---+               |
                                           |
       prompt = alpha * memory_context + (1-alpha) * "Answer from your own knowledge"
```

Train the gate on the correction signal -- when the user corrects, the gate should
have been lower (don't trust memory). When memory was right, the gate should have
been higher. This is a ~50k parameter MLP trained on interaction logs. Converges fast.

### Why it matters

- The fundamental missing piece for hybrid scaling
- Without it, scaling either side (more memory or bigger LLM) doesn't help -- more
  memory just means more noise injected, and a bigger LLM is ignored because the
  system prompt says "CRITICAL: You MUST use the Context from AMM"
- Makes the hybrid actually hybrid instead of memory-always-wins

---

## 4. Graph-Structured Memory (replaces flat deque)

The 2-hop mechanism is a workaround for a flat data structure. It works by
regex-extracting entity names and querying for them -- a poor man's graph traversal.

### Proposed implementation

```
Current:   deque of (key, value, metadata)
Proposed:  nodes = {id: (key, value, metadata)}
           edges = {(id_a, id_b): weight}

On write:
  - Add node
  - For each existing node with cosine_sim > 0.5, add edge weighted by similarity
  - Edges also created when two nodes are retrieved together and the turn succeeds

On retrieve:
  - Hop 1: cosine search as before -> seed nodes
  - Hop 2+: traverse edges from seed nodes, weighted by edge strength
  - No regex, no entity extraction, no bridge patterns
  - Works for ANY relationship type, not just "[entity] led by [person]"
```

### Why it matters

- Kills the entire `hop2_bridge_patterns`, `hop2_location_intent_patterns`, and
  `hop2_query_templates_by_intent` config complexity
- The graph IS the multi-hop mechanism
- Generalises to N-hop automatically -- just keep traversing until edge weights
  drop below threshold
- Works for relationship types the current regex set can't handle

---

## 5. Per-Fact LoRA Micro-Adapters (the "LLM as efficient as AMM" play)

AMM's killer feature is write latency: ~1ms to store a fact. LLM weight updates take
minutes-to-hours. This aims to get parametric learning down to AMM-speed.

### Proposed implementation

```
Per-fact LoRA micro-adapters:

  1. User says "my dog is Rex"
  2. Generate a tiny LoRA adapter (rank=1, ~10k params) trained on:
     Input:  "What is the user's dog's name?"
     Output: "Rex"
     (3-5 gradient steps, <100ms on GPU)
  3. Store the adapter weights in AMM alongside the text
  4. At inference: retrieve relevant adapters, merge them into the base model
     via LoRA composition before generating

  Result: parametric knowledge that is:
    - Written in ~100ms (AMM-speed)
    - Retrieved by relevance (AMM-mechanism)
    - Compositional (LoRA merging)
    - Generalises (it's in the weights)
```

### Why it matters

- Turns AMM from a text store into a **weight store**
- Each memory isn't a string the LLM reads -- it's a weight delta the LLM *becomes*
- The LLM literally changes shape for each query based on what memories are relevant
- Related work exists (LoRA composition, model merging) but per-fact micro-adapters
  with retrieval-based composition is unexplored

---

## Recommended Priority Order

| Priority | Idea | Effort | Risk | Impact |
|----------|------|--------|------|--------|
| 1 | Memory consolidation | Low | Low | High -- immediate scaling fix |
| 2 | Graph-structured memory | Medium | Low | High -- replaces brittle hop-2 |
| 3 | Confidence gate | Low | Low | Medium -- makes hybrid real |
| 4 | Encoder tuning | Medium | Medium | Medium -- start data collection early |
| 5 | LoRA micro-adapters | High | High | High -- needs GPU, save for when 1-4 stable |

Ideas 1-3 are implementable within the current codebase without changing the external
interface. The agent still looks the same to the user -- it just gets better.
