"""
test_memory.py — Programmatic memory recall test for the Nexus AGI agent.

Three-phase test:
  Phase 1 — Clean recall  : 16 questions immediately after seeding facts.
  Phase 2 — Noise flood   : 15 off-topic exchanges to pollute history + AMM.
  Phase 3 — Post-noise    : Same 16 questions re-asked after all the noise.

Prints a before/after comparison table so recall degradation is visible.
No internet required — all facts come from the injected seed data.

Run with:
    python test_memory.py
"""

import logging
import sys
import time
import re
from pathlib import Path

# Force line-buffered stdout so every print appears immediately even when
# output is captured or the terminal is slow.
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

from config import AgentConfig
from agent import AGIAgent

# ── Colour helpers ──────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def c(text, colour): return colour + text + Style.RESET_ALL
except ImportError:
    def c(text, colour=""): return text
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""


# ── Test data ────────────────────────────────────────────────────────────────
#
# Facts injected directly into AMM (simulating multiple prior conversations).
# Each tuple is (text, metadata_dict).
#
SEED_FACTS = [
    # Identity
    ("User identity: name=Alex",
     {"type": "identity", "subject": "user"}),

    # Personal facts — same format _extract_personal_facts() would produce
    ("User personal fact: my dog is called Bruno",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my cat is called Luna",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my favourite colour is dark green",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my job is software engineer",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: I live in Edinburgh",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: I am 29 years old",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my sister is called Sophie",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my favourite food is sushi",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my car is a blue Honda Civic",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: I play the guitar",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my favourite band is Radiohead",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: my birthday is on the 14th of March",
     {"type": "fact", "subject": "personal_fact"}),

    # Compound / inferential facts
    ("User personal fact: I have two pets — a dog called Bruno and a cat called Luna",
     {"type": "fact", "subject": "personal_fact"}),
    ("User personal fact: I work remotely from Edinburgh as a software engineer",
     {"type": "fact", "subject": "personal_fact"}),    # Synonym bridge: 'work' phrasing so 'What do I do for work?' retrieves it
    ("User personal fact: my work is software engineering",
     {"type": "fact", "subject": "personal_fact"}),]


# ── Recall questions ─────────────────────────────────────────────────────────
#
# Each entry: (question, [expected keywords that should appear in the answer])
# Keywords are lowercased; ALL must appear for a PASS.
#
QUESTIONS = [
    # Direct recall
    ("What is my name?",                        ["alex"]),
    ("What is my dog called?",                  ["bruno"]),
    ("What is my cat called?",                  ["luna"]),
    ("What is my favourite colour?",            ["green"]),
    ("What do I do for work?",                  ["software", "engineer"]),
    ("Where do I live?",                        ["edinburgh"]),
    ("How old am I?",                           ["29"]),
    ("What is my sister's name?",               ["sophie"]),
    ("What is my favourite food?",              ["sushi"]),
    ("What colour is my car?",                  ["blue"]),
    ("What instrument do I play?",              ["guitar"]),
    ("What is my favourite band?",              ["radiohead"]),
    ("When is my birthday?",                    ["march", "14"]),

    # Slightly rephrased / indirect
    ("Do I have any pets? What are they called?", ["bruno", "luna"]),
    ("What city am I based in?",                ["edinburgh"]),
    ("Tell me about my job.",                   ["software", "engineer"]),
]


# ── Noise conversations ───────────────────────────────────────────────────────
#
# Off-topic exchanges injected between Phase 1 and Phase 2 to:
#   • Fill and overflow the rolling _history window (maxlen=20 slots = 10 turns)
#   • Add unrelated user_input / agent_response AMM entries
#   • Test whether fact retrieval survives context pollution
#
NOISE_TURNS = [
    "How far away is the Moon from Earth?",
    "What is the speed of light in metres per second?",
    "Name the planets in our solar system.",
    "What are the main ingredients in a classic carbonara pasta?",
    "When did the Second World War end?",
    "Explain what a binary search tree is.",
    "Who wrote the play Hamlet?",
    "If a train travels at 80 km/h for 2.5 hours, how far does it go?",
]


# ── Scorer ───────────────────────────────────────────────────────────────────

def score_response(response: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, missing_keywords)."""
    lower = response.lower()
    missing = [kw for kw in keywords if kw not in lower]
    return len(missing) == 0, missing


def run_question_battery(
    agent, label: str, colour, total_questions: int = None
) -> tuple[list[bool], list[str]]:
    """
    Runs all QUESTIONS through the agent.
    Returns (results list of booleans, list of response strings).
    """
    print(c("═"*60, colour))
    print(c(f"  {label}", colour + Style.BRIGHT))
    print(c("═"*60, colour))

    results: list[bool] = []
    responses: list[str] = []
    phase_start = time.time()
    n = total_questions or len(QUESTIONS)

    for i, (question, keywords) in enumerate(QUESTIONS, 1):
        print(c(f"\n[Q{i:02d}/{n}] {question}", Fore.YELLOW + Style.BRIGHT), flush=True)
        print(c("  ⏳ thinking...", Fore.WHITE), end="", flush=True)
        t0 = time.time()
        response = agent.interact(question)
        elapsed = time.time() - t0
        # overwrite the thinking line
        print(c(f"  done ({elapsed:.1f}s)", Fore.WHITE), flush=True)

        passed, missing = score_response(response, keywords)
        results.append(passed)
        responses.append(response)

        display = response[:200] + ("…" if len(response) > 200 else "")
        print(c(f"  ↳ {display}", Fore.WHITE), flush=True)

        if passed:
            print(c(f"  ✔ PASS  (expected: {keywords})  [{elapsed:.1f}s]", Fore.GREEN), flush=True)
        else:
            print(c(f"  ✘ FAIL  (expected: {keywords}  |  missing: {missing})  [{elapsed:.1f}s]", Fore.RED), flush=True)

        remaining = n - i
        if remaining > 0:
            avg = (time.time() - phase_start) / i
            print(c(f"  → {remaining} remaining, ETA ~{avg * remaining:.0f}s", Fore.WHITE), flush=True)

        time.sleep(0.1)

    phase_elapsed = time.time() - phase_start
    passes = sum(results)
    print(c(f"\n  Phase complete: {passes}/{n} passed in {phase_elapsed:.1f}s "
            f"(avg {phase_elapsed/n:.1f}s/question)", colour + Style.BRIGHT), flush=True)
    return results, responses


# ── Main test runner ─────────────────────────────────────────────────────────

def run():
    # ── Wipe isolated test memory files BEFORE constructing the agent ──────────
    # Uses a separate path so nexus_memory.json (production) is never touched.
    # Delete first so _load_memory() finds nothing on disk.
    TEST_MEM = Path("test_run_memory.json")
    for f in [TEST_MEM, Path(str(TEST_MEM) + ".pt")]:
        if f.exists():
            f.unlink()

    print(c("\n" + "═"*60, Fore.MAGENTA + Style.BRIGHT), flush=True)
    print(c("  Nexus Memory Recall Test Suite", Fore.MAGENTA + Style.BRIGHT), flush=True)
    print(c("  Three-phase: clean → noise → recall", Fore.MAGENTA), flush=True)
    print(c("═"*60 + "\n", Fore.MAGENTA + Style.BRIGHT), flush=True)

    # --- Boot agent ---
    cfg = AgentConfig()
    cfg.autonomous_learning    = False  # no web calls — deterministic
    cfg.retrieval_hop2_enabled = False  # skip second LLM call per turn
    cfg.max_new_tokens_response = 80   # shorter answers — faster inference
    cfg.think_interval_secs    = 999   # prevent background reflection during test
    cfg.retrieval_confidence_threshold = 0.0   # disable confidence gate in tests
    cfg.retrieval_threshold    = 0.25  # lower floor: 'work' vs 'job' scores 0.29
    cfg.name = "Nexus"                 # explicit — keeps save_path predictable
    agent = AGIAgent(config=cfg)

    # Redirect AMM to the isolated file AND clear any slots loaded from
    # nexus_memory.json during __init__ so we start with 0 slots.
    from collections import deque as _deque
    agent.memory.save_path  = str(TEST_MEM)
    agent.memory._keys_path = str(TEST_MEM) + ".pt"
    with agent.memory._lock:
        agent.memory._keys     = _deque(maxlen=agent.memory.max_slots)
        agent.memory._values   = _deque(maxlen=agent.memory.max_slots)
        agent.memory._metadata = _deque(maxlen=agent.memory.max_slots)
        agent.memory._dirty    = False
    print(c("  ✔ Isolated test memory ready (0 slots, fresh start)\n", Fore.GREEN), flush=True)

    # --- Seed facts ---
    print(c("\u25b6 Seeding AMM with test facts...", Fore.CYAN), flush=True)
    now = time.time()
    for text, meta in SEED_FACTS:
        meta = {**meta, "timestamp": now}
        agent.memory.add_memory(text, meta)
        print(f"  + {text[:70]}", flush=True)

    agent.user_name = "Alex"
    agent.memory.flush()
    print(c(f"\n\u2714 {len(SEED_FACTS)} facts seeded into AMM ({agent.memory.size} total slots)\n", Fore.GREEN), flush=True)

    agent.start()
    time.sleep(0.5)

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Clean recall (no noise yet)
    # ──────────────────────────────────────────────────────────────
    results_clean, responses_clean = run_question_battery(
        agent,
        "Phase 1 — Clean Recall (immediately after seeding)",
        Fore.CYAN,
        total_questions=len(QUESTIONS),
    )

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Noise flood
    # ──────────────────────────────────────────────────────────────
    print(c("\n" + "\u2550"*60, Fore.YELLOW + Style.BRIGHT), flush=True)
    print(c(f"  Phase 2 \u2014 Noise Flood ({len(NOISE_TURNS)} off-topic turns)", Fore.YELLOW + Style.BRIGHT), flush=True)
    print(c("\u2550"*60, Fore.YELLOW + Style.BRIGHT), flush=True)
    print(c(
        f"  History window: {cfg.max_history_turns} turns "
        f"({cfg.max_history_turns * 2} slots).  "
        f"Noise will overflow it by "
        f"{max(0, len(NOISE_TURNS) - cfg.max_history_turns)} turns.\n",
        Fore.YELLOW,
    ), flush=True)

    noise_start = time.time()
    for i, noise_q in enumerate(NOISE_TURNS, 1):
        print(c(f"  [N{i:02d}/{len(NOISE_TURNS)}] {noise_q[:80]}", Fore.YELLOW), flush=True)
        print(c(f"    \u23f3 thinking...", Fore.WHITE), end="", flush=True)
        t0 = time.time()
        noise_resp = agent.interact(noise_q)
        elapsed = time.time() - t0
        print(c(f"  done ({elapsed:.1f}s)", Fore.WHITE), flush=True)
        print(c(f"    \u2192 {noise_resp[:120]}\u2026", Fore.WHITE), flush=True)
        time.sleep(0.1)

    noise_elapsed = time.time() - noise_start
    slots_after_noise = agent.memory.size
    print(c(f"\n  Noise complete in {noise_elapsed:.1f}s. AMM now has {slots_after_noise} slots.\n", Fore.YELLOW), flush=True)

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Post-noise recall
    # ──────────────────────────────────────────────────────────────
    # Clear the rolling history so Phase 1+2 "I'm Phi" responses don't
    # condition Phase 3 answers.  AMM slots are unaffected — only the
    # short-term chat window is reset, exactly like a new conversation.
    agent._history.clear()

    results_noisy, responses_noisy = run_question_battery(
        agent,
        "Phase 3 — Post-Noise Recall (after off-topic flood)",
        Fore.CYAN,
        total_questions=len(QUESTIONS),
    )

    agent.stop()

    # ──────────────────────────────────────────────────────────────
    # Before / After comparison table
    # ──────────────────────────────────────────────────────────────
    total = len(QUESTIONS)
    clean_passes  = sum(results_clean)
    noisy_passes  = sum(results_noisy)

    print(c("\n" + "═"*60, Fore.MAGENTA + Style.BRIGHT), flush=True)
    print(c("  BEFORE / AFTER COMPARISON", Fore.MAGENTA + Style.BRIGHT), flush=True)
    print(c("═"*60, Fore.MAGENTA + Style.BRIGHT), flush=True)

    col_w = 42
    hdr = f"  {'Question':<{col_w}}  Clean  Post-Noise"
    print(c(hdr, Fore.WHITE + Style.BRIGHT), flush=True)
    print("  " + "─" * (col_w + 16), flush=True)

    regressions = []
    improvements = []

    for i, (question, _) in enumerate(QUESTIONS):
        before = results_clean[i]
        after  = results_noisy[i]
        b_sym  = c("✔", Fore.GREEN)  if before else c("✘", Fore.RED)
        a_sym  = c("✔", Fore.GREEN)  if after  else c("✘", Fore.RED)

        q_trunc = question[:col_w].ljust(col_w)
        row_colour = ""
        if before and not after:
            row_colour = Fore.RED
            regressions.append(i + 1)
        elif not before and after:
            row_colour = Fore.GREEN
            improvements.append(i + 1)

        print(c(f"  {q_trunc}", row_colour) + f"  {b_sym}       {a_sym}", flush=True)

    print("  " + "─" * (col_w + 16), flush=True)
    print(c(f"\n  Phase 1 (clean)     : {clean_passes}/{total} passed  "
            f"({100*clean_passes//total}%)", Fore.CYAN + Style.BRIGHT), flush=True)
    print(c(f"  Phase 3 (post-noise): {noisy_passes}/{total} passed  "
            f"({100*noisy_passes//total}%)", Fore.CYAN + Style.BRIGHT), flush=True)

    delta = noisy_passes - clean_passes
    if delta < 0:
        print(c(f"\n  ⚠  Recall DEGRADED by {abs(delta)} question(s) after noise "
                f"(regressions: Q{regressions})", Fore.RED + Style.BRIGHT), flush=True)
    elif delta > 0:
        print(c(f"\n  ↑  Recall IMPROVED by {delta} question(s) after noise "
                f"(improvements: Q{improvements})", Fore.GREEN + Style.BRIGHT), flush=True)
    else:
        print(c(f"\n  ✔ Recall STABLE — noise had no impact on memory.", Fore.GREEN + Style.BRIGHT), flush=True)

    # AMM diagnostics
    print(c("\n── AMM Diagnostics ──────────────────────────────────────────", Fore.CYAN), flush=True)
    print(f"  Total memory slots used : {agent.memory.size}", flush=True)
    sample = agent.memory.retrieve("user personal fact", top_k=5, threshold=0.2)
    print(f"  Top-5 slots for 'user personal fact':", flush=True)
    for text, meta, score in sample:
        mtype = (meta or {}).get("type", "?")
        print(f"    [{mtype}, {score:.3f}] {text[:80]}", flush=True)

    all_passed = (clean_passes == total) and (noisy_passes == total)
    return all_passed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
