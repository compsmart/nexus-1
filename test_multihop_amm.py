"""
test_multihop_amm.py

Stress-test Nexus AMM multi-hop retrieval with measurable metrics.

What this script measures:
  - answer_accuracy: keyword match on final answers
  - hop1_recall_at_k: whether expected hop-1 support fact appears in retrieve() #1
  - hop2_recall_at_k: whether expected hop-2 support fact appears in retrieve() #2
  - two_hop_success_rate: answer correct + hop1 hit + hop2 hit
  - latency p50/p95/max
  - tool_call_rate

It runs two phases over the same seeded memory:
  1) retrieval_hop2_enabled = False
  2) retrieval_hop2_enabled = True

Usage:
  python test_multihop_amm.py
  python test_multihop_amm.py --model microsoft/Phi-3.5-mini-instruct --use-4bit
  python test_multihop_amm.py --distractors 500 --top-k 7 --threshold 0.25
"""

import argparse
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

from agent import AGIAgent
from config import AgentConfig


TWO_HOP_CASES: List[Dict] = [
    {
        "case_id": "C01",
        "hop1_id": "MH01",
        "hop2_id": "MH02",
        "hop1_fact": "Project Atlas is led by Dana.",
        "hop2_fact": "Dana is based in Lisbon.",
        "question": "Which city is the lead of Project Atlas based in?",
        "expected_keywords": ["lisbon"],
    },
    {
        "case_id": "C02",
        "hop1_id": "MH03",
        "hop2_id": "MH04",
        "hop1_fact": "Project Orion is managed by Malik.",
        "hop2_fact": "Malik works from Nairobi.",
        "question": "Where does the manager of Project Orion work from?",
        "expected_keywords": ["nairobi"],
    },
    {
        "case_id": "C03",
        "hop1_id": "MH05",
        "hop2_id": "MH06",
        "hop1_fact": "Service Kappa is owned by Lina.",
        "hop2_fact": "Lina is located in Seoul.",
        "question": "What city is the owner of Service Kappa located in?",
        "expected_keywords": ["seoul"],
    },
    {
        "case_id": "C04",
        "hop1_id": "MH07",
        "hop2_id": "MH08",
        "hop1_fact": "Warehouse Delta is supervised by Omar.",
        "hop2_fact": "Omar is based in Dubai.",
        "question": "Which city is the supervisor of Warehouse Delta based in?",
        "expected_keywords": ["dubai"],
    },
    {
        "case_id": "C05",
        "hop1_id": "MH09",
        "hop2_id": "MH10",
        "hop1_fact": "Program Nimbus is coordinated by Hana.",
        "hop2_fact": "Hana works in Osaka.",
        "question": "Where does the coordinator of Program Nimbus work?",
        "expected_keywords": ["osaka"],
    },
    {
        "case_id": "C06",
        "hop1_id": "MH11",
        "hop2_id": "MH12",
        "hop1_fact": "Lab Echo is run by Victor.",
        "hop2_fact": "Victor is based in Boston.",
        "question": "What city is the person running Lab Echo based in?",
        "expected_keywords": ["boston"],
    },
    {
        "case_id": "C07",
        "hop1_id": "MH13",
        "hop2_id": "MH14",
        "hop1_fact": "Study Aurora is led by Elise.",
        "hop2_fact": "Elise works out of Zurich.",
        "question": "Where does the leader of Study Aurora work out of?",
        "expected_keywords": ["zurich"],
    },
    {
        "case_id": "C08",
        "hop1_id": "MH15",
        "hop2_id": "MH16",
        "hop1_fact": "Account Borealis is handled by Jorge.",
        "hop2_fact": "Jorge is located in Mexico City.",
        "question": "Which city is the person handling Account Borealis located in?",
        "expected_keywords": ["mexico", "city"],
    },
]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def score_keywords(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def build_seed_facts(distractors: int, seed: int) -> List[Tuple[str, Dict]]:
    rng = random.Random(seed)
    seeded: List[Tuple[str, Dict]] = []
    now = time.time()

    for case in TWO_HOP_CASES:
        seeded.append(
            (f"[{case['hop1_id']}] {case['hop1_fact']}",
             {"type": "fact", "subject": "multi_hop", "timestamp": now})
        )
        seeded.append(
            (f"[{case['hop2_id']}] {case['hop2_fact']}",
             {"type": "fact", "subject": "multi_hop", "timestamp": now})
        )

    people = [
        "Asha", "Noah", "Marta", "Kenji", "Ravi", "Leah", "Iris", "Pavel",
        "Yuna", "Diego", "Sara", "Tariq", "Anya", "Leo", "Farah", "Ivan",
    ]
    cities = [
        "Madrid", "Berlin", "Prague", "Helsinki", "Austin", "Denver",
        "Lima", "Doha", "Jakarta", "Riyadh", "Athens", "Santiago",
    ]
    projects = [
        "Mercury", "Quasar", "Apex", "Beacon", "Cascade", "Zenith",
        "Pulse", "Vector", "Fusion", "Vertex", "Summit", "Harbor",
    ]
    roles = [
        "led by", "managed by", "supervised by", "coordinated by",
        "owned by", "handled by",
    ]

    for idx in range(distractors):
        p = rng.choice(people)
        c = rng.choice(cities)
        pr = rng.choice(projects)
        role = rng.choice(roles)
        text = f"[DX{idx:04d}] Project {pr} is {role} {p}, and {p} recently visited {c}."
        seeded.append(
            (text, {"type": "fact", "subject": "distractor", "timestamp": now})
        )

    rng.shuffle(seeded)
    return seeded


def reset_agent_memory(agent: AGIAgent, mem_path: Path, seeded_facts: List[Tuple[str, Dict]]) -> None:
    agent.memory.save_path = str(mem_path)
    agent.memory._keys_path = str(mem_path) + ".pt"
    with agent.memory._lock:
        agent.memory._keys = deque(maxlen=agent.memory.max_slots)
        agent.memory._values = deque(maxlen=agent.memory.max_slots)
        agent.memory._metadata = deque(maxlen=agent.memory.max_slots)
        agent.memory._dirty = False
    agent._history.clear()
    agent.user_name = None

    for text, meta in seeded_facts:
        agent.memory.add_memory(text, meta)
    agent.memory.flush()


def run_phase(agent: AGIAgent, hop2_enabled: bool) -> Dict:
    agent.config.retrieval_hop2_enabled = hop2_enabled
    phase_rows = []
    current_retrieves = []
    current_tools = []

    original_retrieve = agent.memory.retrieve
    original_execute = agent._execute_tool
    original_parse_tool_call = agent._parse_tool_call

    def wrapped_retrieve(*args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        query = args[0] if args else kwargs.get("query", "")
        current_retrieves.append({
            "query": query,
            "result_texts": [r[0] for r in result],
        })
        return result

    def wrapped_execute(tool_name: str, arg: str):
        current_tools.append((tool_name, arg))
        return original_execute(tool_name, arg)

    agent.memory.retrieve = wrapped_retrieve
    agent._execute_tool = wrapped_execute
    # Isolate retrieval-hops only: ignore LLM tool-call strings for this benchmark.
    agent._parse_tool_call = lambda _text: None

    try:
        for case in TWO_HOP_CASES:
            current_retrieves.clear()
            current_tools.clear()

            started = time.perf_counter()
            response = agent.interact(case["question"])
            latency_s = time.perf_counter() - started

            hop1_results = current_retrieves[0]["result_texts"] if len(current_retrieves) >= 1 else []
            hop2_result_pool = []
            if len(current_retrieves) >= 2:
                for rr in current_retrieves[1:]:
                    hop2_result_pool.extend(rr["result_texts"])

            hop1_tag = f"[{case['hop1_id']}]"
            hop2_tag = f"[{case['hop2_id']}]"
            hop1_hit = any(hop1_tag in t for t in hop1_results)
            hop2_called = len(current_retrieves) >= 2
            hop2_hit = hop2_called and any(hop2_tag in t for t in hop2_result_pool)
            answer_ok = score_keywords(response, case["expected_keywords"])

            phase_rows.append({
                "case_id": case["case_id"],
                "question": case["question"],
                "answer": response,
                "answer_ok": answer_ok,
                "latency_s": latency_s,
                "hop_count": len(current_retrieves),
                "hop1_hit": hop1_hit,
                "hop2_called": hop2_called,
                "hop2_hit": hop2_hit,
                "tool_calls": list(current_tools),
                "retrieve_queries": [r["query"] for r in current_retrieves],
            })
    finally:
        agent.memory.retrieve = original_retrieve
        agent._execute_tool = original_execute
        agent._parse_tool_call = original_parse_tool_call

    latencies = [r["latency_s"] for r in phase_rows]
    total_tools = sum(len(r["tool_calls"]) for r in phase_rows)
    turns_with_tool = sum(1 for r in phase_rows if r["tool_calls"])

    summary = {
        "rows": phase_rows,
        "answer_accuracy": sum(1 for r in phase_rows if r["answer_ok"]) / len(phase_rows),
        "hop1_recall_at_k": sum(1 for r in phase_rows if r["hop1_hit"]) / len(phase_rows),
        "hop2_trigger_rate": sum(1 for r in phase_rows if r["hop2_called"]) / len(phase_rows),
        "hop2_recall_at_k": sum(1 for r in phase_rows if r["hop2_hit"]) / len(phase_rows),
        "two_hop_success_rate": (
            sum(1 for r in phase_rows if r["answer_ok"] and r["hop1_hit"] and r["hop2_hit"])
            / len(phase_rows)
        ),
        "avg_latency_s": statistics.mean(latencies),
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "max_latency_s": max(latencies),
        "tool_call_rate": turns_with_tool / len(phase_rows),
        "avg_tool_calls_per_turn": total_tools / len(phase_rows),
    }
    return summary


def print_phase(name: str, phase: Dict) -> None:
    print("=" * 72)
    print(f"{name}")
    print("=" * 72)
    for row in phase["rows"]:
        print(
            f"{row['case_id']} "
            f"ok={'Y' if row['answer_ok'] else 'N'} "
            f"hops={row['hop_count']} "
            f"hop1={'Y' if row['hop1_hit'] else 'N'} "
            f"hop2_called={'Y' if row['hop2_called'] else 'N'} "
            f"hop2={'Y' if row['hop2_hit'] else 'N'} "
            f"lat={row['latency_s']:.2f}s "
            f"tools={len(row['tool_calls'])}"
        )
    print("-" * 72)
    print(f"answer_accuracy      : {phase['answer_accuracy']:.3f}")
    print(f"hop1_recall_at_k     : {phase['hop1_recall_at_k']:.3f}")
    print(f"hop2_trigger_rate    : {phase['hop2_trigger_rate']:.3f}")
    print(f"hop2_recall_at_k     : {phase['hop2_recall_at_k']:.3f}")
    print(f"two_hop_success_rate : {phase['two_hop_success_rate']:.3f}")
    print(f"avg_latency_s        : {phase['avg_latency_s']:.2f}")
    print(f"p50_latency_s        : {phase['p50_latency_s']:.2f}")
    print(f"p95_latency_s        : {phase['p95_latency_s']:.2f}")
    print(f"max_latency_s        : {phase['max_latency_s']:.2f}")
    print(f"tool_call_rate       : {phase['tool_call_rate']:.3f}")
    print(f"avg_tool_calls/turn  : {phase['avg_tool_calls_per_turn']:.3f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AMM two-hop retrieval quality and performance.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use-4bit", action="store_true", help="Enable 4-bit quantization for the chat model.")
    parser.add_argument("--distractors", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--memory-file", default="multihop_bench_memory.json")
    args = parser.parse_args()

    mem_path = Path(args.memory_file)
    for f in [mem_path, Path(str(mem_path) + ".pt")]:
        if f.exists():
            f.unlink()

    cfg = AgentConfig()
    cfg.name = "MultiHopBench"
    cfg.model_name = args.model
    cfg.use_4bit = args.use_4bit
    cfg.autonomous_learning = False
    cfg.retrieval_hop2_enabled = True
    cfg.retrieval_top_k = args.top_k
    cfg.retrieval_threshold = args.threshold
    cfg.retrieval_confidence_threshold = 0.0
    cfg.max_new_tokens_response = args.max_new_tokens
    cfg.max_new_tokens_extraction = 40
    cfg.think_interval_secs = 9999.0
    cfg.idle_threshold_secs = 9999.0
    # keep deterministic routing behavior for this retrieval benchmark
    cfg.tool_routing_backend = "pattern"

    print("Preparing benchmark...")
    print(f"model={cfg.model_name}")
    print(f"use_4bit={cfg.use_4bit}")
    print(f"distractors={args.distractors}")
    print(f"retrieval_top_k={cfg.retrieval_top_k}")
    print(f"retrieval_threshold={cfg.retrieval_threshold}")
    print()

    seeded_facts = build_seed_facts(args.distractors, args.seed)
    print(f"seeded_fact_count={len(seeded_facts)}")

    agent = AGIAgent(config=cfg)
    try:
        reset_agent_memory(agent, mem_path, seeded_facts)
        phase_hop1 = run_phase(agent, hop2_enabled=False)
        print_phase("PHASE A: hop2 disabled", phase_hop1)

        reset_agent_memory(agent, mem_path, seeded_facts)
        phase_hop2 = run_phase(agent, hop2_enabled=True)
        print_phase("PHASE B: hop2 enabled", phase_hop2)

        print("=" * 72)
        print("DELTA (hop2 enabled - hop2 disabled)")
        print("=" * 72)
        print(f"answer_accuracy_delta      : {phase_hop2['answer_accuracy'] - phase_hop1['answer_accuracy']:+.3f}")
        print(f"hop1_recall_at_k_delta     : {phase_hop2['hop1_recall_at_k'] - phase_hop1['hop1_recall_at_k']:+.3f}")
        print(f"hop2_recall_at_k_delta     : {phase_hop2['hop2_recall_at_k'] - phase_hop1['hop2_recall_at_k']:+.3f}")
        print(f"two_hop_success_rate_delta : {phase_hop2['two_hop_success_rate'] - phase_hop1['two_hop_success_rate']:+.3f}")
        print(f"avg_latency_s_delta        : {phase_hop2['avg_latency_s'] - phase_hop1['avg_latency_s']:+.2f}")
        print(f"p95_latency_s_delta        : {phase_hop2['p95_latency_s'] - phase_hop1['p95_latency_s']:+.2f}")
        print()
    finally:
        agent.stop()
        for f in [mem_path, Path(str(mem_path) + ".pt")]:
            if f.exists():
                f.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
