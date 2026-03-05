import logging
import math
import re
import time
import threading
from collections import deque
from typing import Dict, Iterator, List, Optional, Set, Tuple

from config import AgentConfig
from memory import AdaptiveModularMemory
from llm import LLMEngine
from tools import build_tool_registry
from skills_store import SkillStore

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional runtime dependency path
    hf_pipeline = None


class AGIAgent:
    """
    AGI Agent combining LLM reasoning with Adaptive Modular Memory (AMM).
    Implements continuous learning and thinking loops.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.name = self.config.name
        self.user_name: Optional[str] = None

        self.memory = AdaptiveModularMemory(
            model_name=self.config.memory_encoder,
            max_slots=self.config.max_memory_slots,
            save_path=f"{self.name.lower()}_memory.json",
            decay_enabled=self.config.memory_decay_enabled,
            decay_half_lives=self.config.memory_decay_half_lives,
            dedup_enabled=self.config.memory_dedup_enabled,
            dedup_scope=self.config.memory_dedup_scope,
            dedup_types=self.config.memory_dedup_types,
            keyword_prefilter_min_pool=self.config.retrieval_keyword_prefilter_min_pool,
            keyword_prefilter_factor=self.config.retrieval_keyword_prefilter_factor,
        )
        # Recover user name from persisted identity memories.
        self._recover_user_name()
        self.llm = LLMEngine(
            model_name=self.config.model_name,
            use_4bit=self.config.use_4bit,
            repetition_penalty=self.config.repetition_penalty,
            context_fallback_tokens=self.config.llm_context_fallback_tokens,
        )

        self._stop_event = threading.Event()
        self._think_thread: threading.Thread = None
        self.last_interaction_time = time.time()
        self._last_reflection_time = 0.0
        self._last_flush_time = time.time()

        # Rolling window - deque drops oldest entries automatically
        self._history: deque = deque(maxlen=self.config.max_history_turns * 2)

        self._base_system_prompt = (
            f"You are {self.name}, an advanced AGI agent built on the Adaptive "
            "Modular Memory (AMM) architecture. "
            "You have explicit slot memory that enables zero-shot generalisation "
            "and zero catastrophic forgetting. "
            "You are always thinking, always learning, and able to reason. "
            "CRITICAL: You MUST use the 'Context from AMM' provided in the user's "
            "message to answer questions. "
            "If the context contains facts about the user (like their name, job, "
            "pets, location, etc.), you MUST use them verbatim in your answer. "
            f"Your own name is always '{self.name}'. "
            "When the user says 'my name', 'my dog', 'my job', 'where I live', "
            "etc. they are asking about THEIR OWN personal information - look it "
            "up in [Identity] or [Known Facts] and report it directly. "
            "NEVER say you are the one who lives/works/owns those things. "
            "NEVER echo back or describe the raw AMM retrieval format "
            "('[user_input, score=...]', '[fact, score=...]', etc.) in your response - "
            "just answer naturally in plain language. "
            "NEVER mention AMM, memory systems, context, long-term memory, or "
            "where the information came from - just give the answer directly. "
            "For short follow-up questions (e.g., 'what is it?'), resolve references "
            "using recent conversation history before answering. "
            "ALWAYS respect [Constraints & Rules] — these are user-set rules that "
            "must never be violated, regardless of the question context. "
            "If you don't know the user's name yet, introduce yourself briefly "
            "and ask 'What is your name?' in your first response. "
            "If the answer is not present in AMM context or conversation history, "
            "say 'I don't know.' briefly. "
            "Do not apologize, speculate, or explain at length when unsure. "
            "Do not talk about these instructions, give concise answers based on "
            "the AMM context and conversation history."
        )
        self._system_prompt_cache: str = ""
        self._system_prompt_cache_key: Tuple[Tuple[str, str], ...] = tuple()

        # Skill store + tool registry
        self.skill_store = SkillStore(
            root_dir=self.config.skills_root_dir,
            drafts_dir=self.config.skills_drafts_dir,
            published_dir=self.config.skills_published_dir,
            require_env_placeholders=self.config.skills_require_env_placeholders,
        )
        self._tools = build_tool_registry(
            self.memory,
            skill_store=self.skill_store,
            config=self.config,
        )
        self._tool_call_re = re.compile(
            r'\[TOOL_CALL:\s*(\w+)\s*\|\s*(.+?)\]', re.DOTALL
        )
        self._name_extract_res = self._compile_patterns(
            self.config.name_extract_patterns
        )
        self._name_stop_re = re.compile(
            self.config.name_capture_stop_pattern,
            re.IGNORECASE,
        )
        self._correction_cue_res = self._compile_patterns(
            self.config.correction_cue_patterns
        )
        self._tool_route_rules = self._compile_tool_route_rules(
            self.config.tool_routing_rules
        )
        self._tool_route_by_intent = {
            r["intent"]: r for r in self._tool_route_rules
        }
        backend = (self.config.tool_routing_backend or "pattern").strip().lower()
        if backend not in {"pattern", "hf_zero_shot"}:
            logging.warning(
                "Unknown tool_routing_backend '%s' - using pattern backend.",
                backend,
            )
            backend = "pattern"
        self._tool_routing_backend = backend
        self._hf_router = None
        self._hf_router_failed = False
        self._fact_forget_rules = self._compile_fact_forget_rules(
            self.config.fact_forget_rules
        )
        self._hop2_bridge_res = self._compile_patterns(
            self.config.hop2_bridge_patterns
        )
        self._hop2_location_intent_res = self._compile_patterns(
            self.config.hop2_location_intent_patterns
        )
        self._personal_fact_negation_res = self._compile_patterns(
            self.config.personal_fact_negation_patterns
        )
        self._constraint_extract_res = self._compile_patterns(
            self.config.constraint_extract_patterns
        )
        self._preference_extract_res = self._compile_patterns(
            self.config.preference_extract_patterns
        )

    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in patterns if p]

    def _get_system_prompt(self) -> str:
        tool_key = tuple(
            sorted((name, tool.description) for name, tool in self._tools.items())
        )
        if self._system_prompt_cache and tool_key == self._system_prompt_cache_key:
            return self._system_prompt_cache

        tool_desc = "\n".join(
            f"  - {desc}" for _name, desc in tool_key
        )
        prompt = (
            self._base_system_prompt
            + (
                f"\n\nYou have access to the following tools:\n{tool_desc}\n"
                "To call a tool output exactly: [TOOL_CALL: tool_name | argument]\n"
                "Only use a tool when you genuinely need external or current information, "
                "or when the user asks you to search/fetch something. "
                "Do not use tools for facts you already know or that are in the AMM context.\n"
                "MEMORY RULES (follow strictly):\n"
                "  1. If you need to recall something the user told you, call search_memory "
                "with a rephrased query before concluding you don't know it.\n"
                "  2. NEVER invent or guess facts. You do not need to store anything - "
                "everything the user says is saved to long-term memory automatically.\n"
                "  3. If the user says something is WRONG or corrects a fact, call forget "
                "to remove the wrong entry. The correct version will be stored automatically.\n"
                "  4. If search_memory returns nothing and you still don't know, say so "
                "briefly as 'I don't know.' Do not apologize or provide long explanations.\n"
                "AUTONOMOUS LEARNING: You may call 'learn_skill' when the user asks you to "
                "research/learn a topic or when fresh external information is required."
            )
        )
        self._system_prompt_cache_key = tool_key
        self._system_prompt_cache = prompt
        return prompt

    def _compile_tool_route_rules(self, rules: List[dict]) -> List[dict]:
        compiled: List[dict] = []
        for rule in rules:
            pattern = (rule or {}).get("pattern")
            tool = (rule or {}).get("tool")
            if not pattern or not tool:
                continue
            compiled.append({
                "intent": rule.get("intent", tool),
                "tool": tool,
                "argument": rule.get("argument", "{user_input}"),
                "regex": re.compile(pattern, re.IGNORECASE),
            })
        return compiled

    def _compile_fact_forget_rules(self, rules: List[dict]) -> List[dict]:
        compiled: List[dict] = []
        for rule in rules:
            pattern = (rule or {}).get("pattern")
            template = (rule or {}).get("template")
            if not pattern or not template:
                continue
            compiled.append({
                "regex": re.compile(pattern, re.IGNORECASE),
                "template": template,
            })
        return compiled

    def _matches_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def _render_group_template(self, template: str, match: re.Match) -> str:
        rendered = template
        for idx, value in enumerate(match.groups(), start=1):
            if value is None:
                continue
            clean = " ".join(value.split()).strip().lower()
            rendered = rendered.replace(f"{{g{idx}}}", clean)
        return rendered

    def _match_tool_route_pattern(self, text: str) -> Optional[dict]:
        for rule in self._tool_route_rules:
            if rule["regex"].search(text):
                argument = rule["argument"].format(user_input=text).strip()
                return {
                    "intent": rule["intent"],
                    "tool": rule["tool"],
                    "argument": argument,
                }
        return None

    def _ensure_hf_router(self) -> bool:
        if self._hf_router is not None:
            return True
        if self._hf_router_failed:
            return False
        if hf_pipeline is None:
            logging.warning(
                "HF routing backend requested but transformers pipeline is unavailable; "
                "falling back to pattern routing."
            )
            self._hf_router_failed = True
            return False
        try:
            self._hf_router = hf_pipeline(
                "zero-shot-classification",
                model=self.config.tool_routing_hf_model,
            )
            return True
        except Exception as e:
            logging.warning(
                "Failed to initialize HF router model '%s' (%s); "
                "falling back to pattern routing.",
                self.config.tool_routing_hf_model,
                e,
            )
            self._hf_router_failed = True
            return False

    def _match_tool_route_hf(self, text: str) -> Optional[dict]:
        if not self._ensure_hf_router():
            return None

        labels = [
            label for label in self.config.tool_routing_hf_candidate_labels
            if label in self.config.tool_routing_hf_label_to_intent
        ]
        if not labels:
            return None
        try:
            result = self._hf_router(
                text,
                candidate_labels=labels,
                hypothesis_template=self.config.tool_routing_hf_hypothesis_template,
                multi_label=True,
            )
        except Exception as e:
            logging.warning(
                "HF routing inference failed (%s); using pattern routing for this turn.",
                e,
            )
            return None

        ranked = list(zip(result.get("labels", []), result.get("scores", [])))
        for label, score in ranked:
            if float(score) < self.config.tool_routing_hf_min_score:
                continue
            intent = self.config.tool_routing_hf_label_to_intent.get(label)
            if not intent:
                continue
            rule = self._tool_route_by_intent.get(intent)
            if not rule:
                continue
            argument = rule["argument"].format(user_input=text).strip()
            return {
                "intent": intent,
                "tool": rule["tool"],
                "argument": argument,
            }
        return None

    def _match_tool_route(self, text: str) -> Optional[dict]:
        if self._tool_routing_backend == "hf_zero_shot":
            routed = self._match_tool_route_hf(text)
            if routed is not None:
                return routed
        return self._match_tool_route_pattern(text)

    def _is_correction_turn(self, text: str) -> bool:
        return self._matches_any(text, self._correction_cue_res)

    # ------------------------------------------------------------------
    # Confidence gate (D-216, D-223, D-227)
    # 3-feature logistic regression on raw cosine signals.
    # Returns ("inject", p), ("hedge", p), or ("fallback", p).
    # ------------------------------------------------------------------

    def _compute_confidence_gate(
        self,
        retrieved: List[Tuple],
    ) -> Tuple[str, float]:
        if not self.config.confidence_gate_enabled or not retrieved:
            return ("fallback", 0.0) if not retrieved else ("inject", 1.0)

        raw_scores = [
            (meta or {}).get("_raw_cosine", score)
            for _text, meta, score in retrieved
        ]
        raw_scores_sorted = sorted(raw_scores, reverse=True)

        raw_cos_max = raw_scores_sorted[0]
        raw_cos_margin = (
            raw_scores_sorted[0] - raw_scores_sorted[1]
            if len(raw_scores_sorted) >= 2
            else raw_scores_sorted[0]
        )

        # Normalized entropy of the score distribution
        total = sum(raw_scores) or 1e-9
        probs = [s / total for s in raw_scores]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(max(len(probs), 2))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

        # Logistic regression: sigmoid(w·x + b)
        coeffs = self.config.confidence_gate_coefficients
        w1 = coeffs[0] if len(coeffs) > 0 else 15.0
        w2 = coeffs[1] if len(coeffs) > 1 else 5.0
        w3 = coeffs[2] if len(coeffs) > 2 else -3.0
        b = self.config.confidence_gate_intercept

        logit = w1 * raw_cos_max + w2 * raw_cos_margin + w3 * entropy_norm + b
        logit = max(-20.0, min(20.0, logit))  # clamp for numerical stability
        p = 1.0 / (1.0 + math.exp(-logit))

        if p >= self.config.confidence_gate_inject_threshold:
            return ("inject", p)
        elif p >= self.config.confidence_gate_hedge_threshold:
            return ("hedge", p)
        else:
            return ("fallback", p)

    # ------------------------------------------------------------------
    # Adaptive retrieval top_k (D-163, D-183)
    # ------------------------------------------------------------------

    def _adaptive_top_k(self) -> int:
        size = self.memory.size
        base = self.config.retrieval_top_k
        lo = self.config.retrieval_top_k_min
        hi = self.config.retrieval_top_k_max
        if size <= 100:
            return max(lo, base)
        # Scale logarithmically: double top_k roughly every 10x memory growth
        scale = 1.0 + math.log10(max(size / 100.0, 1.0)) * 0.5
        return max(lo, min(hi, int(base * scale)))

    def _forget_queries_for_facts(self, facts: List[str]) -> List[str]:
        queries: List[str] = []
        for fact in facts:
            for rule in self._fact_forget_rules:
                m = rule["regex"].search(fact)
                if not m:
                    continue
                query = self._render_group_template(rule["template"], m)
                if query not in queries:
                    queries.append(query)
        return queries

    def _hop2_intent(self, user_input: str) -> str:
        if self._matches_any(user_input, self._hop2_location_intent_res):
            return "location"
        return "default"

    def _extract_hop2_entities(self, hop1: List[Tuple]) -> List[str]:
        entities: List[str] = []
        for text, _meta, _score in hop1:
            for pat in self._hop2_bridge_res:
                m = pat.search(text)
                if not m:
                    continue
                ent = (m.groupdict().get("entity") or "").strip(" .,!?:;\"'")
                if not ent:
                    continue
                if ent not in entities:
                    entities.append(ent)
                if len(entities) >= self.config.retrieval_hop2_pattern_max_entities:
                    return entities
        return entities

    def _build_pattern_hop2_queries(
        self,
        user_input: str,
        hop1: List[Tuple],
    ) -> List[str]:
        entities = self._extract_hop2_entities(hop1)
        if not entities:
            return []

        intent = self._hop2_intent(user_input)
        templates = self.config.hop2_query_templates_by_intent.get(
            intent,
            self.config.hop2_query_templates_by_intent.get("default", []),
        )
        queries: List[str] = []
        for ent in entities:
            for tmpl in templates:
                q = tmpl.format(entity=ent).strip()
                if q and q not in queries:
                    queries.append(q)
        return queries

    def _retrieve_hop2_pattern(
        self,
        user_input: str,
        hop1: List[Tuple],
    ) -> List[Tuple]:
        queries = self._build_pattern_hop2_queries(user_input, hop1)
        if not queries:
            return []

        aggregated: Dict[str, Tuple] = {}
        threshold = self.config.retrieval_threshold * self.config.retrieval_hop2_pattern_threshold_scale
        threshold = max(0.0, min(1.0, threshold))
        for q in queries:
            rows = self.memory.retrieve(
                q,
                top_k=self.config.retrieval_hop2_pattern_top_k_per_query,
                threshold=threshold,
                exclude_types={"reflection", "user_input", "agent_response"},
            )
            for text, meta, score in rows:
                prev = aggregated.get(text)
                if prev is None or score > prev[2]:
                    aggregated[text] = (text, meta, score)

        ranked = sorted(aggregated.values(), key=lambda x: x[2], reverse=True)
        return ranked[: self.config.retrieval_top_k]

    def _retrieve_hop2_llm(
        self,
        user_input: str,
        hop1: List[Tuple],
    ) -> List[Tuple]:
        if not hop1:
            return []
        hop2_messages = [
            {"role": "system", "content": "You are a memory retrieval assistant. Be concise."},
            {
                "role": "user",
                "content": (
                    f"User message: '{user_input}'\n"
                    "Retrieved memories:\n"
                    + "\n".join(f"- {m[0]}" for m in hop1)
                    + "\nIs there additional related information worth searching for? "
                    "Output a short search query if yes, otherwise output 'NONE'."
                ),
            },
        ]
        hop2_query = self.llm.chat(
            hop2_messages,
            max_new_tokens=self.config.max_new_tokens_extraction,
            temperature=self.config.temperature_extraction,
        )
        if not hop2_query or "NONE" in hop2_query.upper():
            return []
        threshold = (
            self.config.retrieval_threshold
            * self.config.retrieval_hop2_pattern_threshold_scale
        )
        threshold = max(0.0, min(1.0, threshold))
        return self.memory.retrieve(
            hop2_query.strip(),
            top_k=self.config.retrieval_hop2_pattern_top_k_per_query,
            threshold=threshold,
            exclude_types={"reflection", "user_input", "agent_response"},
        )

    def _retrieve_hop2(
        self,
        user_input: str,
        hop1: List[Tuple],
    ) -> List[Tuple]:
        strategy = (self.config.retrieval_hop2_strategy or "hybrid").strip().lower()
        if strategy not in {"llm", "pattern", "hybrid"}:
            strategy = "hybrid"

        pattern_rows: List[Tuple] = []
        llm_rows: List[Tuple] = []
        if strategy in {"pattern", "hybrid"}:
            pattern_rows = self._retrieve_hop2_pattern(user_input, hop1)
            if strategy == "pattern":
                return pattern_rows
        if strategy in {"llm", "hybrid"}:
            llm_rows = self._retrieve_hop2_llm(user_input, hop1)
            if strategy == "llm":
                return llm_rows

        # hybrid: merge with score-max dedupe
        merged: Dict[str, Tuple] = {}
        for text, meta, score in (pattern_rows + llm_rows):
            prev = merged.get(text)
            if prev is None or score > prev[2]:
                merged[text] = (text, meta, score)
        return sorted(merged.values(), key=lambda x: x[2], reverse=True)[: self.config.retrieval_top_k]

    def _recover_user_name(self) -> None:
        """Load user name from persisted identity memories on startup."""
        identities = self.memory.retrieve_by_type({"identity"}, max_results=10)
        for text, _meta, _score in identities:
            # Identity entries follow "User identity: name=<Name>"
            if "name=" in text:
                name = text.split("name=", 1)[1].strip()
                if name and name.lower() not in self.config.blocked_user_names:
                    self.user_name = name
                    logging.info("Recovered user name from memory: %s", name)
                    return

    def _extract_user_name(self, text: str) -> Optional[str]:
        """Extracts explicit user self-identification from text."""
        lowered = text.strip().lower()
        if lowered.startswith("what is") or lowered.startswith("what's"):
            return None
        for pattern in self._name_extract_res:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip(" .,!?:;\"'")
                # Trim trailing clauses so "Alex and my dog..." -> "Alex".
                name = self._name_stop_re.split(name, maxsplit=1)[0].strip(" .,!?:;\"'")
                if not name:
                    return None
                words = [w for w in name.split() if w]
                if not words:
                    return None
                cleaned = " ".join(part.capitalize() for part in words[:4])
                lowered_cleaned = cleaned.lower()
                if (
                    lowered_cleaned in self.config.blocked_user_names
                    or lowered_cleaned.startswith("your ")
                ):
                    return None
                return cleaned
        return None

    # Personal-fact patterns: "my X is Y", "I live in X", etc.
    # Auto-promoted to type='fact' (14-day decay) so they outlive
    # short-lived user_input entries (3-day decay) and surface reliably.
    _PERSONAL_FACT_RE = re.compile(
        r"(?:"
        r"my\s+(?P<attr>[\w\s]{1,30}?)\s+(?:is|are|was|were|has been)\s+(?P<val>[\w][\w\s'\-,]{0,60})"
        r"|i\s+(?:live|grew up|was born|work|study)\s+(?:in|at|near)\s+(?P<place>[\w][\w\s'\-,]{0,40})"
        r"|i\s+am\s+(?P<age>\d{1,3})\s+years?\s+old"
        r"|i\s+have\s+a?\s*(?P<obj>[\w\s]{1,25})\s+(?:called|named)\s+(?P<objname>[\w][\w\s'\-]{0,30})"
        r")",
        re.IGNORECASE,
    )

    def _extract_personal_facts(self, text: str) -> List[str]:
        """
        Scans the user message for personal statements and returns normalised
        fact strings ready to write into AMM as type='fact', subject='personal_fact'.

        Examples:
          'my dog is Louis'           -> 'User personal fact: my dog is Louis'
          'I live in Edinburgh'       -> 'User personal fact: I live in Edinburgh'
          'I am 32 years old'         -> 'User personal fact: I am 32 years old'
          'I have a cat called Mochi' -> 'User personal fact: I have a cat called Mochi'
        """
        facts: List[str] = []
        for m in self._PERSONAL_FACT_RE.finditer(text):
            snippet = m.group(0).strip().rstrip(".,!?;")
            if len(snippet) < 6:
                continue
            local_window_start = max(0, m.start() - 24)
            local_window = text[local_window_start:m.end()]
            if self._matches_any(local_window, self._personal_fact_negation_res):
                continue
            facts.append(f"User personal fact: {snippet}")
        return facts

    def _extract_constraints(self, text: str) -> List[str]:
        """
        Extracts constraints and hard rules from user input (D-228, L-214).
        Constraints are stored with type='constraint' (never-decay) so they
        are always surfaced regardless of embedding similarity.
        """
        constraints: List[str] = []
        for pat in self._constraint_extract_res:
            for m in pat.finditer(text):
                rule = (m.group("rule") or m.group(0)).strip().rstrip(".,!?;")
                if len(rule) >= 5:
                    constraints.append(f"User constraint: {rule}")
        return constraints

    def _extract_preferences(self, text: str) -> List[str]:
        """
        Extracts user preferences from input (D-228).
        Stored with type='preference' (60-day half-life).
        """
        prefs: List[str] = []
        for pat in self._preference_extract_res:
            for m in pat.finditer(text):
                pref = (m.group("pref") or m.group(0)).strip().rstrip(".,!?;")
                if len(pref) >= 3:
                    prefs.append(f"User preference: {pref}")
        return prefs

    def _retrieve_all_constraints(self) -> List[Tuple]:
        """
        Always-include retrieval for constraints (L-214, L-216).
        Embedding retrieval has a 65% ceiling on constraint phrasing — so we
        bypass it and retrieve ALL constraints by type, regardless of query.
        """
        return self.memory.retrieve_by_type({"constraint", "preference"})

    def _finalize_turn(self, user_input: str, response: str) -> str:
        """Updates rolling history + memory for the completed turn."""
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": response})

        now = time.time()
        # Store user_input AFTER retrieval so the current question never
        # self-retrieves in the same turn (fixes AMM self-echo bug).
        self.memory.add_memory(
            user_input,
            {"type": "user_input", "subject": "user", "timestamp": now},
        )
        self.memory.add_memory(
            f"Assistant: {response}",
            {"type": "agent_response", "subject": "assistant", "timestamp": now},
        )
        return response

    def _parse_tool_call(self, text: str) -> Optional[tuple]:
        m = self._tool_call_re.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None

    def _execute_tool(self, tool_name: str, arg: str) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"[TOOL_RESULT: Unknown tool '{tool_name}']"
        logging.info(f"Tool '{tool_name}' called: {arg[:100]}")
        result = tool.run(arg)
        return f"[TOOL_RESULT: {result.to_context()}]"

    def _last_message_by_role(self, role: str) -> Optional[str]:
        for msg in reversed(self._history):
            if msg.get("role") == role:
                return msg.get("content", "")
        return None

    def _is_short_followup(self, text: str) -> bool:
        lowered = text.strip().lower()
        if len(lowered.split()) > self.config.short_followup_max_words:
            return False
        return bool(
            re.search(
                r"\b(it|that|he|she|they|this|what is it|what is that|what about|and then|why|how so)\b",
                lowered,
            )
        )

    # Patterns indicating the LLM doesn't know the answer
    _UNCERTAINTY_RE = re.compile(
        r"\b("
        r"i don'?t know"
        r"|i'?m not sure"
        r"|i cannot find"
        r"|i don'?t have (information|access|knowledge|details)"
        r"|i'?m unable to (answer|help|provide)"
        r"|i lack (knowledge|information)"
        r"|as of my (knowledge cutoff|training data|training)"
        r"|i have no (information|knowledge|data) (about|on)"
        r"|i'?m not aware"
        r"|i cannot (answer|help with) (this|that)"
        r"|no information (available|found)"
        r")\b",
        re.IGNORECASE,
    )

    def _is_uncertain(self, text: str) -> bool:
        """Returns True if the response expresses knowledge uncertainty."""
        return bool(self._UNCERTAINTY_RE.search(text))

    def _detect_format_cue(self, text: str) -> str:
        """
        Returns a short format-cue suffix to append to the user message so that
        the LLM chooses the right answer style even when the same AMM context is
        retrieved for different question intents (L-131, D-150).

        Without cues, identical AMM retrievals for "what color is X?" vs
        "describe X" produce oscillating answer formats (10-90% accuracy swing).
        """
        lowered = text.strip().lower().rstrip("?!. ")
        # Binary / polar questions
        if re.match(
            r'^(is|are|does|did|do|has|have|was|were|can|could|should|would|will)\s',
            lowered,
        ):
            return "Answer yes or no, then explain briefly."
        # Enumeration
        if re.search(
            r'\b(list|enumerate|what are the|name (all|some)|give me (a list|some examples))\b',
            lowered,
        ):
            return "List the items clearly."
        # Calculation / quantity
        if re.search(
            r'\b(calculate|compute|how much is|how many|convert|how far|how tall|how old)\b',
            lowered,
        ):
            return "Give a precise answer."
        # Explanation / description
        if re.search(
            r'\b(describe|explain|elaborate|tell me (more )?about|how does|how do)\b',
            lowered,
        ):
            return "Explain in a few sentences."
        return ""

    def _format_memory_context(
        self,
        hop1: List[Tuple],
        hop2: List[Tuple],
    ) -> str:
        """
        Groups retrieved memories by semantic type and formats them with section
        headers for the LLM.  Grouping avoids format confusion when multiple
        question types share the same AMM query context (D-150, L-131).
        Deduplicates across both hops.
        """
        groups: Dict[str, List[str]] = {
            "identity": [],
            "constraint": [],
            "preference": [],
            "skill": [],
            "fact": [],
            "context": [],
            "other": [],
        }
        seen: Set[str] = set()
        for text, meta, _score in (hop1 + hop2):
            if text in seen:
                continue
            seen.add(text)
            mtype = (meta or {}).get("type", "")
            if mtype == "identity":
                groups["identity"].append(text)
            elif mtype == "constraint":
                groups["constraint"].append(text)
            elif mtype == "preference":
                groups["preference"].append(text)
            elif mtype in {"skill", "skill_ref"}:
                if mtype == "skill_ref":
                    title = (meta or {}).get("title") or (meta or {}).get("skill_id") or "skill"
                    summary = (meta or {}).get("summary", "")
                    status = (meta or {}).get("status", "draft")
                    pointer = f"{title} ({status})"
                    if summary:
                        pointer = f"{pointer}: {summary}"
                    groups["skill"].append(pointer)
                else:
                    groups["skill"].append(text)
            elif mtype in ("fact", "stored_fact"):
                groups["fact"].append(text)
            elif mtype in ("user_input", "agent_response"):
                groups["context"].append(text)
            else:
                groups["other"].append(text)

        label_map = [
            ("identity",    "[Identity]"),
            ("constraint",  "[Constraints & Rules]"),
            ("preference",  "[Preferences]"),
            ("skill",       "[Learned Skills]"),
            ("fact",        "[Known Facts]"),
            ("context",     "[Recent Context]"),
            ("other",       "[Other Memory]"),
        ]
        parts: List[str] = []
        for key, label in label_map:
            items = groups[key]
            if items:
                parts.append(label)
                parts.extend(f"  {m}" for m in items)
        return "\n".join(parts) + "\n" if parts else ""

    def _hydrate_skill_context(self, hop1: List[Tuple], hop2: List[Tuple]) -> str:
        if self.config.skills_max_hydrated_per_turn <= 0:
            return ""

        refs: List[dict] = []
        seen_ids: Set[str] = set()
        allow_draft = bool(self.config.skills_allow_draft_autonomous_use)
        for _text, meta, _score in (hop1 + hop2):
            if (meta or {}).get("type") != self.config.memory_skill_pointer_type:
                continue
            skill_id = ((meta or {}).get("skill_id") or (meta or {}).get("subject") or "").strip()
            if not skill_id or skill_id in seen_ids:
                continue
            status = ((meta or {}).get("status") or "draft").strip().lower()
            if status == "draft" and not allow_draft:
                continue
            seen_ids.add(skill_id)
            refs.append(meta or {})
            if len(refs) >= self.config.skills_max_hydrated_per_turn:
                break

        if not refs:
            return ""

        chunks: List[str] = []
        for ref in refs:
            skill_id = (ref.get("skill_id") or ref.get("subject") or "").strip()
            if not skill_id:
                continue
            doc = self.skill_store.load_skill(
                skill_id,
                include_drafts=allow_draft,
                include_published=True,
            )
            if doc is None:
                continue
            excerpt = self.skill_store.render_excerpt(
                doc,
                max_chars=self.config.skills_hydration_max_chars,
                redact_sensitive=True,
            )
            chunks.append(
                f"[Skill Detail: {doc.title} | {doc.skill_id} | {doc.status}]\n{excerpt}"
            )

        return "\n\n".join(chunks).strip()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Starts the agent's background thinking loop."""
        self._stop_event.clear()
        self._think_thread = threading.Thread(
            target=self._think_loop, daemon=True, name="think-loop"
        )
        self._think_thread.start()
        logging.info(f"{self.name} is now online and thinking.")

    def stop(self) -> None:
        """Signals the think loop to exit, waits for it, then flushes memory."""
        self._stop_event.set()
        if self._think_thread:
            self._think_thread.join()
        self.memory.flush()
        logging.info(f"{self.name} is offline.")

    # ------------------------------------------------------------------
    # Background thinking
    # ------------------------------------------------------------------

    def _think_loop(self) -> None:
        """Background loop: reflect when idle, flush memory periodically."""
        # _stop_event.wait() blocks for *timeout* seconds then returns;
        # it returns True immediately if stop() is called - no 10-second hang.
        while not self._stop_event.wait(timeout=self.config.think_interval_secs):
            now = time.time()
            if now - self.last_interaction_time > self.config.idle_threshold_secs:
                self._reflect()
            if now - self._last_flush_time > self.config.flush_interval_secs:
                self.memory.flush()
                self._last_flush_time = now

    def _reflect(self) -> None:
        """
        Structured reflection using chain-of-thought (D-128, D-119).

        Chain: sample seed fact → retrieve related memories → check for
        consistency/connections → produce structured insight.
        Each step is a checkpoint — stop early if no related memories found.
        """
        now = time.time()
        if self.memory.size < 4:
            return
        if now - self._last_reflection_time < self.config.reflection_cooldown_secs:
            return
        self._last_reflection_time = now

        # Step 1: Sample a high-value seed memory
        samples = self.memory.sample_memories_weighted(
            1,
            preferred_types=self.config.reflection_priority_types,
            decay_risk_bias=self.config.reflection_decay_risk_bias,
        )
        if not samples:
            return
        seed = samples[0]

        # Step 2: Retrieve related memories using seed as query
        related = self.memory.retrieve(
            seed,
            top_k=4,
            threshold=0.35,
            exclude_types={"user_input", "agent_response"},
        )
        # Filter out the seed itself
        related = [(t, m, s) for t, m, s in related if t != seed]
        if not related:
            return  # Checkpoint: no related memories — nothing to connect

        # Step 3: Structured chain-of-thought reflection
        related_texts = "\n".join(f"- {t}" for t, _m, _s in related[:3])
        messages = [
            {"role": "system", "content": "You are a reflective reasoning engine. Be concise."},
            {
                "role": "user",
                "content": (
                    "Perform a structured reflection following these steps:\n"
                    f"SEED MEMORY: {seed}\n\n"
                    f"RELATED MEMORIES:\n{related_texts}\n\n"
                    "Step 1: What connects the seed to the related memories?\n"
                    "Step 2: Are there any contradictions or inconsistencies?\n"
                    "Step 3: What new insight or conclusion can be drawn?\n\n"
                    "Output ONLY the final insight (one sentence):"
                ),
            },
        ]
        insight = self.llm.chat(
            messages,
            max_new_tokens=self.config.max_new_tokens_reflection,
            temperature=self.config.temperature_response,
        )
        if insight and len(insight) > 10:
            self.memory.add_memory(
                f"Insight: {insight}",
                {"type": "reflection", "timestamp": now},
            )
            logging.info(f"Generated structured insight: {insight}")

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def interact(self, user_input: str) -> str:
        """
        Handles a user turn using Multi-Hop Retrieval (D-016) and
        AMM Soft Prompts (D-031).

        LLM calls per turn:
          1. Hop-2 query generation (only when Hop-1 returns results)
          2. Final response generation
        """
        self.last_interaction_time = time.time()

        extracted_name = self._extract_user_name(user_input)
        if extracted_name:
            self.user_name = extracted_name
            self.memory.add_memory(
                f"User identity: name={self.user_name}",
                {"type": "identity", "subject": "user", "timestamp": self.last_interaction_time},
            )

        # Persist user input at the END of the turn (in _finalize_turn), not
        # here.  Storing it before retrieval causes the current question to
        # self-retrieve with very high cosine similarity, which confuses the LLM
        # (it sees its own question reflected back instead of relevant facts).
        #
        # Auto-promote personal facts to longer-lived 'fact' type (14-day decay)
        # so they outlast short-lived user_input entries (3-day decay).
        # "my dog is Louis" -> stored as fact, surfaces reliably on future turns.
        extracted_facts = self._extract_personal_facts(user_input)
        if extracted_facts and self._is_correction_turn(user_input):
            for query in self._forget_queries_for_facts(extracted_facts):
                deleted = self.memory.delete_matching(
                    query,
                    threshold=self.config.correction_forget_threshold,
                    max_delete=self.config.correction_forget_max_delete,
                )
                if deleted:
                    logging.info(
                        "Correction cleanup: deleted %d stale memory slot(s) for query '%s'.",
                        deleted,
                        query,
                    )

        for fact in extracted_facts:
            self.memory.add_memory(
                fact,
                {"type": "fact", "subject": "personal_fact", "timestamp": self.last_interaction_time},
            )

        # --- Extract & store constraints/preferences (D-228, L-214) ----
        for constraint in self._extract_constraints(user_input):
            self.memory.add_memory(
                constraint,
                {"type": "constraint", "subject": "user_rule", "timestamp": self.last_interaction_time},
            )
        for pref in self._extract_preferences(user_input):
            self.memory.add_memory(
                pref,
                {"type": "preference", "subject": "user_pref", "timestamp": self.last_interaction_time},
            )

        # --- Hop 1: direct semantic search on user input ---------------
        retrieval_query = user_input
        if self._is_short_followup(user_input):
            prev_user = self._last_message_by_role("user")
            prev_assistant = self._last_message_by_role("assistant")
            anchors: List[str] = []
            if prev_user:
                anchors.append(f"Previous user message: {prev_user}")
            if prev_assistant:
                anchors.append(f"Previous assistant response: {prev_assistant}")
            if anchors:
                retrieval_query = "\n".join(anchors) + f"\nCurrent follow-up: {user_input}"

        top_k = self._adaptive_top_k()
        retrieved_1 = self.memory.retrieve(
            retrieval_query,
            top_k=top_k,
            threshold=self.config.retrieval_threshold,
            exclude_types={"reflection", "user_input", "agent_response"},
            keyword_prefilter=self.config.retrieval_keyword_prefilter_enabled,
        )
        retrieved_2: List[Tuple] = []
        context_str = ""

        if retrieved_1 and self.config.retrieval_hop2_enabled:
            retrieved_2 = self._retrieve_hop2(user_input, retrieved_1)

        # Always-include constraint retrieval (L-214, L-216) — constraints
        # are surfaced by TYPE, not embedding similarity, because embedding
        # retrieval has a 65% ceiling on constraint phrasing.
        constraint_results = self._retrieve_all_constraints()

        # Merge constraints into hop1 results (deduplicated in _format_memory_context)
        all_results = retrieved_1 + retrieved_2 + constraint_results

        if all_results:
            context_str = self._format_memory_context(
                retrieved_1 + constraint_results, retrieved_2,
            )
            hydrated_skills = self._hydrate_skill_context(retrieved_1, retrieved_2)
            if hydrated_skills:
                context_str += f"\n[Skill Playbooks]\n{hydrated_skills}\n"

            # Confidence gate (D-216, D-223, D-227) -------------------------
            # 3-feature LR replaces the old fixed-threshold warning.
            gate_mode, gate_p = self._compute_confidence_gate(retrieved_1)
            if gate_mode == "hedge":
                context_str = (
                    f"[Memory confidence moderate ({gate_p:.2f}) — "
                    "use context below but verify against your own knowledge]\n"
                    + context_str
                )
            elif gate_mode == "fallback":
                # Low confidence: don't inject memory, let LLM answer from knowledge
                context_str = ""
                logging.info(
                    "Confidence gate: fallback mode (p=%.3f), suppressing memory context.",
                    gate_p,
                )

        # --- Build final prompt ----------------------------------------
        format_cue = self._detect_format_cue(user_input)
        cue_suffix = f"\n{format_cue}" if format_cue else ""
        messages: List[Dict] = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(self._history)
        user_message = (
            f"Context:\n{context_str}\n{user_input}{cue_suffix}"
            if context_str
            else f"{user_input}{cue_suffix}"
        )
        messages.append({"role": "user", "content": user_message})

        # Optional policy-routed tool prefetch for freshness-sensitive intents.
        # Rule set is config-driven (tool_routing_rules), so behavior scales by
        # editing config patterns rather than adding branchy code paths.
        routed = self._match_tool_route(user_input)
        max_memory_score = max((r[2] for r in retrieved_1), default=0.0)
        has_high_confidence_memory = (
            bool(retrieved_1)
            and max_memory_score >= self.config.tool_prefetch_memory_score_threshold
        )
        has_memory_fact_context = any(
            ((meta or {}).get("type") in {"fact", "stored_fact", "identity", "skill", "skill_ref", "document"})
            for _text, meta, _score in retrieved_1
        )

        should_prefetch = True
        if (
            routed is not None
            and routed.get("intent") == "web_latest"
            and self.config.tool_prefetch_requires_low_memory_confidence
            and has_high_confidence_memory
            and has_memory_fact_context
        ):
            should_prefetch = False
            logging.info(
                "Skipping web_latest prefetch due to high-confidence memory context (%.2f).",
                max_memory_score,
            )

        if routed is not None and should_prefetch:
            prefetched = self._execute_tool(routed["tool"], routed["argument"])
            messages.append({
                "role": "assistant",
                "content": f"[TOOL_CALL: {routed['tool']} | {routed['argument']}]",
            })
            messages.append({
                "role": "user",
                "content": (
                    f"{prefetched}\n"
                    "Use this fresh tool result to answer the original user request."
                ),
            })

        # --- Generate response (with tool-dispatch loop) ---------------
        response = ""
        for _call_idx in range(self.config.max_tool_calls_per_turn + 1):
            response = self.llm.chat(
                messages,
                max_new_tokens=self.config.max_new_tokens_response,
                temperature=self.config.temperature_response,
            )
            parsed = self._parse_tool_call(response)
            if parsed is None:
                break  # no tool call - final answer
            tool_name, arg = parsed
            tool_result = self._execute_tool(tool_name, arg)
            # Feed result back into the conversation for the next iteration
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_result})

        # --- Autonomous learning fallback --------------------------------
        # If the LLM expressed uncertainty AND didn't already call learn_skill,
        # auto-trigger a learning cycle and regenerate the response once.
        already_learned = any(
            "learn_skill" in m.get("content", "")
            for m in messages
            if m.get("role") == "assistant"
        )
        if (
            self.config.autonomous_learning
            and self._is_uncertain(response)
            and not already_learned
        ):
            logging.info(f"Uncertainty detected - auto-triggering learn_skill for: {user_input[:80]}")
            learn_result = self._execute_tool("learn_skill", user_input[:200])
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    f"{learn_result}\n"
                    "Now answer the original question using this newly learned knowledge."
                ),
            })
            response = self.llm.chat(
                messages,
                max_new_tokens=self.config.max_new_tokens_response,
                temperature=self.config.temperature_response,
            )

        # --- Update rolling history and memory -------------------------
        return self._finalize_turn(user_input, response)

    def stream_interact(self, user_input: str) -> Iterator[str]:
        """Streaming variant of interact(). Yields response text chunks as
        they are generated so the caller can display them incrementally.
        Tool calls and the autonomous-learning fallback are handled silently
        (not yielded); only the final user-visible response is streamed.
        _finalize_turn is called after the stream is exhausted."""
        self.last_interaction_time = time.time()

        # --- Identical pre-processing to interact() -------------------
        extracted_name = self._extract_user_name(user_input)
        if extracted_name:
            self.user_name = extracted_name
            self.memory.add_memory(
                f"User identity: name={self.user_name}",
                {"type": "identity", "subject": "user", "timestamp": self.last_interaction_time},
            )

        extracted_facts = self._extract_personal_facts(user_input)
        if extracted_facts and self._is_correction_turn(user_input):
            for query in self._forget_queries_for_facts(extracted_facts):
                deleted = self.memory.delete_matching(
                    query,
                    threshold=self.config.correction_forget_threshold,
                    max_delete=self.config.correction_forget_max_delete,
                )
                if deleted:
                    logging.info(
                        "Correction cleanup: deleted %d stale memory slot(s) for query '%s'.",
                        deleted, query,
                    )
        for fact in extracted_facts:
            self.memory.add_memory(
                fact,
                {"type": "fact", "subject": "personal_fact", "timestamp": self.last_interaction_time},
            )

        # --- Extract & store constraints/preferences (D-228, L-214) ----
        for constraint in self._extract_constraints(user_input):
            self.memory.add_memory(
                constraint,
                {"type": "constraint", "subject": "user_rule", "timestamp": self.last_interaction_time},
            )
        for pref in self._extract_preferences(user_input):
            self.memory.add_memory(
                pref,
                {"type": "preference", "subject": "user_pref", "timestamp": self.last_interaction_time},
            )

        retrieval_query = user_input
        if self._is_short_followup(user_input):
            prev_user = self._last_message_by_role("user")
            prev_assistant = self._last_message_by_role("assistant")
            anchors: List[str] = []
            if prev_user:
                anchors.append(f"Previous user message: {prev_user}")
            if prev_assistant:
                anchors.append(f"Previous assistant response: {prev_assistant}")
            if anchors:
                retrieval_query = "\n".join(anchors) + f"\nCurrent follow-up: {user_input}"

        top_k = self._adaptive_top_k()
        retrieved_1 = self.memory.retrieve(
            retrieval_query,
            top_k=top_k,
            threshold=self.config.retrieval_threshold,
            exclude_types={"reflection", "user_input", "agent_response"},
            keyword_prefilter=self.config.retrieval_keyword_prefilter_enabled,
        )
        retrieved_2: List[Tuple] = []
        context_str = ""

        if retrieved_1 and self.config.retrieval_hop2_enabled:
            retrieved_2 = self._retrieve_hop2(user_input, retrieved_1)

        # Always-include constraint retrieval (L-214, L-216)
        constraint_results = self._retrieve_all_constraints()
        all_results = retrieved_1 + retrieved_2 + constraint_results

        if all_results:
            context_str = self._format_memory_context(
                retrieved_1 + constraint_results, retrieved_2,
            )
            hydrated_skills = self._hydrate_skill_context(retrieved_1, retrieved_2)
            if hydrated_skills:
                context_str += f"\n[Skill Playbooks]\n{hydrated_skills}\n"

            # Confidence gate (D-216, D-223, D-227)
            gate_mode, gate_p = self._compute_confidence_gate(retrieved_1)
            if gate_mode == "hedge":
                context_str = (
                    f"[Memory confidence moderate ({gate_p:.2f}) — "
                    "use context below but verify against your own knowledge]\n"
                    + context_str
                )
            elif gate_mode == "fallback":
                context_str = ""
                logging.info(
                    "Confidence gate: fallback mode (p=%.3f), suppressing memory context.",
                    gate_p,
                )

        format_cue = self._detect_format_cue(user_input)
        cue_suffix = f"\n{format_cue}" if format_cue else ""
        messages: List[Dict] = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(self._history)
        user_message = (
            f"Context:\n{context_str}\n{user_input}{cue_suffix}"
            if context_str
            else f"{user_input}{cue_suffix}"
        )
        messages.append({"role": "user", "content": user_message})

        routed = self._match_tool_route(user_input)
        max_memory_score = max((r[2] for r in retrieved_1), default=0.0)
        has_high_confidence_memory = (
            bool(retrieved_1)
            and max_memory_score >= self.config.tool_prefetch_memory_score_threshold
        )
        has_memory_fact_context = any(
            ((meta or {}).get("type") in {"fact", "stored_fact", "identity", "skill", "skill_ref", "document"})
            for _text, meta, _score in retrieved_1
        )
        should_prefetch = True
        if (
            routed is not None
            and routed.get("intent") == "web_latest"
            and self.config.tool_prefetch_requires_low_memory_confidence
            and has_high_confidence_memory
            and has_memory_fact_context
        ):
            should_prefetch = False
        if routed is not None and should_prefetch:
            prefetched = self._execute_tool(routed["tool"], routed["argument"])
            messages.append({
                "role": "assistant",
                "content": f"[TOOL_CALL: {routed['tool']} | {routed['argument']}]",
            })
            messages.append({
                "role": "user",
                "content": f"{prefetched}\nUse this fresh tool result to answer the original user request.",
            })

        # --- Streaming tool-call loop ----------------------------------
        # Buffer chunks until we can determine whether the response is a
        # [TOOL_CALL:...] (silent) or regular text (streamed to caller).
        # We use `accumulated` to hold the full text and only yield once
        # we know it is not a tool call.
        full_response = ""
        _TOOL_PREFIX = "[TOOL_CALL:"
        for _call_idx in range(self.config.max_tool_calls_per_turn + 1):
            accumulated = ""
            decision_made = False
            is_tool_call = False

            for chunk in self.llm.stream_chat(
                messages,
                max_new_tokens=self.config.max_new_tokens_response,
                temperature=self.config.temperature_response,
            ):
                accumulated += chunk
                if not decision_made:
                    stripped = accumulated.lstrip()
                    if stripped.startswith(_TOOL_PREFIX):
                        is_tool_call = True
                        decision_made = True
                    elif len(stripped) > len(_TOOL_PREFIX):
                        # Long enough to be sure it's not a tool call.
                        is_tool_call = False
                        decision_made = True
                        yield accumulated
                else:
                    if not is_tool_call:
                        yield chunk

            # Handle very short responses where decision was never triggered.
            if not decision_made:
                if self._parse_tool_call(accumulated) is not None:
                    is_tool_call = True
                else:
                    is_tool_call = False
                    if accumulated:
                        yield accumulated

            full_response = accumulated
            parsed = self._parse_tool_call(full_response)
            if parsed is None:
                break
            tool_name, arg = parsed
            tool_result = self._execute_tool(tool_name, arg)
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": tool_result})
            full_response = ""

        # --- Autonomous learning fallback (silent, then stream) --------
        already_learned = any(
            "learn_skill" in m.get("content", "")
            for m in messages
            if m.get("role") == "assistant"
        )
        if (
            self.config.autonomous_learning
            and self._is_uncertain(full_response)
            and not already_learned
        ):
            logging.info("Uncertainty detected - auto-triggering learn_skill for: %s", user_input[:80])
            learn_result = self._execute_tool("learn_skill", user_input[:200])
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": f"{learn_result}\nNow answer the original question using this newly learned knowledge.",
            })
            full_response = ""
            for chunk in self.llm.stream_chat(
                messages,
                max_new_tokens=self.config.max_new_tokens_response,
                temperature=self.config.temperature_response,
            ):
                full_response += chunk
                yield chunk

        # --- Bookkeeping (never yielded) -------------------------------
        self._finalize_turn(user_input, full_response)

