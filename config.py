from dataclasses import dataclass, field


_VALID_HOP2_STRATEGIES = {"llm", "pattern", "hybrid"}
_VALID_TOOL_BACKENDS = {"pattern", "hf_zero_shot"}
_VALID_MEMORY_DEDUP_SCOPES = {"exact_text", "normalized_text", "off"}


@dataclass
class AgentConfig:
    """Central configuration for the AGI agent and all sub-components."""

    # Agent identity
    name: str = "Nexus"

    # LLM backbone
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    repetition_penalty: float = 1.25
    llm_context_fallback_tokens: int = 8192

    # Memory
    memory_encoder: str = "all-MiniLM-L6-v2"
    max_memory_slots: int = 10_000
    retrieval_threshold: float = 0.3
    retrieval_top_k: int = 5
    retrieval_top_k_min: int = 3
    retrieval_top_k_max: int = 15
    retrieval_confidence_threshold: float = 0.25
    retrieval_keyword_prefilter_enabled: bool = True
    retrieval_keyword_prefilter_min_pool: int = 100
    retrieval_keyword_prefilter_factor: int = 20
    memory_dedup_enabled: bool = True

    # Confidence gate (D-216, D-223, D-227)
    # 3-feature LR trained on real AMM episodes: 0.9% hallucination at 100% coverage.
    # Features: raw_cos_max, raw_cos_margin, entropy_norm
    confidence_gate_enabled: bool = True
    confidence_gate_coefficients: list[float] = field(default_factory=lambda: [
        12.0,   # raw_cos_max — dominant signal (Cohen's d=3.18)
        3.0,    # raw_cos_margin — top1-top2 gap
        -1.5,   # entropy_norm — high entropy = less certain
    ])
    confidence_gate_intercept: float = -4.0
    confidence_gate_inject_threshold: float = 0.7
    confidence_gate_hedge_threshold: float = 0.3
    memory_dedup_scope: str = "exact_text"
    memory_dedup_types: set[str] = field(default_factory=lambda: {
        "fact",
        "identity",
        "constraint",
        "preference",
        "skill",
        "skill_ref",
        "document",
    })

    # Temporal decay
    memory_decay_enabled: bool = True
    memory_decay_half_lives: dict = field(default_factory=lambda: {
        "identity": None,
        "constraint": None,        # Constraints never decay (D-228)
        "preference": 5_184_000.0,  # Preferences: 60-day half-life
        "skill": 2_592_000.0,
        "fact": 1_209_600.0,
        "stored_fact": 1_209_600.0,
        "document": 1_209_600.0,
        "agent_response": 259_200.0,
        "user_input": 259_200.0,
        "default": 604_800.0,
    })

    # Background thinking
    think_interval_secs: float = 10.0
    idle_threshold_secs: float = 30.0
    reflection_cooldown_secs: float = 60.0
    reflection_sample_size: int = 3
    reflection_priority_types: set[str] = field(default_factory=lambda: {"fact", "skill", "document"})
    reflection_decay_risk_bias: float = 2.0

    # LLM generation
    max_new_tokens_extraction: int = 50
    max_new_tokens_reflection: int = 150
    max_new_tokens_response: int = 200
    temperature_extraction: float = 0.1
    temperature_response: float = 0.7

    # Multi-hop retrieval
    retrieval_hop2_enabled: bool = True
    retrieval_hop2_strategy: str = "hybrid"
    retrieval_hop2_pattern_max_entities: int = 4
    retrieval_hop2_pattern_top_k_per_query: int = 2
    retrieval_hop2_pattern_threshold_scale: float = 1.0
    hop2_bridge_patterns: list[str] = field(default_factory=lambda: [
        r"\b(?:is|was|are|were)\s+(?:led|managed|owned|supervised|coordinated|run|handled)\s+by\s+(?P<entity>[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})",
        r"\b(?:leader|manager|owner|supervisor|coordinator)\s+is\s+(?P<entity>[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})",
        r"\b(?P<entity>[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})\s+(?:is|was)\s+(?:based|located)\s+in\b",
        r"\bby\s+(?P<entity>[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})\b",
    ])
    hop2_location_intent_patterns: list[str] = field(default_factory=lambda: [
        r"\b(where|which city|what city|based in|located in|work from|works from|work in|works out of|work out of|located)\b",
    ])
    hop2_query_templates_by_intent: dict = field(default_factory=lambda: {
        "location": [
            "{entity} based in",
            "{entity} located in",
            "{entity} works from",
            "{entity} works out of",
            "{entity} work out of",
            "{entity} works in",
            "{entity} live in",
            "{entity} location",
        ],
        "default": [
            "{entity}",
            "{entity} details",
            "{entity} profile",
        ],
    })

    # Tools
    max_tool_calls_per_turn: int = 3
    autonomous_learning: bool = True
    tool_routing_rules: list = field(default_factory=lambda: [
        {
            "intent": "time_now",
            "pattern": r"\b(what\s+time(?:\s+is\s+it)?(?:\s+right\s+now)?|what(?:'s|\s+is)\s+(?:the\s+)?time|current\s+time|time\s+(?:right\s+)?now|date\s+today|today(?:'s)?\s+date)\b",
            "tool": "datetime_now",
            "argument": "",
        },
        {
            "intent": "web_latest",
            "pattern": r"\b(search(?:\s+the)?\s+web|look\s+up|find\s+online|latest|headlines|news|most\s+recent)\b",
            "tool": "web_search",
            "argument": "{user_input}",
        },
        {
            # Fire when the user's message contains an explicit URL or a
            # recognisable domain (e.g. "compsmart.cloud", "www.example.com").
            "intent": "web_fetch_url",
            "pattern": r"(?:https?://\S+|(?<!\w)(?:www\.)?[\w-]+\.(?:com|net|org|io|cloud|dev|ai|co|app|site|uk|tech|edu|gov|info|me|us)(?:/[^\s]*)?(?!\w))",
            "tool": "web_fetch",
            "argument": "{user_input}",
        },
        {
            # Fire on browse/visit/check intent even when no URL is visible yet.
            "intent": "web_browse_intent",
            "pattern": r"\b(visit|check|open|browse|go\s+to|look\s+at|can\s+you\s+see|pull\s+up)\s+(?:my\s+)?(?:website|site|page|url|link)\b",
            "tool": "web_fetch",
            "argument": "{user_input}",
        },
    ])
    tool_routing_backend: str = "pattern"
    tool_routing_hf_model: str = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
    tool_routing_hf_candidate_labels: list = field(default_factory=lambda: [
        "time query",
        "latest web search",
    ])
    tool_routing_hf_label_to_intent: dict = field(default_factory=lambda: {
        "time query": "time_now",
        "latest web search": "web_latest",
    })
    tool_routing_hf_min_score: float = 0.50
    tool_routing_hf_hypothesis_template: str = "This request is about {}."
    tool_prefetch_requires_low_memory_confidence: bool = True
    tool_prefetch_memory_score_threshold: float = 0.35

    # Conversation
    max_history_turns: int = 10
    short_followup_max_words: int = 7

    # Persistence
    flush_interval_secs: float = 60.0

    # Pattern policies
    # Only unambiguous name-introduction patterns.  Ambiguous ones
    # ("i am X", "i'm X") were removed — the agent now asks for the
    # user's name if it doesn't know it, then catches the reply here.
    name_extract_patterns: list = field(default_factory=lambda: [
        r"\bmy name is\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z0-9'\- ]{0,40})",
    ])
    name_capture_stop_pattern: str = r"\b(and|but|because|that|who|where|when)\b|[,.;:!?]"
    blocked_user_names: set[str] = field(default_factory=lambda: {
        "your creator",
        "creator",
        "nexus",
        "assistant",
        "ai",
    })

    personal_fact_negation_patterns: list[str] = field(default_factory=lambda: [
        r"\b(?:don't|do not|didn't|did not|never|not)\b",
        r"\b(?:is|are|was|were|am|have|has|had)\s+not\b",
        r"\bno\s+longer\b",
    ])

    # Constraint / preference extraction (D-228, L-214, L-216)
    # Embedding retrieval has a 65% ceiling on constraints — typed extraction fixes this.
    constraint_extract_patterns: list[str] = field(default_factory=lambda: [
        r"\b(?:never|always|don'?t|do not|must not|must always|should never|should always|make sure to|be sure to)\s+(?P<rule>.{5,80})",
        r"\b(?:remember to|don'?t forget to|keep in mind)\s+(?P<rule>.{5,80})",
        r"\b(?:i(?:'m| am) (?:allergic|intolerant|sensitive) to)\s+(?P<rule>.{3,60})",
        r"\b(?:(?:only|exclusively) (?:use|eat|drink|buy|accept|allow))\s+(?P<rule>.{3,60})",
    ])
    preference_extract_patterns: list[str] = field(default_factory=lambda: [
        r"\bi (?:prefer|like|love|enjoy|hate|dislike|avoid|can'?t stand)\s+(?P<pref>.{3,80})",
        r"\b(?:my (?:favorite|favourite|preferred|go-to))\s+(?P<pref>.{3,60})",
    ])

    correction_cue_patterns: list = field(default_factory=lambda: [
        r"\b(actually|correction|i meant|rather|instead|update|not anymore|now)\b",
        r"\b(that(?:'s| is)\s+wrong)\b",
    ])
    fact_forget_rules: list = field(default_factory=lambda: [
        {
            "pattern": r"user personal fact:\s*my\s+([a-z][a-z0-9\s]{0,25}?)\s+(?:is|are|was|were|has been)\s+",
            "template": "User personal fact: my {g1}",
        },
        {
            "pattern": r"user personal fact:\s*i\s+(live|grew up|was born|work|study)\s+(?:in|at|near)\s+",
            "template": "User personal fact: i {g1}",
        },
        {
            "pattern": r"user personal fact:\s*i\s+am\s+\d{1,3}\s+years?\s+old",
            "template": "User personal fact: i am",
        },
        {
            "pattern": r"user personal fact:\s*i\s+have\s+a?\s*([a-z][a-z0-9\s]{0,25})\s+(?:called|named)\s+",
            "template": "User personal fact: i have {g1}",
        },
    ])
    correction_forget_threshold: float = 0.60
    correction_forget_max_delete: int = 4

    # Skill store
    skills_root_dir: str = "skills"
    skills_drafts_dir: str = "skills/drafts"
    skills_published_dir: str = "skills/published"
    skills_max_hydrated_per_turn: int = 2
    skills_hydration_max_chars: int = 3000
    skills_allow_draft_autonomous_use: bool = False
    skills_require_env_placeholders: bool = True
    memory_skill_pointer_type: str = "skill_ref"

    def __post_init__(self) -> None:
        strategy = (self.retrieval_hop2_strategy or "").strip().lower()
        if strategy not in _VALID_HOP2_STRATEGIES:
            raise ValueError(
                f"Invalid retrieval_hop2_strategy '{self.retrieval_hop2_strategy}'. "
                f"Expected one of {sorted(_VALID_HOP2_STRATEGIES)}."
            )
        self.retrieval_hop2_strategy = strategy

        backend = (self.tool_routing_backend or "").strip().lower()
        if backend not in _VALID_TOOL_BACKENDS:
            raise ValueError(
                f"Invalid tool_routing_backend '{self.tool_routing_backend}'. "
                f"Expected one of {sorted(_VALID_TOOL_BACKENDS)}."
            )
        self.tool_routing_backend = backend

        dedup_scope = (self.memory_dedup_scope or "").strip().lower()
        if dedup_scope not in _VALID_MEMORY_DEDUP_SCOPES:
            raise ValueError(
                f"Invalid memory_dedup_scope '{self.memory_dedup_scope}'. "
                f"Expected one of {sorted(_VALID_MEMORY_DEDUP_SCOPES)}."
            )
        self.memory_dedup_scope = dedup_scope

        if self.retrieval_hop2_pattern_max_entities < 1:
            raise ValueError("retrieval_hop2_pattern_max_entities must be >= 1.")
        if self.retrieval_hop2_pattern_top_k_per_query < 1:
            raise ValueError("retrieval_hop2_pattern_top_k_per_query must be >= 1.")
        if not (0.0 <= self.retrieval_hop2_pattern_threshold_scale <= 2.0):
            raise ValueError("retrieval_hop2_pattern_threshold_scale must be in [0.0, 2.0].")
        if self.short_followup_max_words < 1:
            raise ValueError("short_followup_max_words must be >= 1.")
        if not (0.0 <= self.tool_prefetch_memory_score_threshold <= 1.0):
            raise ValueError("tool_prefetch_memory_score_threshold must be in [0.0, 1.0].")
        if self.llm_context_fallback_tokens < 512:
            raise ValueError("llm_context_fallback_tokens must be >= 512.")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0.")
        if self.skills_max_hydrated_per_turn < 0:
            raise ValueError("skills_max_hydrated_per_turn must be >= 0.")
        if self.skills_hydration_max_chars < 256:
            raise ValueError("skills_hydration_max_chars must be >= 256.")
        if not self.memory_skill_pointer_type.strip():
            raise ValueError("memory_skill_pointer_type must be non-empty.")
