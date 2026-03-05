import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# Keyword pre-filter stopwords for hop-1 retrieval (D-119 per-hop specialization)
_KEYWORD_STOPWORDS = frozenset(
    "what does or the a an is are was were do did have has how which where who like own of in at to for and by".split()
)


class AdaptiveModularMemory:
    """
    Adaptive Modular Memory (AMM) for the AGI Agent.

    Thread-safe:
    - A single lock protects in-memory state.
    - Retrieval uses snapshots.
    - delete_matching is conflict-safe (no snapshot rebuild race).

    Persistence:
    - JSON payload + .pt sidecar with snapshot_version markers.
    - Atomic temp-file swap on flush.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_slots: int = 10_000,
        save_path: str = "memory.json",
        decay_enabled: bool = True,
        decay_half_lives: Optional[Dict] = None,
        dedup_enabled: bool = True,
        dedup_scope: str = "exact_text",
        dedup_types: Optional[Set[str]] = None,
        keyword_prefilter_min_pool: int = 100,
        keyword_prefilter_factor: int = 20,
    ):
        self.encoder = SentenceTransformer(model_name)
        self.d_key: int = self.encoder.get_sentence_embedding_dimension()
        self.max_slots = max_slots
        self.save_path = save_path
        self._keys_path = save_path + ".pt"
        self._decay_enabled = decay_enabled
        self._decay_half_lives: Dict = decay_half_lives or {
            "identity": None,
            "constraint": None,
            "preference": 5_184_000.0,
            "skill": 2_592_000.0,
            "fact": 1_209_600.0,
            "stored_fact": 1_209_600.0,
            "document": 1_209_600.0,
            "agent_response": 259_200.0,
            "user_input": 259_200.0,
            "default": 604_800.0,
        }

        self._dedup_enabled = dedup_enabled
        self._dedup_scope = (dedup_scope or "off").strip().lower()
        self._dedup_types = set(dedup_types or {"fact", "identity", "skill", "skill_ref", "document"})
        self._dedup_counts: Dict[Tuple[str, str], int] = {}

        self._keyword_prefilter_min_pool = keyword_prefilter_min_pool
        self._keyword_prefilter_factor = keyword_prefilter_factor

        self._lock = threading.Lock()
        self._dirty = False
        self._version = 0

        # Retrieval cache: stacking all key vectors every retrieve() call is
        # costly at 10k slots. Cache the stacked tensor and invalidate it on
        # any mutation (version bump).
        self._keys_tensor_cache: Optional[torch.Tensor] = None
        self._keys_tensor_cache_version: int = -1

        self._keys: deque = deque(maxlen=max_slots)
        self._values: deque = deque(maxlen=max_slots)
        self._metadata: deque = deque(maxlen=max_slots)

        self._load_memory()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._values)

    def _decay_multiplier(self, meta: Dict, now: float) -> float:
        if not self._decay_enabled:
            return 1.0
        mtype = (meta or {}).get("type", "")
        half_life = self._decay_half_lives.get(mtype, self._decay_half_lives.get("default", 604_800.0))
        if half_life is None:
            return 1.0
        ts = meta.get("timestamp")
        if ts is None:
            return 1.0
        age_secs = max(0.0, now - float(ts))
        return 0.5 ** (age_secs / half_life)

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @staticmethod
    def _keyword_score(query_tokens: List[str], text: str) -> int:
        """
        Score stored text by keyword overlap with query tokens.
        Returns count of query tokens that appear as substrings in the text.
        """
        text_lower = (text or "").lower()
        count = 0
        for token in query_tokens:
            if token in text_lower:
                count += 1
        return count

    def _dedup_key(self, text: str, meta: Dict) -> Optional[Tuple[str, str]]:
        if not self._dedup_enabled:
            return None
        if self._dedup_scope == "off":
            return None
        mtype = (meta or {}).get("type", "")
        if mtype not in self._dedup_types:
            return None
        if self._dedup_scope == "normalized_text":
            return (mtype, self._normalize_text(text))
        return (mtype, (text or "").strip())

    def _dedup_inc(self, key: Optional[Tuple[str, str]]) -> None:
        if key is None:
            return
        self._dedup_counts[key] = self._dedup_counts.get(key, 0) + 1

    def _dedup_dec(self, key: Optional[Tuple[str, str]]) -> None:
        if key is None:
            return
        prev = self._dedup_counts.get(key, 0)
        if prev <= 1:
            self._dedup_counts.pop(key, None)
            return
        self._dedup_counts[key] = prev - 1

    def _has_duplicate(self, key: Optional[Tuple[str, str]]) -> bool:
        if key is None:
            return False
        return self._dedup_counts.get(key, 0) > 0

    def _evict_left_if_needed(self) -> None:
        if len(self._values) < self.max_slots:
            return
        evicted_value = self._values.popleft()
        evicted_meta = self._metadata.popleft()
        self._keys.popleft()
        self._dedup_dec(self._dedup_key(evicted_value, evicted_meta))

    def _rebuild_dedup_counts(self) -> None:
        self._dedup_counts.clear()
        for value, meta in zip(self._values, self._metadata):
            self._dedup_inc(self._dedup_key(value, meta or {}))

    def add_memory(self, text: str, meta: Optional[Dict] = None) -> None:
        meta_obj = dict(meta or {})
        key = self._dedup_key(text, meta_obj)

        with self._lock:
            if self._has_duplicate(key):
                return

        with torch.no_grad():
            encoded = self.encoder.encode(text, convert_to_tensor=True).cpu()

        with self._lock:
            if self._has_duplicate(key):
                return
            self._evict_left_if_needed()
            self._keys.append(encoded)
            self._values.append(text)
            self._metadata.append(meta_obj)
            self._dedup_inc(key)
            self._dirty = True
            self._version += 1
            self._keys_tensor_cache = None
            self._keys_tensor_cache_version = -1

    def upsert_by_meta(
        self,
        text: str,
        meta: Dict,
        match_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Insert or replace an entry by metadata identity keys.
        Example: match_keys=['type', 'skill_id'] for skill pointer upserts.
        """
        meta_obj = dict(meta or {})
        keys = list(match_keys or [])
        key = self._dedup_key(text, meta_obj)

        with torch.no_grad():
            encoded = self.encoder.encode(text, convert_to_tensor=True).cpu()

        with self._lock:
            idx = None
            if keys:
                for pos, existing_meta in enumerate(self._metadata):
                    if all((existing_meta or {}).get(k) == meta_obj.get(k) for k in keys):
                        idx = pos
                        break

            if idx is None:
                if self._has_duplicate(key):
                    return
                self._evict_left_if_needed()
                self._keys.append(encoded)
                self._values.append(text)
                self._metadata.append(meta_obj)
                self._dedup_inc(key)
                self._dirty = True
                self._version += 1
                self._keys_tensor_cache = None
                self._keys_tensor_cache_version = -1
                return

            old_text = self._values[idx]
            old_meta = self._metadata[idx] or {}
            old_key = self._dedup_key(old_text, old_meta)
            if key != old_key and self._has_duplicate(key):
                return

            self._dedup_dec(old_key)
            self._keys[idx] = encoded
            self._values[idx] = text
            self._metadata[idx] = meta_obj
            self._dedup_inc(key)
            self._dirty = True
            self._version += 1
            self._keys_tensor_cache = None
            self._keys_tensor_cache_version = -1

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
        include_types: Optional[Set[str]] = None,
        exclude_types: Optional[Set[str]] = None,
        subject: Optional[str] = None,
        keyword_prefilter: bool = False,
    ) -> List[Tuple[str, Dict, float]]:
        keys_tensor = None
        keys_snap = None
        version = -1
        with self._lock:
            if not self._values:
                return []
            vals_snap = list(self._values)
            meta_snap = list(self._metadata)
            version = self._version
            if (
                self._keys_tensor_cache is not None
                and self._keys_tensor_cache_version == version
                and self._keys_tensor_cache.shape[0] == len(self._keys)
            ):
                keys_tensor = self._keys_tensor_cache
            else:
                # Snapshot keys for this retrieval to keep data aligned with
                # vals/meta even if another thread mutates after we release lock.
                keys_snap = list(self._keys)

        pool_size = len(vals_snap)
        candidate_indices = None

        # Keyword prefilter (D-119 per-hop specialization) — narrow embedding candidate pool
        # for large memory pools using keyword overlap scoring.
        if keyword_prefilter and pool_size > self._keyword_prefilter_min_pool:
            # Tokenize query and remove stopwords
            query_tokens = [
                t for t in re.findall(r"[a-z0-9_]+", query.lower())
                if t not in _KEYWORD_STOPWORDS
            ]

            if query_tokens:
                # Score each stored text by keyword hit count
                kw_scores = [
                    self._keyword_score(query_tokens, vals_snap[i])
                    for i in range(pool_size)
                ]

                # If any keyword matches exist, narrow the candidate pool
                if max(kw_scores) > 0:
                    # Compute candidate pool size: max(min_pool, top_k * factor)
                    min_candidates = max(self._keyword_prefilter_min_pool, top_k * self._keyword_prefilter_factor)
                    num_candidates = min(pool_size, min_candidates)

                    # Select top-scoring indices by keyword overlap
                    indexed_scores = [(i, score) for i, score in enumerate(kw_scores)]
                    indexed_scores.sort(key=lambda x: x[1], reverse=True)
                    candidate_indices = [i for i, _ in indexed_scores[:num_candidates]]

        with torch.no_grad():
            query_key = self.encoder.encode(query, convert_to_tensor=True).cpu()
            if keys_tensor is None:
                keys_tensor = torch.stack(keys_snap) if keys_snap else torch.zeros((0, self.d_key), dtype=torch.float32)
                # Populate cache opportunistically if memory hasn't changed.
                # (Full-pool cache is NOT updated when prefilter is active)
                if candidate_indices is None:
                    with self._lock:
                        if (
                            self._version == version
                            and (self._keys_tensor_cache is None or self._keys_tensor_cache_version != version)
                            and len(self._keys) == keys_tensor.shape[0]
                        ):
                            self._keys_tensor_cache = keys_tensor
                            self._keys_tensor_cache_version = version

            # If prefilter is active and produced candidates, build candidate tensor
            if candidate_indices is not None:
                candidate_tensor = torch.stack([keys_snap[i] for i in candidate_indices])
                similarities = F.cosine_similarity(query_key.unsqueeze(0), candidate_tensor)
            else:
                similarities = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor)

        now = time.time()
        candidates: List[Tuple[str, Dict, float]] = []

        if candidate_indices is not None:
            # Map local similarity index back to snapshot index
            for local_idx, raw_score in enumerate(similarities.tolist()):
                if raw_score < threshold:
                    continue
                snap_idx = candidate_indices[local_idx]
                meta = meta_snap[snap_idx] or {}
                mtype = meta.get("type")
                msubject = meta.get("subject")
                if include_types is not None and mtype not in include_types:
                    continue
                if exclude_types is not None and mtype in exclude_types:
                    continue
                if subject is not None and msubject != subject:
                    continue
                decayed = float(raw_score) * self._decay_multiplier(meta, now)
                meta_out = dict(meta)
                meta_out["_raw_cosine"] = float(raw_score)
                candidates.append((vals_snap[snap_idx], meta_out, decayed))
        else:
            # Full-pool retrieval (no prefilter)
            for idx, raw_score in enumerate(similarities.tolist()):
                if raw_score < threshold:
                    continue
                meta = meta_snap[idx] or {}
                mtype = meta.get("type")
                msubject = meta.get("subject")
                if include_types is not None and mtype not in include_types:
                    continue
                if exclude_types is not None and mtype in exclude_types:
                    continue
                if subject is not None and msubject != subject:
                    continue
                decayed = float(raw_score) * self._decay_multiplier(meta, now)
                # Annotate with raw cosine for confidence gate (D-216, D-223).
                # Copy meta to avoid mutating stored metadata.
                meta_out = dict(meta)
                meta_out["_raw_cosine"] = float(raw_score)
                candidates.append((vals_snap[idx], meta_out, decayed))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[: max(0, top_k)]

    def retrieve_by_type(
        self,
        types: Set[str],
        max_results: int = 50,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Retrieve all memories of the given types without embedding similarity.
        Used for constraint/preference retrieval where embedding similarity
        fails (L-214: 65% ceiling on constraint phrasing).
        Returns (text, metadata, 1.0) tuples — score is always 1.0 since
        type match is binary.
        """
        with self._lock:
            if not self._values:
                return []
            vals_snap = list(self._values)
            meta_snap = list(self._metadata)

        results: List[Tuple[str, Dict, float]] = []
        for idx, meta in enumerate(meta_snap):
            mtype = (meta or {}).get("type", "")
            if mtype in types:
                meta_out = dict(meta or {})
                meta_out["_raw_cosine"] = 1.0
                results.append((vals_snap[idx], meta_out, 1.0))
                if len(results) >= max_results:
                    break
        return results

    def sample_memories(self, k: int) -> List[str]:
        with self._lock:
            vals = list(self._values)
        if not vals:
            return []
        return random.sample(vals, min(k, len(vals)))

    def sample_memories_weighted(
        self,
        k: int,
        preferred_types: Optional[Set[str]] = None,
        decay_risk_bias: float = 2.0,
    ) -> List[str]:
        with self._lock:
            vals = list(self._values)
            metas = list(self._metadata)
        if not vals:
            return []

        preferred = set(preferred_types or set())
        now = time.time()
        weights: List[float] = []
        for meta in metas:
            m = meta or {}
            mtype = m.get("type", "")
            half_life = self._decay_half_lives.get(mtype, self._decay_half_lives.get("default", 604_800.0))
            ts = m.get("timestamp")
            if half_life is None or ts is None:
                decay_risk = 0.1
            else:
                age = max(0.0, now - float(ts))
                decay_risk = 1.0 - (0.5 ** (age / half_life))
            w = 0.1 + (decay_risk_bias * decay_risk)
            if mtype in preferred:
                w *= 1.75
            if mtype in {"user_input", "agent_response"}:
                w *= 0.5
            weights.append(max(0.0001, w))

        selected: List[str] = []
        pool_idx = list(range(len(vals)))
        pool_weights = list(weights)
        limit = min(k, len(vals))
        for _ in range(limit):
            picked = random.choices(pool_idx, weights=pool_weights, k=1)[0]
            selected.append(vals[picked])
            remove_at = pool_idx.index(picked)
            pool_idx.pop(remove_at)
            pool_weights.pop(remove_at)
            if not pool_idx:
                break
        return selected

    def delete_matching(
        self,
        query: str,
        threshold: float = 0.75,
        max_delete: int = 10,
    ) -> int:
        with torch.no_grad():
            query_key = self.encoder.encode(query, convert_to_tensor=True).cpu()

        with self._lock:
            if not self._values:
                return 0

            keys_snap = list(self._keys)
            vals_snap = list(self._values)
            meta_snap = list(self._metadata)

            keys_tensor = torch.stack(keys_snap)
            similarities = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor).tolist()

            delete_candidates = [i for i, score in enumerate(similarities) if score >= threshold]
            if not delete_candidates:
                return 0

            if len(delete_candidates) > max_delete:
                scored = sorted(
                    ((i, similarities[i]) for i in delete_candidates),
                    key=lambda x: x[1],
                    reverse=True,
                )
                delete_set = {i for i, _ in scored[:max_delete]}
            else:
                delete_set = set(delete_candidates)

            keep_indices = [i for i in range(len(vals_snap)) if i not in delete_set]
            deleted = len(vals_snap) - len(keep_indices)
            if deleted <= 0:
                return 0

            self._keys = deque((keys_snap[i] for i in keep_indices), maxlen=self.max_slots)
            self._values = deque((vals_snap[i] for i in keep_indices), maxlen=self.max_slots)
            self._metadata = deque((meta_snap[i] for i in keep_indices), maxlen=self.max_slots)
            self._rebuild_dedup_counts()
            self._dirty = True
            self._version += 1
            self._keys_tensor_cache = None
            self._keys_tensor_cache_version = -1
            return deleted

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            vals = list(self._values)
            meta = list(self._metadata)
            keys_snap = list(self._keys)
            snapshot_version = self._version

        keys_tensor = (
            torch.stack(keys_snap)
            if keys_snap
            else torch.zeros((0, self.d_key), dtype=torch.float32)
        )
        json_payload = {
            "values": vals,
            "metadata": meta,
            "snapshot_version": snapshot_version,
        }
        pt_payload = {
            "keys": keys_tensor,
            "snapshot_version": snapshot_version,
        }

        json_tmp = self.save_path + ".tmp"
        pt_tmp = self._keys_path + ".tmp"
        try:
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(json_payload, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as fsync_err:
                    logging.debug(f"json fsync skipped: {fsync_err}")

            torch.save(pt_payload, pt_tmp)
            with open(pt_tmp, "rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError as fsync_err:
                    logging.debug(f"pt fsync skipped: {fsync_err}")

            os.replace(pt_tmp, self._keys_path)
            os.replace(json_tmp, self.save_path)

            with self._lock:
                if self._version == snapshot_version:
                    self._dirty = False
        except Exception as e:
            logging.error(f"Memory flush error: {e}")
            with self._lock:
                self._dirty = True
            for tmp in (json_tmp, pt_tmp):
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

    def _load_memory(self) -> None:
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vals: List[str] = data.get("values", [])
            meta: List[dict] = data.get("metadata", [])
            json_snapshot = data.get("snapshot_version")
            if not vals:
                return

            keys_tensor = None
            if os.path.exists(self._keys_path):
                loaded = torch.load(self._keys_path, weights_only=True)
                if isinstance(loaded, dict):
                    pt_snapshot = loaded.get("snapshot_version")
                    keys_tensor = loaded.get("keys")
                    if json_snapshot is not None and pt_snapshot is not None and json_snapshot != pt_snapshot:
                        keys_tensor = None
                elif torch.is_tensor(loaded):
                    keys_tensor = loaded

            if (
                keys_tensor is None
                or not torch.is_tensor(keys_tensor)
                or keys_tensor.shape[0] != len(vals)
            ):
                logging.info(f"Re-encoding {len(vals)} memories (load reconciliation).")
                with torch.no_grad():
                    keys_tensor = self.encoder.encode(vals, convert_to_tensor=True).cpu()

            for idx, value in enumerate(vals):
                m = meta[idx] if idx < len(meta) else {}
                self._keys.append(keys_tensor[idx])
                self._values.append(value)
                self._metadata.append(m)
                self._dedup_inc(self._dedup_key(value, m))

            self._dirty = False
            self._version = len(self._values)
        except Exception as e:
            logging.error(f"Failed to load memory: {e}. Starting fresh.")
