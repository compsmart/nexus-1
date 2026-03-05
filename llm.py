import logging
import threading
from typing import Dict, Iterator, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# Below this temperature the output is near-deterministic; use greedy decoding
# to avoid stochastic noise.
_GREEDY_TEMP_THRESHOLD = 0.15

# Some tokenizers report huge sentinel values; treat them as "unknown".
_MODEL_MAX_LENGTH_SENTINEL = 10_000_000


class LLMEngine:
    """
    Shared LLM wrapper for both the agent and benchmark baselines.

    Features:
    - model-aware context window resolution
    - repetition penalty defaults
    - process-wide model/tokenizer cache for identical load configs
    - graceful fallback when 4-bit load fails on constrained GPU memory
    """

    _CACHE_LOCK = threading.Lock()
    _MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[AutoTokenizer, AutoModelForCausalLM, int, str]] = {}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_4bit: bool = False,
        repetition_penalty: float = 1.1,
        context_fallback_tokens: int = 8192,
        shared_cache: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self._context_fallback_tokens = int(context_fallback_tokens)

        cache_key = (model_name, device, bool(use_4bit))
        if shared_cache:
            with self._CACHE_LOCK:
                cached = self._MODEL_CACHE.get(cache_key)
            if cached is not None:
                self.tokenizer, self.model, self._max_context_tokens, loaded_device = cached
                self.device = loaded_device
                logging.info(
                    "Reusing cached LLM %s on %s (4-bit=%s).",
                    model_name,
                    loaded_device,
                    use_4bit,
                )
                return

        tokenizer, model, loaded_device = self._load_model(model_name, device, use_4bit)
        self.tokenizer = tokenizer
        self.model = model
        self.device = loaded_device
        self._max_context_tokens = self._resolve_context_limit(
            tokenizer,
            model,
            self._context_fallback_tokens,
        )

        logging.info(
            "LLM loaded: model=%s device=%s context=%s 4-bit=%s",
            model_name,
            self.device,
            self._max_context_tokens,
            bool(use_4bit),
        )

        if shared_cache:
            with self._CACHE_LOCK:
                self._MODEL_CACHE[cache_key] = (
                    self.tokenizer,
                    self.model,
                    self._max_context_tokens,
                    self.device,
                )

    def _load_model(
        self,
        model_name: str,
        device: str,
        use_4bit: bool,
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
        logging.info("Loading LLM %s on %s (4-bit=%s)...", model_name, device, use_4bit)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_4bit and device == "cuda":
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                model.eval()
                return tokenizer, model, "cuda"
            except Exception as e:
                logging.warning(
                    "4-bit load failed for %s on CUDA (%s). Falling back to non-quantized load.",
                    model_name,
                    e,
                )

        # Primary fallback: regular load on requested device.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                model.to(device)
            model.eval()
            return tokenizer, model, device
        except Exception as e:
            if device != "cuda":
                raise
            logging.warning(
                "CUDA model load failed for %s (%s). Falling back to CPU.",
                model_name,
                e,
            )

        # Final fallback: CPU load.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.to("cpu")
        model.eval()
        return tokenizer, model, "cpu"

    def _resolve_context_limit(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        fallback: int,
    ) -> int:
        candidates = []

        tok_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < _MODEL_MAX_LENGTH_SENTINEL:
            candidates.append(tok_max)

        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("max_position_embeddings", "n_positions", "seq_length"):
                value = getattr(cfg, attr, None)
                if isinstance(value, int) and value > 0:
                    candidates.append(value)

        if candidates:
            return max(candidates)
        return max(512, int(fallback))

    def _should_sample(self, temperature: float) -> bool:
        return temperature > _GREEDY_TEMP_THRESHOLD

    def _build_generate_kwargs(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> dict:
        available = self._max_context_tokens - max_new_tokens
        if available <= 0:
            available = max(256, self._max_context_tokens // 2)
        if input_ids.shape[1] > available:
            logging.warning("Prompt truncated: %s -> %s tokens.", input_ids.shape[1], available)
            input_ids = input_ids[:, -available:]

        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": self._should_sample(temperature),
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = temperature
        return kwargs

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            kwargs = self._build_generate_kwargs(
                inputs["input_ids"],
                max_new_tokens,
                temperature,
            )
            prompt_len = kwargs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(**kwargs)
            return self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            logging.error("LLM generate error: %s", e)
            return ""

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            kwargs = self._build_generate_kwargs(
                inputs["input_ids"],
                max_new_tokens,
                temperature,
            )
            prompt_len = kwargs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(**kwargs)
            return self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            logging.error("LLM chat error: %s", e)
            return ""

    def stream_chat(
        self,
        messages: list,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Streaming variant of chat(). Yields decoded string chunks as they
        are generated. Falls back to yielding the full response as a single
        chunk if the model or tokenizer does not support streaming."""
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            kwargs = self._build_generate_kwargs(
                inputs["input_ids"],
                max_new_tokens,
                temperature,
            )
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            kwargs["streamer"] = streamer
            # Generate on a daemon thread so we can yield from the main thread.
            gen_thread = threading.Thread(
                target=self.model.generate,
                kwargs=kwargs,
                daemon=True,
            )
            gen_thread.start()
            for chunk in streamer:
                if chunk:
                    yield chunk
            gen_thread.join()
        except Exception as e:
            logging.error("LLM stream_chat error: %s", e)
            # Graceful fallback: yield the full response as one chunk.
            fallback = self.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)
            if fallback:
                yield fallback
