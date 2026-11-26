"""LLM backends: Google Gemini (cloud) and Ollama (local, private). Same interface.

Gemini: retries with exponential backoff on 429/5xx and falls back through a list of models,
because hosted endpoints *do* return "high demand" 503s (observed while building this).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMError(RuntimeError):
    pass


class GeminiLLM:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, models: tuple[str, ...] = ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"),
                 api_key: str | None = None, max_retries: int = 2, timeout: int = 60):
        self.models = models
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY not set")
        self.max_retries, self.timeout = max_retries, timeout
        self.name = f"gemini:{models[0]}"

    def generate(self, system: str, user: str, temperature: float = 0.1, json_mode: bool = False) -> LLMResponse:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature,
                                 **({"responseMimeType": "application/json"} if json_mode else {})},
        }
        last_err = None
        for model in self.models:
            for attempt in range(self.max_retries):
                t0 = time.perf_counter()
                try:
                    r = requests.post(self.URL.format(model=model), json=body, timeout=self.timeout,
                                      headers={"x-goog-api-key": self.api_key})
                except requests.RequestException as e:
                    last_err = str(e)
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 200:
                    d = r.json()
                    try:
                        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                    except (KeyError, IndexError) as e:
                        raise LLMError(f"unexpected Gemini response: {str(d)[:300]}") from e
                    u = d.get("usageMetadata", {})
                    return LLMResponse(text, model, time.perf_counter() - t0,
                                       u.get("promptTokenCount"), u.get("candidatesTokenCount"))
                last_err = f"{model} HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                break  # 4xx other than 429: this model won't work, try the next one
        raise LLMError(f"all Gemini models failed; last error: {last_err}")


class OllamaLLM:
    def __init__(self, model: str = "qwen3:8b", host: str | None = None, timeout: int = 300):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def generate(self, system: str, user: str, temperature: float = 0.1, json_mode: bool = False) -> LLMResponse:
        t0 = time.perf_counter()
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_ctx": 8192}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed ({self.host}): {e}") from e
        d = r.json()
        return LLMResponse(d["message"]["content"], self.model, time.perf_counter() - t0,
                           d.get("prompt_eval_count"), d.get("eval_count"))


def get_llm(backend: str):
    if backend == "gemini":
        return GeminiLLM()
    if backend.startswith("ollama"):
        _, _, model = backend.partition(":")
        return OllamaLLM(model or "qwen3:8b")
    raise ValueError(f"unknown backend {backend!r} (use 'gemini' or 'ollama[:model]')")
