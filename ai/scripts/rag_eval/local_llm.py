"""로컬 생성 모델(llama-server, OpenAI 호환) 호출 — 코퍼스 생성 전용.

- 생각 모드는 서버 기본값으로 꺼져 있다 (servers/llama_server.py). 응답마다 reasoning_content 가
  비었는지 단언한다 (SPEC §5.2, P-013).
- 호출마다 시드·토큰·지연을 calls.jsonl 에 남긴다 (생성 주체·시드 기록, SPEC §9.2 generator).
- cache_prompt 기본값은 False (P-018). llama-server 는 프롬프트 캐시 재사용 여부에 따라 같은 seed 로도
  출력이 갈린다 (results/E1-corpus/determinism-*.json). None 이면 서버 기본값(켜짐)을 쓴다 — E1a 코퍼스 생성 조건.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any


class LocalLLMError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, base_url: str, *, model: str, log_path: Path, timeout: float = 1800,
                 cache_prompt: bool | None = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.cache_prompt = cache_prompt

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        temperature: float,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        tag: str = "",
    ) -> str:
        body: dict[str, Any] = {"messages": messages, "seed": seed, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if self.cache_prompt is not None:
            body["cache_prompt"] = self.cache_prompt
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema}}
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        latency = time.perf_counter() - t0
        choice = d["choices"][0]
        msg = choice["message"]
        rc = msg.get("reasoning_content")
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": self.model, "seed": seed,
            "temperature": temperature, "max_tokens": max_tokens, "cache_prompt": self.cache_prompt, "json_schema": json_schema is not None,
            "finish_reason": choice.get("finish_reason"), "usage": d.get("usage"),
            "latency_sec": round(latency, 2), "reasoning_chars": len(rc or ""),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rc:
            raise LocalLLMError(f"[{tag}] reasoning_content 가 비어 있지 않다 ({len(rc)}자) — 생각 모드가 켜져 있다")
        if choice.get("finish_reason") == "length":
            raise LocalLLMError(f"[{tag}] 출력 상한에 걸렸다 (max_tokens={max_tokens})")
        return msg.get("content") or ""

    def chat_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any], **kw: Any) -> Any:
        return json.loads(self.chat(messages, json_schema=schema, **kw))
