"""RAG 실험용 Anthropic 호출 래퍼 — 일반 호출 · Batch API · 프롬프트 캐시 · 구조화 출력 · 비용 원장.

- API 키는 환경변수 ANTHROPIC_API_KEY 로만 읽는다 (SDK 기본 동작). 어디에도 저장·출력하지 않는다.
- 호출마다 SPEC §9.3 필드로 stackup-lab/results/_ledger.jsonl 에 1줄을 추가한다.
- 지시서(task_id) 단위 예산 상한을 받는다. 원장의 그 task_id 누적 비용(+ 이번 호출 예상치)이
  상한의 80% 에 닿으면 BudgetExceeded 를 내고 호출하지 않는다 (SPEC §10-6).
- 판정 모델이 조용히 바뀌면 판정자 비교가 깨지므로 서버 측 fallbacks 는 쓰지 않는다.
  refusal 등은 stop_reason 그대로 원장에 남긴다.

사용 예:
    from llm_client import LabClient, cached_text, json_of
    lab = LabClient(task_id="T01", budget_usd=1.0, run_id="E0-smoke-6")
    msg = lab.call(model="claude-haiku-5-5", max_tokens=512,
                   system=[cached_text(RUBRIC)], messages=[...],
                   json_schema=SCHEMA, output_config={"effort": "low"})
    data = json_of(msg)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# 백만 토큰당 USD. 단가가 바뀌면 PRICES 와 PRICE_VERSION 을 같이 고친다 —
# 원장의 price_version 으로 어떤 단가표로 계산했는지 식별한다 (SPEC §9.3).
# 캐시 쓰기: 5분 TTL 1.25x, 1시간 TTL 2x (입력 단가 기준). Batch: 모든 토큰 50% 할인.
# Haiku 5.5 는 프롬프트가 10만 토큰을 넘으면 $0.50/$2.50 단가가 적용된다.
PRICE_VERSION = "anthropic-2026-10-a"
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
}
_LONG_PROMPT_TOKENS = 100_000
_HAIKU_LONG = {"input": 0.50, "output": 2.50, "cache_read": 0.05}
_CACHE_WRITE_5M = 1.25
_CACHE_WRITE_1H = 2.0
_BATCH_DISCOUNT = 0.5
BUDGET_STOP_RATIO = 0.8

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../stackup
DEFAULT_LAB_DIR = Path(
    os.environ.get("STACKUP_LAB_DIR", _REPO_ROOT.parent / "stackup-lab")
)


class BudgetExceeded(RuntimeError):
    pass


def cached_text(text: str, ttl: str | None = None) -> dict[str, Any]:
    """캐시 브레이크포인트가 붙은 text 블록 (system 또는 user content 에 사용). ttl: None(5m) | "1h"."""
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    return {"type": "text", "text": text, "cache_control": cc}


def git_commit(repo: Path) -> str:
    """짧은 커밋 해시. 추적 파일에 커밋 안 된 변경이 있으면 -dirty 를 붙인다."""
    # WSL 에서 Windows 체크아웃(/mnt/c)을 읽으면 소유자·파일 모드·줄바꿈이 달라 git 이 거부하거나
    # 전부 수정됨으로 본다. 전역 설정은 건드리지 않고 이 호출에만 Windows 쪽과 같은 옵션을 준다.
    git = [
        "git", "-c", f"safe.directory={repo}", "-c", "core.filemode=false",
        "-c", "core.autocrlf=true", "-C", str(repo),
    ]
    try:
        out = subprocess.run(
            [*git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def stackup_commit() -> str:
    return git_commit(_REPO_ROOT)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0

    @classmethod
    def from_sdk(cls, u: Any) -> "Usage":
        cw_total = getattr(u, "cache_creation_input_tokens", 0) or 0
        detail = getattr(u, "cache_creation", None)
        cw_5m = getattr(detail, "ephemeral_5m_input_tokens", None) if detail else None
        cw_1h = getattr(detail, "ephemeral_1h_input_tokens", None) if detail else None
        if cw_5m is None and cw_1h is None:
            # TTL 분해가 없으면 보수적으로 전부 5분으로 본다 (1h 는 명시해야만 쓰인다).
            cw_5m, cw_1h = cw_total, 0
        return cls(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_write_5m=cw_5m or 0,
            cache_write_1h=cw_1h or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_write_5m + self.cache_write_1h + self.cache_read


def price_for(model: str, prompt_tokens: int) -> dict[str, float]:
    if model not in PRICES:
        raise KeyError(f"단가 미등록 모델: {model} — llm_client.PRICES 에 추가")
    if model == "claude-haiku-5-5" and prompt_tokens > _LONG_PROMPT_TOKENS:
        return _HAIKU_LONG
    return PRICES[model]


def cost_usd(model: str, usage: Usage, *, batch: bool) -> float:
    p = price_for(model, usage.prompt_tokens)
    c = (
        usage.input_tokens * p["input"]
        + usage.cache_write_5m * p["input"] * _CACHE_WRITE_5M
        + usage.cache_write_1h * p["input"] * _CACHE_WRITE_1H
        + usage.cache_read * p["cache_read"]
        + usage.output_tokens * p["output"]
    ) / 1_000_000
    return c * (_BATCH_DISCOUNT if batch else 1.0)


def _with_schema(params: dict[str, Any], json_schema: dict[str, Any] | None) -> dict[str, Any]:
    """json_schema 를 output_config.format 으로 넣는다 (effort 등 기존 output_config 와 병합)."""
    if json_schema is None:
        return params
    oc = dict(params.get("output_config") or {})
    oc["format"] = {"type": "json_schema", "schema": json_schema}
    return {**params, "output_config": oc}


class LabClient:
    def __init__(
        self,
        *,
        task_id: str,
        budget_usd: float,
        run_id: str | None = None,
        lab_dir: Path | str = DEFAULT_LAB_DIR,
        client: Any | None = None,
    ) -> None:
        if budget_usd is None or budget_usd <= 0:
            raise ValueError("budget_usd 는 양수여야 한다 — 상한 없는 호출은 금지")
        self.task_id = task_id
        self.run_id = run_id
        self.budget_usd = budget_usd
        self.ledger_path = Path(lab_dir) / "results" / "_ledger.jsonl"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client if client is not None else anthropic.Anthropic()
        self.commit = stackup_commit()

    # ---- 원장 ---------------------------------------------------------------
    def spent_usd(self, task_id: str | None = None) -> float:
        tid = task_id or self.task_id
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        with self.ledger_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("task_id") == tid:
                    total += rec.get("cost_usd", 0.0)
        return total

    @property
    def stop_at_usd(self) -> float:
        return self.budget_usd * BUDGET_STOP_RATIO

    def check_budget(self, estimate_usd: float = 0.0) -> None:
        spent = self.spent_usd()
        if spent + estimate_usd >= self.stop_at_usd:
            raise BudgetExceeded(
                f"{self.task_id}: 누적 ${spent:.4f} + 예상 ${estimate_usd:.4f} 가 "
                f"상한 ${self.budget_usd} 의 {int(BUDGET_STOP_RATIO * 100)}% "
                f"(${self.stop_at_usd:.4f}) 이상 — 관리자 보고 필요"
            )

    def _record(
        self,
        *,
        model: str,
        usage: Usage,
        batch: bool,
        request_id: str | None,
        batch_id: str | None,
        custom_id: str | None,
        stop_reason: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "task_id": self.task_id,
            "run_id": self.run_id,
            "model": model,
            "batch": batch,
            "request_id": request_id,
            "batch_id": batch_id,
            "custom_id": custom_id,
            "input_tokens": usage.input_tokens,
            "cache_write_5m_tokens": usage.cache_write_5m,
            "cache_write_1h_tokens": usage.cache_write_1h,
            "cache_read_tokens": usage.cache_read,
            "output_tokens": usage.output_tokens,
            "stop_reason": stop_reason,
            "price_version": PRICE_VERSION,
            "cost_usd": round(cost_usd(model, usage, batch=batch), 8),
            "stackup_commit": self.commit,
        }
        if extra:
            rec.update(extra)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---- 일반 호출 ------------------------------------------------------------
    def call(
        self,
        *,
        estimate_usd: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> Any:
        """client.messages.create 와 같은 인자 + json_schema(구조화 출력). 응답 Message 를 돌려준다."""
        self.check_budget(estimate_usd)
        params = _with_schema(params, json_schema)
        t0 = time.perf_counter()
        msg = self.client.messages.create(**params)
        latency = round(time.perf_counter() - t0, 3)
        self._record(
            model=params["model"],
            usage=Usage.from_sdk(msg.usage),
            batch=False,
            request_id=getattr(msg, "_request_id", None),
            batch_id=None,
            custom_id=None,
            stop_reason=msg.stop_reason,
            extra={"latency_sec": latency},
        )
        return msg

    # ---- Batch API ------------------------------------------------------------
    def submit_batch(
        self,
        requests: Iterable[tuple[str, dict[str, Any]]],
        *,
        estimate_usd: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """(custom_id, messages.create 인자) 목록을 Batch 로 제출하고 batch id 를 돌려준다.
        비용은 결과를 받을 때(collect_batch) custom_id 마다 원장에 기록된다.
        estimate_usd 는 Batch 전체(할인 반영) 예상치다."""
        self.check_budget(estimate_usd)
        reqs = [
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(**_with_schema(params, json_schema)),
            )
            for cid, params in requests
        ]
        batch = self.client.messages.batches.create(requests=reqs)
        return batch.id

    def wait_batch(
        self, batch_id: str, *, poll_sec: float = 30.0, timeout_sec: float | None = None
    ) -> Any:
        t0 = time.monotonic()
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                return b
            if timeout_sec is not None and time.monotonic() - t0 > timeout_sec:
                raise TimeoutError(f"batch {batch_id} 미완료 ({b.processing_status})")
            time.sleep(poll_sec)

    def collect_batch(self, batch_id: str) -> dict[str, Any]:
        """끝난 Batch 결과를 custom_id → 결과 dict 로 모으고 성공 건마다 원장에 기록한다.
        결과 순서는 보장되지 않으므로 반드시 custom_id 로 매칭한다. 같은 batch 를 두 번
        회수해도 원장이 중복되지 않게 이미 기록된 (batch_id, custom_id) 는 건너뛴다."""
        seen = self._ledger_batch_keys(batch_id)
        out: dict[str, Any] = {}
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                m = r.result.message
                if r.custom_id not in seen:
                    self._record(
                        model=m.model,
                        usage=Usage.from_sdk(m.usage),
                        batch=True,
                        request_id=None,
                        batch_id=batch_id,
                        custom_id=r.custom_id,
                        stop_reason=m.stop_reason,
                    )
                out[r.custom_id] = {"type": kind, "message": m}
            else:
                # errored·canceled·expired 는 과금되지 않는다.
                err = getattr(r.result, "error", None)
                out[r.custom_id] = {"type": kind, "error": str(err) if err else None}
        return out

    def _ledger_batch_keys(self, batch_id: str) -> set[str]:
        if not self.ledger_path.exists():
            return set()
        keys = set()
        with self.ledger_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("batch_id") == batch_id:
                        keys.add(rec.get("custom_id"))
        return keys


def text_of(msg: Any) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


def json_of(msg: Any) -> Any:
    """구조화 출력(output_config.format) 응답의 JSON 을 파싱한다."""
    return json.loads(text_of(msg))
