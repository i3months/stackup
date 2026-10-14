"""로컬 생성 서버 결정성 점검 (관리자 T03 답변: P36 의 500 이 같은 seed 로 재실행에서 성공).

SPEC §5.2·§10.4 는 "-np 1 + seed 고정 = 재현"을 전제한다. 운영 꼬리질문 프롬프트(스모크 5 와 같은 사례·
컨텍스트)를 운영 Flash 설정(temperature·max_tokens)과 고정 seed 로 보낸다. 묶음은 다음과 같다.

    A  서버 시작 직후 5회 (cache_prompt 기본값 = 켜짐)
    B  같은 서버에서 cache_prompt=false 로 5회 (프롬프트 캐시 재사용 영향 분리)
    C  서버 재시작 뒤 5회 (cache_prompt 기본값)

출력이 A 의 첫 응답과 바이트 단위로 같은 비율과, 다르면 처음 갈라지는 문자 위치를 남긴다.

    python determinism.py --profile M-M
    python determinism.py --profile M-S

결과: stackup-lab/results/E1-corpus/determinism-<profile>.json. API 비용 없음.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import now_iso, results_dir, stackup_commit, write_json  # noqa: E402
from servers.llama_server import base_url, server_record, start, stop  # noqa: E402
from smoke import _followup_case, _settings, smoke_dir  # noqa: E402

SEED = 1234
N = 5


def prod_messages() -> list[dict[str, str]]:
    """운영 스트리밍 꼬리질문 생성기의 프롬프트 그대로 (format_instructions 포함)."""
    from ai_server.chain.followup_generation_chain import build_streaming_followup_generator  # noqa: PLC0415

    case = dict(_followup_case())
    case["context"] = (smoke_dir() / "2-emb1.context.txt").read_text(encoding="utf-8")
    gen = build_streaming_followup_generator(_settings(llm_flash_api_key="local"))
    kwargs = {k: case[k] for k in ("job_category", "mode", "previous_question", "answer_text",
                                    "context", "parent_category", "expected_signal", "history")}
    role = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role[m.type], "content": m.content} for m in gen._prompt.format_messages(**kwargs)]  # noqa: SLF001


def call(profile: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
         cache_prompt: bool | None) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": messages, "seed": SEED, "temperature": temperature, "max_tokens": max_tokens}
    if cache_prompt is not None:
        body["cache_prompt"] = cache_prompt
    req = urllib.request.Request(f"{base_url(profile)}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    assert not msg.get("reasoning_content"), "생각 모드가 켜져 있다"
    text = msg.get("content") or ""
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text), "text": text,
            "finish_reason": d["choices"][0].get("finish_reason"), "usage": d.get("usage"),
            "latency_sec": round(time.perf_counter() - t0, 2)}


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["M-S", "M-M"], required=True)
    ap.add_argument("--keep-server", action="store_true", help="이미 떠 있는 서버로 A·B 를 돌린다")
    args = ap.parse_args()
    p = args.profile
    s = _settings(llm_flash_api_key="local")
    temp, mt = s.llm_flash_temperature, s.llm_flash_max_tokens
    msgs = prod_messages()

    runs: list[dict[str, Any]] = []
    servers = []
    if not args.keep_server:
        start(p)
    servers.append(server_record(p))
    for group, cp in (("A", None), ("B", False)):
        for i in range(N):
            runs.append({"group": group, "i": i, "cache_prompt": cp,
                         **call(p, msgs, temperature=temp, max_tokens=mt, cache_prompt=cp)})
            print(group, i, runs[-1]["sha256"][:12], runs[-1]["chars"])
    stop(p)
    start(p)
    servers.append(server_record(p))
    for i in range(N):
        runs.append({"group": "C", "i": i, "cache_prompt": None,
                     **call(p, msgs, temperature=temp, max_tokens=mt, cache_prompt=None)})
        print("C", i, runs[-1]["sha256"][:12], runs[-1]["chars"])
    if not args.keep_server:
        stop(p)

    ref = runs[0]["text"]
    for r in runs:
        r["same_as_ref"] = r["text"] == ref
        r["first_diff_char"] = first_diff(ref, r["text"])
    by = {g: sum(r["same_as_ref"] for r in runs if r["group"] == g) for g in "ABC"}
    out = {
        "created_at": now_iso(), "stackup_commit": stackup_commit(), "profile": p, "seed": SEED,
        "temperature": temp, "max_tokens": mt,
        "case": "smoke 5 (cases_v2 followup, 2-emb1 context)", "n_per_group": N, "servers": servers,
        "same_as_first": {"A": f"{by['A']}/{N}", "B_cache_prompt_false": f"{by['B']}/{N}", "C_after_restart": f"{by['C']}/{N}"},
        "byte_identical_rate": round(sum(by.values()) / len(runs), 3),
        "distinct_outputs": len({r["sha256"] for r in runs}),
        "runs": runs,
    }
    write_json(results_dir("E1-corpus") / f"determinism-{p}.json", out)
    print(json.dumps(out["same_as_first"], ensure_ascii=False), "distinct", out["distinct_outputs"])


if __name__ == "__main__":
    main()
