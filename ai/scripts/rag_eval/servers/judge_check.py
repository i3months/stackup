"""로컬 판정자 후보(SPEC §5.3 J-LOC-KO, P-009) 실행 확인 — 품질 선정은 T07.

    python servers/judge_check.py            # 후보마다 서버를 띄워 한국어 JSON 판정 1건 → results/E0-env/judge-check.json

후보 파일은 ~/rag-lab/models/judge/ 에 고정 리비전으로 받아 둔다 (T01-report 표).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LLAMA_BIN_DIR, MODELS_DIR, gpu_mem_used_mib, llama_build, now_iso, results_dir, write_json  # noqa: E402

PORT = 8090
CTX = 8192
CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "J-LOC-EXAONE4-32B",
        "repo": "LGAI-EXAONE/EXAONE-4.0-32B-GGUF",
        "revision": "b390c77d8be64849356ff7bc6213c8049b5a34b2",
        "file": "EXAONE-4.0-32B-Q4_K_M.gguf",
        "sha256": "738060fec3193ccbd4615b1ac7e6404805af25c0f627093d89cf04a099dfdde8",
        "license": "EXAONE AI Model License 1.2 - NC",
        # 64층 dense. 12GB 에 맞춰 일부 층만 GPU. 4.0 은 비추론 모드가 기본이지만 명시한다.
        "args": ["-ngl", "30", "-ctk", "q8_0", "-ctv", "q8_0",
                 "--chat-template-kwargs", '{"enable_thinking":false}'],
    },
    {
        "id": "J-LOC-MISTRAL-S32-24B",
        "repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "revision": "b750ec2299225e492f1bd27cab88a0a595fa848f",
        "file": "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        "sha256": "a3cc56310807ed0d145eaf9f018ccda9ae7ad8edb41ec870aa2454b0d4700b3c",
        "license": "apache-2.0",
        "args": ["-ngl", "28"],
    },
]

SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["grounded", "reason"],
    "additionalProperties": False,
}
SYSTEM = ("당신은 한국어 면접 꼬리질문을 보는 평가자입니다. 꼬리질문이 지원자 답변에 실제로 나온 내용만 "
          "전제하는지 판정하세요. grounded 는 참/거짓, reason 은 한 문장입니다. JSON 으로만 답하세요.")
USER = ("[지원자 답변]\n주문 저장과 Kafka 발행을 한 트랜잭션으로 묶으려고 Outbox 를 도입했고, 중복 주문이 월 30건에서 0건이 됐습니다.\n\n"
        "[꼬리질문]\nRedis 분산 락으로 중복 주문을 막으셨다고 했는데, 락 만료 시간은 어떻게 정하셨나요?")


def _post(path: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def check(c: dict[str, Any]) -> dict[str, Any]:
    path = MODELS_DIR / "judge" / c["file"]
    args = ["-m", str(path), "--host", "127.0.0.1", "--port", str(PORT), "-c", str(CTX), "-np", "1", "--jinja",
            *c["args"]]
    row: dict[str, Any] = {k: c[k] for k in ("id", "repo", "revision", "file", "sha256", "license")}
    row.update(args=args, bytes=path.stat().st_size)
    log = Path.home() / "rag-lab" / "run" / f"{c['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.Popen([str(LLAMA_BIN_DIR / "llama-server"), *args], stdout=fh, stderr=subprocess.STDOUT)
    try:
        t0 = time.monotonic()
        while not _healthy():
            if proc.poll() is not None:
                raise RuntimeError("시작 실패: " + " | ".join(log.read_text(errors="replace").splitlines()[-5:]))
            if time.monotonic() - t0 > 900:
                raise TimeoutError("준비 안 됨")
            time.sleep(2)
        row["load_sec"] = round(time.monotonic() - t0, 1)
        row["vram_used_mib"] = gpu_mem_used_mib()
        t0 = time.perf_counter()
        d = _post("/v1/chat/completions", {
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
            "max_tokens": 256, "temperature": 0.0, "seed": 1234,
            "response_format": {"type": "json_schema", "json_schema": {"name": "judge", "schema": SCHEMA}},
        })
        row["latency_sec"] = round(time.perf_counter() - t0, 2)
        msg = d["choices"][0]["message"]
        row["content"] = msg.get("content")
        row["reasoning_content"] = msg.get("reasoning_content")
        row["usage"] = d.get("usage")
        t = d.get("timings") or {}
        row["prompt_tok_s"] = round(t.get("prompt_per_second", 0), 1)
        row["gen_tok_s"] = round(t.get("predicted_per_second", 0), 1)
        parsed = json.loads(row["content"])
        row["parsed"] = parsed
        # 이 예시는 답변에 없는 "Redis 분산 락" 을 전제한다 → 정답은 grounded=false
        row["runs"] = True
        row["json_ok"] = set(parsed) == {"grounded", "reason"}
        row["verdict_correct"] = parsed.get("grounded") is False
    except Exception as exc:  # noqa: BLE001
        row.update(runs=False, error=f"{type(exc).__name__}: {str(exc)[-400:]}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        time.sleep(3)
    return row


def main() -> None:
    rows = []
    for c in CANDIDATES:
        r = check(c)
        print(json.dumps({k: r.get(k) for k in ("id", "runs", "load_sec", "vram_used_mib", "latency_sec",
                                                 "gen_tok_s", "json_ok", "verdict_correct", "content", "error")},
                         ensure_ascii=False), flush=True)
        rows.append(r)
    out = results_dir("E0-env") / "judge-check.json"
    write_json(out, {"checked_at": now_iso(), "ctx": CTX, "binary_build": llama_build(), "rows": rows})
    print(out)


if __name__ == "__main__":
    main()
