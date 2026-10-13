"""로컬 생성 모델 서버 (llama-server) 실행·중지·오프로드 측정.

    python servers/llama_server.py start M-S                 # 전부 GPU, :8081
    python servers/llama_server.py start M-M [--n-cpu-moe N] # MoE 전문가 일부 CPU, :8082
    python servers/llama_server.py stop M-M
    python servers/llama_server.py status
    python servers/llama_server.py bench-moe --ns 30,26,22,20,18   # M-M N 별 VRAM·tok/s

- 결정성(SPEC §5.2): 슬롯 1개(`-np 1`). 시드는 요청마다 준다.
- 실행 상태는 ~/rag-lab/run/<프로필>.json 에 남긴다 (args, binary_build, n_parallel, pid).
  run.json 의 server 필드는 이 파일에서 읽는다 (`server_record`).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    LLAMA_BIN_DIR,
    gpu_mem_used_mib,
    llama_build,
    model_path,
    now_iso,
    results_dir,
    write_json,
)

RUN_DIR = Path(os.environ.get("RAG_RUN_DIR", Path.home() / "rag-lab" / "run"))
CTX = 16384
# M-M 기본 N — T01 bench-moe 실측(48GB, ctx 16384): N=14 에서 VRAM 11.6/12.0GB, 생성 14.4 tok/s.
# N=12 가 더 빠르지만(17.2) 여유가 330MB 뿐이라 데스크톱 VRAM 변동에 취약하다. 바꾸면 T01-report 와 같이 고친다.
M_M_N_CPU_MOE_DEFAULT = int(os.environ.get("RAG_M_M_N_CPU_MOE", "14"))

PROFILES: dict[str, dict[str, Any]] = {
    "M-S": {"port": 8081, "extra": []},
    # Gemma 4 템플릿은 생각(thinking) 모드가 기본으로 켜져 있어 운영 출력 상한 512 토큰을 생각에 다 쓴다
    # (T01 스모크 5: 40초, 빈 질문). SPEC §5.2 thinking off 를 서버 기본값으로 건다 — 운영 클라이언트는 무변경.
    "M-M": {"port": 8082, "extra": ["--chat-template-kwargs", '{"enable_thinking":false}']},
}


def base_url(profile: str) -> str:
    return f"http://127.0.0.1:{PROFILES[profile]['port']}"


def build_args(profile: str, n_cpu_moe: int | None) -> list[str]:
    p = PROFILES[profile]
    args = [
        "-m", str(model_path(profile)),
        "--host", "127.0.0.1",
        "--port", str(p["port"]),
        "-ngl", "99",
        "-c", str(CTX),
        "-np", "1",
        "--jinja",
        "--alias", profile,
    ]
    if profile == "M-M":
        n = M_M_N_CPU_MOE_DEFAULT if n_cpu_moe is None else n_cpu_moe
        args += ["--n-cpu-moe", str(n)]
    return args + list(p["extra"])


def _state_path(profile: str) -> Path:
    return RUN_DIR / f"{profile}.json"


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(profile: str, n_cpu_moe: int | None = None, *, timeout: float = 900) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    st = _state_path(profile)
    if st.exists():
        old = json.loads(st.read_text())
        if _alive(old["pid"]):
            raise RuntimeError(f"{profile} 이미 실행 중 (pid {old['pid']}) — 먼저 stop")
    args = build_args(profile, n_cpu_moe)
    vram_before = gpu_mem_used_mib()
    log = RUN_DIR / f"{profile}.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [str(LLAMA_BIN_DIR / "llama-server"), *args],
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    t0 = time.monotonic()
    url = base_url(profile)
    while not _healthy(url):
        if proc.poll() is not None:
            tail = log.read_text(errors="replace").splitlines()[-15:]
            raise RuntimeError(f"{profile} 시작 실패 (exit {proc.returncode}):\n" + "\n".join(tail))
        if time.monotonic() - t0 > timeout:
            proc.terminate()
            raise TimeoutError(f"{profile} {timeout}s 안에 준비되지 않음")
        time.sleep(2)
    state = {
        "profile": profile,
        "pid": proc.pid,
        "base_url": url,
        "binary_build": llama_build(),
        "args": args,
        "n_parallel": 1,
        "n_cpu_moe": n_cpu_moe if profile == "M-M" and n_cpu_moe is not None
        else (M_M_N_CPU_MOE_DEFAULT if profile == "M-M" else None),
        "load_sec": round(time.monotonic() - t0, 1),
        "vram_before_mib": vram_before,
        "vram_after_mib": gpu_mem_used_mib(),
        "started_at": now_iso(),
        "log": str(log),
    }
    write_json(st, state)
    return state


def stop(profile: str) -> None:
    st = _state_path(profile)
    if not st.exists():
        return
    pid = json.loads(st.read_text())["pid"]
    if _alive(pid):
        os.killpg(pid, signal.SIGTERM)
        for _ in range(60):
            if not _alive(pid):
                break
            time.sleep(0.5)
        else:
            os.killpg(pid, signal.SIGKILL)
    st.unlink()


def server_record(profile: str) -> dict[str, Any]:
    """run.json server 필드 (SPEC §9.3)."""
    st = json.loads(_state_path(profile).read_text())
    return {k: st[k] for k in ("binary_build", "args", "n_parallel", "n_cpu_moe", "base_url")}


def _complete(url: str, prompt: str, n_predict: int, seed: int = 1234) -> dict[str, Any]:
    body = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "seed": seed, "temperature": 0.0,
         "ignore_eos": True, "cache_prompt": False}
    ).encode()
    req = urllib.request.Request(f"{url}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def bench_moe(ns: list[int], *, n_predict: int = 256, prompt_tokens: int = 3000) -> list[dict[str, Any]]:
    """N 마다 서버를 띄워 VRAM·프롬프트 처리·생성 속도를 잰다. 실패(OOM)도 기록한다."""
    # 운영 꼬리질문 입력 규모(컨텍스트 5청크 ≈ 3천 토큰)에 맞춘 고정 프롬프트.
    filler = "지원자는 결제 시스템의 정산 배치를 재설계해 처리 시간을 줄였다. " * (prompt_tokens // 20)
    rows = []
    for n in ns:
        row: dict[str, Any] = {"n_cpu_moe": n}
        try:
            st = start("M-M", n)
            row.update(load_sec=st["load_sec"], vram_used_mib=st["vram_after_mib"])
            _complete(st["base_url"], "워밍업", 16)
            res = [_complete(st["base_url"], filler, n_predict, seed=s) for s in (1, 2)]
            t = res[-1]["timings"]
            row.update(
                prompt_n=t["prompt_n"],
                prompt_tok_s=round(t["prompt_per_second"], 1),
                gen_n=t["predicted_n"],
                gen_tok_s=round(t["predicted_per_second"], 1),
                vram_peak_mib=gpu_mem_used_mib(),
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[-400:]}")
        finally:
            stop("M-M")
            time.sleep(3)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("profile", choices=PROFILES)
    s.add_argument("--n-cpu-moe", type=int, default=None)
    t = sub.add_parser("stop")
    t.add_argument("profile", choices=PROFILES)
    sub.add_parser("status")
    b = sub.add_parser("bench-moe")
    b.add_argument("--ns", default="30,26,22,20,18,16")
    a = ap.parse_args()
    if a.cmd == "start":
        print(json.dumps(start(a.profile, a.n_cpu_moe), ensure_ascii=False, indent=1))
    elif a.cmd == "stop":
        stop(a.profile)
    elif a.cmd == "status":
        for p in PROFILES:
            st = _state_path(p)
            if st.exists():
                d = json.loads(st.read_text())
                print(p, "pid", d["pid"], "alive" if _alive(d["pid"]) else "dead", d["base_url"])
            else:
                print(p, "stopped")
    elif a.cmd == "bench-moe":
        rows = bench_moe([int(x) for x in a.ns.split(",")])
        out = results_dir("E0-env") / "m-m-offload.json"
        write_json(out, {"measured_at": now_iso(), "ctx": CTX, "binary_build": llama_build(), "rows": rows})
        print(out)


if __name__ == "__main__":
    main()
