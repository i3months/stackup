"""rag_eval 공통: 결과 경로, 확정 모델 목록(P-008), 실행 기록 run.json (SPEC §9.3), 운영 B0 상수.

    python common.py models      # ~/rag-lab/models 의 리비전·sha256 → results/E0-env/models.json

stackup-lab/CLAUDE.md §2 — 모든 실행 기록에 커밋·설정·모델·시드·시각·하드웨어 상태를 남긴다.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_client import DEFAULT_LAB_DIR, git_commit, stackup_commit

HERE = Path(__file__).resolve().parent
# 운영 케이스(합성 데이터) 재사용: ai/scripts/llm_eval/cases*.py
sys.path.insert(0, str(HERE.parent / "llm_eval"))

# 운영 B0 값 (docs/system-baseline.md).
PG_DSN_DEFAULT = os.environ.get("RAG_PG_DSN", "host=/tmp port=55432 user=postgres dbname=raglab")
PROD_EMBED_DIM = 1536  # document_embeddings.embedding vector(1536)
RRF_K = 60

MODELS_DIR = Path(os.environ.get("RAG_MODELS_DIR", Path.home() / "rag-lab" / "models"))
LLAMA_BIN_DIR = Path(
    os.environ.get("RAG_LLAMA_BIN_DIR", Path.home() / "rag-lab" / "mamba" / "envs" / "raglab" / "bin")
)

# P-008 확정 모델. revision·sha256 은 models.json 이 진실 공급원이다 (다운로드 시 고정).
MODELS: dict[str, dict[str, Any]] = {
    "EMB-1": {"role": "embedding", "repo": "BAAI/bge-m3", "dim": 1024},
    "EMB-2": {"role": "embedding", "repo": "nlpai-lab/KURE-v1", "dim": 1024},
    "EMB-3": {"role": "embedding", "repo": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024},
    "EMB-4": {"role": "embedding", "repo": "Qwen/Qwen3-Embedding-4B", "dim": 2560},
    "RR-1": {"role": "reranker", "repo": "BAAI/bge-reranker-v2-m3"},
    "RR-2": {"role": "reranker", "repo": "nlpai-lab/KURE-Reranker-nano"},
    "M-S": {
        "role": "generator",
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    },
    "M-M": {
        "role": "generator",
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    },
}


def lab_dir() -> Path:
    return Path(DEFAULT_LAB_DIR)


def results_dir(*parts: str) -> Path:
    d = lab_dir() / "results" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_json_path() -> Path:
    return lab_dir() / "results" / "E0-env" / "models.json"


def model_path(model_id: str) -> Path:
    m = MODELS[model_id]
    if "file" in m:
        return MODELS_DIR / m["file"]
    return MODELS_DIR / "hf" / m["repo"]


def load_models_json() -> dict[str, Any]:
    p = models_json_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 `python common.py models`")
    return {m["id"]: m for m in json.loads(p.read_text(encoding="utf-8"))["models"]}


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """줄 단위로 덧붙이고 디스크까지 내린다 (T04: 2026-10-14 전원 꺼짐 — 통째로 다시 쓰는 파일은 중간에 끊기면 잃는다)."""
    import os  # noqa: PLC0415

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---- 하드웨어·소프트웨어 상태 -------------------------------------------------
def gpu_state() -> dict[str, Any]:
    q = "name,memory.used,memory.total,power.limit,driver_version,temperature.gpu,clocks.sm"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}


def gpu_mem_used_mib() -> int | None:
    v = gpu_state().get("memory.used")
    return int(v) if v and v.isdigit() else None


def hardware() -> dict[str, Any]:
    g = gpu_state()
    mem_total_gib = None
    try:
        mem_total_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "gpu": g.get("name"),
        "gpu_mem_total_mib": g.get("memory.total"),
        "power_limit_w": float(g["power.limit"]) if g.get("power.limit") else None,
        "driver": g.get("driver_version"),
        "ram_total_gib": mem_total_gib,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def software() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "sentence_transformers", "psycopg", "anthropic", "langchain_openai"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def llama_build() -> str:
    """conda-forge 패키지 빌드 + llama-server --version 의 커밋."""
    pkg = None
    meta = LLAMA_BIN_DIR.parent / "conda-meta"
    if meta.exists():
        hits = sorted(meta.glob("llama.cpp-*.json"))
        pkg = hits[0].stem if hits else None
    try:
        r = subprocess.run(
            [str(LLAMA_BIN_DIR / "llama-server"), "--version"],
            capture_output=True, text=True, timeout=30,
        )
        ver = next((ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("version")), "")
    except Exception:  # noqa: BLE001
        ver = ""
    return f"{pkg or 'llama.cpp-?'} | {ver.strip()}"


# ---- run.json (SPEC §9.3) ---------------------------------------------------
def model_entries(model_ids: list[str]) -> list[dict[str, Any]]:
    reg = load_models_json()
    out = []
    for mid in model_ids:
        m = reg[mid]
        out.append(
            {
                "id": mid,
                "role": m["role"],
                "name": m["repo"],
                "revision": m["revision"],
                "file_sha256": {f["path"]: f["sha256"] for f in m["files"]},
            }
        )
    return out


class RunRecorder:
    """run.json 기록기. 시작 시 1회 쓰고(ended_at=null), 끝날 때 다시 쓴다.

        with RunRecorder(out_dir, run_id=..., task_id="T01", experiment="E0",
                         condition_id="smoke-emb1", config={...}, model_ids=["EMB-1"]) as rr:
            ...
            rr.n_items = 1
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        run_id: str,
        task_id: str,
        experiment: str,
        condition_id: str | None,
        config: dict[str, Any],
        model_ids: list[str] | None = None,
        server: dict[str, Any] | None = None,
        seed: int | None = None,
        notes: str = "",
        filename: str = "run.json",
    ) -> None:
        self.path = Path(out_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rec: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task_id,
            "experiment": experiment,
            "condition_id": condition_id,
            "config": config,
            "stackup_commit": stackup_commit(),
            "lab_commit": git_commit(lab_dir()),
            "models": model_entries(model_ids or []),
            "server": server,
            "seed": seed,
            "hardware": hardware(),
            "software": software(),
            "started_at": now_iso(),
            "ended_at": None,
            "n_items": 0,
            "notes": notes,
        }
        self.n_items = 0

    def __enter__(self) -> "RunRecorder":
        write_json(self.path, self.rec)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.rec["ended_at"] = now_iso()
        self.rec["n_items"] = self.n_items
        if exc is not None:
            self.rec["error"] = f"{exc_type.__name__}: {exc}"
        write_json(self.path, self.rec)


# ---- models.json -------------------------------------------------------------
def _hf_meta(repo: str) -> dict[str, Any]:
    try:
        from huggingface_hub import model_info

        info = model_info(repo)
        card = info.card_data.to_dict() if info.card_data else {}
        return {"license": card.get("license"), "hub_head_sha": info.sha}
    except Exception as exc:  # noqa: BLE001
        return {"license": None, "hub_head_sha": None, "hub_error": type(exc).__name__}


def cmd_models() -> None:
    revs: dict[str, str] = {}
    for ln in (MODELS_DIR / "revisions.txt").read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if parts:
            revs[parts[0]] = parts[1]
    shas: dict[str, str] = {}
    for ln in (MODELS_DIR / "sha256.txt").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, p = ln.split(maxsplit=1)
            shas[p.strip()] = h
    out = []
    for mid, m in MODELS.items():
        files = []
        for rel, h in shas.items():
            local = MODELS_DIR / rel
            belongs = rel == m.get("file") or rel.startswith(f"hf/{m['repo']}/")
            if belongs:
                files.append({"path": rel, "sha256": h, "bytes": local.stat().st_size})
        if not files:
            raise RuntimeError(f"{mid}: sha256.txt 에 파일이 없다")
        out.append(
            {
                "id": mid,
                **m,
                "revision": revs[m["repo"]],
                "source": f"https://huggingface.co/{m['repo']}",
                "local_path": str(model_path(mid)),
                "files": files,
                **_hf_meta(m["repo"]),
            }
        )
    path = models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "generated_at": now_iso(),
            "generator": "ai/scripts/rag_eval/common.py models",
            "stackup_commit": stackup_commit(),
            "models_dir": str(MODELS_DIR),
            "note": "sha256 은 가중치 파일만. GGUF 두 개는 HF 공시 LFS sha256 과 일치 확인함 (T01).",
            "models": out,
        },
    )
    print(f"{path}: {len(out)} models")


if __name__ == "__main__":
    if sys.argv[1:] == ["models"]:
        cmd_models()
    else:
        print(__doc__)
        sys.exit(2)
