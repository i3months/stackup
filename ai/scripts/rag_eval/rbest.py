"""R-BEST 선택 (SPEC §5.1, dev 전용): B0 에서 출발한 탐욕적 전진 선택.

    python rbest.py            # → E2-retrieval/R-BEST-selection.json (단계별 후보 전체 점수), conditions 에 쓰지 않음

- 단계마다 현재 설정에서 요인 하나만 바꾼 후보 전부를 dev k=5 로 돌려 nDCG@5 를 잰다. 가장 많이 올리는 수준을
  채택하고, 개선이 0.005 미만이면 멈춘다. 개선이 같으면(1e-9 이내) 운영 수준에 가까운(더 단순·저렴한) 쪽을 고른다.
- 청킹 수준은 인덱싱 대상에 따라 다르다: 요약이면 500/100 만 (1500/300·헤딩은 B0 와 같은 청크, P-019), 원문·둘 다면
  500/100·1500/300·헤딩(최대 2000) (탐색 조건 R-SRCraw-CH-* 의 수준).
- EMB-4m·EMB-1-bf16 은 후보에서 뺀다 (점검용). 1단계 후보는 이미 돈 dev 조건 run 을 그대로 쓴다.
- 새 조합 run 은 E2-retrieval/dev-rbest/<조합 id>-k5/ 에 둔다. 끊겨도 다시 돌리면 끝난 run 은 건너뛴다.
- 새 접두가 필요한 조합(원문 + PFX-llm 등)은 prefixes.jsonl 에 그 변형의 llm 접두가 전부 있을 때만 평가하고, 없으면
  `needs_llm_prefix` 로 표시하고 건너뛴다 (P-023: 접두를 만든 뒤 다시 돌린다).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import now_iso, results_dir, write_json  # noqa: E402
from labels import Labels, read_jsonl  # noqa: E402
from t05_dev import LABELS_DIR, SPLIT, _done, _out  # noqa: E402

MIN_GAIN = 0.005
TIE = 1e-9
K = 5

# 요인 → 수준 (운영 수준이 맨 앞, 뒤로 갈수록 복잡·비싸다 — 동률이면 앞쪽)
CHUNK_SUMMARY = [(1000, 200, "recursive"), (500, 100, "recursive")]
CHUNK_RAW = [(1000, 200, "recursive"), (500, 100, "recursive"), (1500, 300, "recursive"), (2000, 0, "heading")]
FACTORS: dict[str, list[Any]] = {
    "index_source": ["summary", "raw", "both"],
    "chunk": None,  # 인덱싱 대상에 따라
    "prefix": ["deterministic", "none", "llm"],
    "embedder": ["EMB-1", "EMB-3", "EMB-2", "EMB-4"],
    "retrieval": ["hybrid_simple", "vector", "bm25_kiwi", "hybrid_kiwi"],
    "cand": ["k", "4k"],
    "reranker": ["none", "RR-2", "RR-1"],
}
EMB_SETTINGS = {"EMB-1": ("none", "fp32"), "EMB-2": ("none", "fp32"), "EMB-3": ("query", "fp32"),
                "EMB-4": ("query", "bf16")}  # 사전 등록 조건과 같은 짝 (query_instruction, dtype)


def levels(cfg: C.Condition, factor: str) -> list[Any]:
    if factor == "chunk":
        return CHUNK_SUMMARY if cfg.index_source == "summary" else CHUNK_RAW
    return FACTORS[factor]


def get(cfg: C.Condition, factor: str) -> Any:
    if factor == "chunk":
        return (cfg.chunk_size, cfg.chunk_overlap, cfg.chunk_mode)
    return getattr(cfg, factor)


def setf(cfg: C.Condition, factor: str, level: Any) -> C.Condition:
    if factor == "chunk":
        cs, co, cm = level
        return dataclasses.replace(cfg, chunk_size=cs, chunk_overlap=co, chunk_mode=cm)
    if factor == "embedder":
        qi, dt = EMB_SETTINGS[level]
        return dataclasses.replace(cfg, embedder=level, query_instruction=qi, dtype=dt)
    if factor == "index_source" and level == "summary" and get(cfg, "chunk") not in CHUNK_SUMMARY:
        cfg = dataclasses.replace(cfg, chunk_size=1000, chunk_overlap=200, chunk_mode="recursive")
    return dataclasses.replace(cfg, **{factor: level})


def signature(cfg: C.Condition) -> tuple:
    d = dataclasses.asdict(cfg)
    return tuple(sorted((k, v) for k, v in d.items() if k not in ("condition_id", "description", "exploratory")))


def combo_id(cfg: C.Condition, base: C.Condition) -> str:
    parts = []
    for f in FACTORS:
        if get(cfg, f) != get(base, f):
            v = get(cfg, f)
            parts.append(f"{f}={'/'.join(map(str, v)) if isinstance(v, tuple) else v}")
    return "B0" if not parts else "RB[" + ",".join(parts) + "]"


def dev_ndcg(run_dir: Path, cond: C.Condition, ctx: dict[str, Any]) -> dict[str, Any]:
    from metrics import per_query, summarize, variant_chunk_keys  # noqa: PLC0415

    if cond.variant_id not in ctx["by_var"]:
        rows = [c for c in read_jsonl(results_dir(LABELS_DIR) / "chunks.jsonl") if c["variant_id"] == cond.variant_id]
        ctx["by_var"][cond.variant_id] = variant_chunk_keys(rows)
    pq = per_query(read_jsonl(run_dir / "rankings.jsonl"), ctx["queries"], ctx["labels"], ctx["by_var"][cond.variant_id])
    s = summarize(pq, K, None)
    return {"ndcg@5": s["ndcg@5"], "fact_recall@5": s["fact_recall@5"], "hit@5": s["hit@5"],
            "unreachable": s["unreachable"]}


def llm_prefixed_variants() -> set[str]:
    """prefixes.jsonl 에 PFX-llm 접두가 전 청크만큼 있는 변형."""
    from t05_dev import _expected  # noqa: PLC0415

    n: dict[str, int] = {}
    for r in read_jsonl(results_dir(LABELS_DIR) / "prefixes.jsonl"):
        if r["prefix_id"] == "llm":
            n[r["variant_id"]] = n.get(r["variant_id"], 0) + 1
    out = set()
    for v, c in n.items():
        src, rest = v.split("-", 1)
        mode, size_ov = rest[0], rest[1:]
        cs, co = size_ov.split("o")
        cond = dataclasses.replace(C.load("B0"), index_source=src, chunk_size=int(cs), chunk_overlap=int(co),
                                   chunk_mode="recursive" if mode == "c" else "heading")
        if _expected(cond)[1] == c:
            out.add(v)
    return out


def ensure_run(cond: C.Condition, cid: str, existing: dict[tuple, str], ctx: dict[str, Any]) -> tuple[Path | None, str]:
    """돈 적 있는 dev run 을 찾거나, 없으면 인덱싱·검색한다. (run 디렉터리, 출처)."""
    sig = signature(cond)
    if sig in existing:
        return _out(SPLIT, f"{existing[sig]}-k{K}"), f"dev/{existing[sig]}"
    if cond.prefix == "llm" and cond.variant_id not in ctx["llm_prefixed"]:
        return None, "needs_llm_prefix"
    d = _out(f"{SPLIT}-rbest", f"{cid}-k{K}")
    if not _done(d, len(ctx["qlist"])):
        # 한 프로세스에서 run 을 이어 돌리면 GPU 메모리가 새서(12GB 가 차서) 10~40배 느려진다 → run 마다 새 프로세스
        import subprocess  # noqa: PLC0415

        subprocess.run([sys.executable, "-u", __file__, "--one", json.dumps(dataclasses.asdict(cond), ensure_ascii=False),
                        "--dir", str(d)], check=True)
    return d, f"dev-rbest/{cid}"


def run_one(cond_json: str, d: str) -> None:
    from index import index  # noqa: PLC0415
    from retrieve import load_queries, run  # noqa: PLC0415
    from t05_dev import _expected, _loaded  # noqa: PLC0415

    cond = C.Condition(**json.loads(cond_json))
    qlist = load_queries(results_dir("E1-queries") / "queries.jsonl", SPLIT)
    if _expected(cond) != _loaded(cond):
        index(cond, "E1", LABELS_DIR)
    run(cond, qlist, K, run_id=f"{cond.condition_id}-k{K}", out_dir=Path(d), labels_dir=LABELS_DIR, task_id="T05",
        experiment="E2", notes="T05 R-BEST 탐욕 선택 (dev)")


def main() -> None:
    from retrieve import load_queries  # noqa: PLC0415

    allc = C.load_all()
    base = allc["B0"]
    existing = {}
    for cid, c in allc.items():
        if cid in ("R-EMB-4m", "R-EMB-1-bf16", "R-CH-1500", "R-CH-head", "R-RR-1-q512"):
            continue
        existing.setdefault(signature(c), cid)
    qlist = load_queries(results_dir("E1-queries") / "queries.jsonl", SPLIT)
    ctx = {"qlist": qlist, "queries": {q["query_id"]: q for q in qlist}, "by_var": {},
           "labels": Labels.load(LABELS_DIR, results_dir("E1-corpus") / "facts.jsonl")}
    ctx["llm_prefixed"] = llm_prefixed_variants()
    cur = base
    cur_score = dev_ndcg(_out(SPLIT, f"B0-k{K}"), base, ctx)
    steps, adopted, stop_reason = [], [], None
    while True:
        cands = []
        for f in FACTORS:
            lv = levels(cur, f)
            for rank, level in enumerate(lv):
                if level == get(cur, f):
                    continue
                cfg = setf(cur, f, level)
                cid = combo_id(cfg, base)
                cfg = dataclasses.replace(cfg, condition_id=cid, description=f"R-BEST 후보: {cid}", exploratory=True)
                d, src = ensure_run(cfg, cid, existing, ctx)
                row = {"factor": f, "level": list(level) if isinstance(level, tuple) else level, "simplicity_rank": rank,
                       "candidate": cid, "source": src, "index_key": cfg.index_key}
                if d is None:
                    row["skipped"] = src
                else:
                    row |= dev_ndcg(d, cfg, ctx)
                    row["gain"] = round(row["ndcg@5"] - cur_score["ndcg@5"], 4)
                cands.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        scored = [c for c in cands if "gain" in c]
        best_gain = max(c["gain"] for c in scored)
        tops = [c for c in scored if c["gain"] >= best_gain - TIE]
        pick = min(tops, key=lambda c: c["simplicity_rank"])
        step = {"step": len(steps) + 1, "from": combo_id(cur, base), "from_ndcg@5": cur_score["ndcg@5"],
                "candidates": sorted(cands, key=lambda c: -c.get("gain", -9)), "best": pick["candidate"],
                "best_gain": pick["gain"], "tied_best": [c["candidate"] for c in tops]}
        if any("skipped" in c for c in cands):
            step["skipped"] = [c["candidate"] for c in cands if "skipped" in c]
        steps.append(step)
        if pick["gain"] < MIN_GAIN:
            stop_reason = f"최대 개선 {pick['gain']} < {MIN_GAIN}"
            break
        lvl = tuple(pick["level"]) if pick["factor"] == "chunk" else pick["level"]
        cur = setf(cur, pick["factor"], lvl)
        cur_score = {k: pick[k] for k in ("ndcg@5", "fact_recall@5", "hit@5", "unreachable")}
        adopted.append({"factor": pick["factor"], "level": pick["level"], "gain": pick["gain"]})
    best = dataclasses.replace(cur, condition_id="R-BEST", description="R-BEST: " + combo_id(cur, base), exploratory=False)
    out = {"rule": "SPEC §5.1 탐욕적 전진 선택, dev nDCG@5, 최소 개선 0.005, 동률이면 운영 수준", "k": K, "split": SPLIT,
           "start": {"condition": "B0", **dev_ndcg(_out(SPLIT, f"B0-k{K}"), base, ctx)}, "adopted": adopted,
           "stop_reason": stop_reason, "r_best": dataclasses.asdict(best) | {"variant_id": best.variant_id,
                                                                           "index_key": best.index_key},
           "r_best_dev": cur_score, "steps": steps, "ts": now_iso()}
    # P-023: 원문 PFX-llm 후보를 건너뛴 첫 경로를 같이 남긴다 (두 경로 비교)
    first = _out() / "R-BEST-selection-path1-no-raw-llm.json"
    if first.exists():
        out["path1_without_raw_llm_prefix"] = json.loads(first.read_text(encoding="utf-8"))
    out["llm_prefixed_variants"] = sorted(ctx["llm_prefixed"])
    write_json(_out() / "R-BEST-selection.json", out)
    print(json.dumps({k: out[k] for k in ("adopted", "stop_reason", "r_best_dev")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if "--one" in sys.argv:
        run_one(sys.argv[sys.argv.index("--one") + 1], sys.argv[sys.argv.index("--dir") + 1])
    else:
        main()
