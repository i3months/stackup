"""T05 A·B: E1 인덱스 22조건, dev 검색 (k = 3·5·10), 결정성 관문, dev 지표·표.

    python t05_dev.py index                 # 고유 인덱스 키마다 E1 인덱스 (이미 맞게 적재돼 있으면 건너뜀)
    python t05_dev.py run [--rep]           # dev 검색 → E2-retrieval/dev/<조건>-k<k>/ (--rep: dev-rep/, 결정성용)
    python t05_dev.py determinism           # dev 와 dev-rep 의 B0·R-RR-1 순위 비교 → E2-retrieval/determinism-dev.json
    python t05_dev.py metrics               # 각 dev run 의 metrics.json + E2-retrieval/dev-table.json

- 끊겨도 다시 돌리면 끝난 것은 건너뛴다: 인덱스는 DB 행 수, run 은 run.json ended_at·오류 없음·순위 행 수로 본다.
- R-CH-1500·R-CH-head 는 요약 문서가 짧아 B0 와 청크가 같다 (T04). 청크 키가 같음을 단언하고 B0 결과를 재사용한다
  (`dev/reuse.json`). R-CH-500 은 따로 돈다 (P-019).
- test 질의는 쓰지 않는다 (SPEC v2.0 동결 전, T05 지시서 B).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import PG_DSN_DEFAULT, append_jsonl, now_iso, results_dir, write_json  # noqa: E402
from labels import Labels, read_jsonl  # noqa: E402

KS = (3, 5, 10)
SPLIT = "dev"
LABELS_DIR = "E1-labels"
REUSE_OF_B0 = ("R-CH-1500", "R-CH-head")
DETERMINISM = ("B0", "R-RR-1")


def _out(*p: str) -> Path:
    return results_dir("E2-retrieval", *p)


def conditions() -> dict[str, C.Condition]:
    return C.load_all()


def run_conditions() -> dict[str, C.Condition]:
    return {k: v for k, v in conditions().items() if k not in REUSE_OF_B0}


# ---- 인덱스 -----------------------------------------------------------------------
def _expected(cond: C.Condition) -> tuple[int, int]:
    """(DB 문서 행 수, 청크 수) — E1 코퍼스 기준."""
    from index import load_corpus, variant_chunks  # noqa: PLC0415

    rows = variant_chunks(cond, load_corpus("E1"))
    return len({(r["doc_id"], r["part"]) for r in rows}), len(rows)


def _loaded(cond: C.Condition) -> tuple[int, int]:
    import psycopg  # noqa: PLC0415

    from index import D2560_TABLE  # noqa: PLC0415
    from common import MODELS, PROD_EMBED_DIM  # noqa: PLC0415

    table = "document_embeddings" if (cond.truncate_dim or MODELS[cond.embedder]["dim"]) <= PROD_EMBED_DIM \
        else D2560_TABLE
    with psycopg.connect(PG_DSN_DEFAULT) as conn, conn.cursor() as cur:
        cur.execute("SELECT db_document_id FROM rag_eval_docmap WHERE index_key=%s", (cond.index_key,))
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0, 0
        cur.execute(f"SELECT count(*) FROM {table} WHERE document_id = ANY(%s)", (ids,))
        return len(ids), cur.fetchone()[0]


def assert_reuse() -> dict[str, Any]:
    from index import load_corpus, variant_chunks  # noqa: PLC0415

    docs = load_corpus("E1")
    conds = conditions()
    base = [(r["doc_id"], r["chunk_index"], r["chunk_key"]) for r in variant_chunks(conds["B0"], docs)]
    out = {}
    for cid in REUSE_OF_B0:
        got = [(r["doc_id"], r["chunk_index"], r["chunk_key"]) for r in variant_chunks(conds[cid], docs)]
        assert got == base, f"{cid} 청크가 B0 와 다르다 — 재사용할 수 없다"
        out[cid] = {"reuses": "B0", "n_chunks": len(got), "same_chunks_as_B0": True}
    # both 는 summary·raw 청크를 합친 것 — 라벨 재사용 (T04 지시서 B-2)
    both = {r["chunk_key"] for r in variant_chunks(conds["R-SRC-both"], docs)}
    parts = {r["chunk_key"] for r in variant_chunks(conds["B0"], docs)} | \
        {r["chunk_key"] for r in variant_chunks(conds["R-SRC-raw"], docs)}
    assert both <= parts, "both 청크 키가 summary ∪ raw 에 없다"
    out["R-SRC-both"] = {"chunk_keys_subset_of_summary_and_raw": True, "n_unique": len(both)}
    write_json(_out(SPLIT) / "reuse.json", out)
    return out


def cmd_index() -> None:
    from index import index  # noqa: PLC0415

    log = _out() / "index-log.jsonl"
    seen: dict[str, str] = {}
    for cid, cond in run_conditions().items():
        if cond.index_key in seen:
            continue
        seen[cond.index_key] = cid
        exp, got = _expected(cond), _loaded(cond)
        if exp == got:
            print(f"skip {cid} {cond.index_key} {got}", flush=True)
            continue
        info = index(cond, "E1", LABELS_DIR)
        info.update(ts=now_iso(), expected=exp)
        assert (info["n_docs"], info["n_chunks"]) == exp, info
        append_jsonl(log, [info])
        print(json.dumps(info, ensure_ascii=False), flush=True)
        gc.collect()
    print(json.dumps(assert_reuse(), ensure_ascii=False))


# ---- 검색 -----------------------------------------------------------------------
def _done(d: Path, n: int) -> bool:
    rj, rk = d / "run.json", d / "rankings.jsonl"
    if not (rj.exists() and rk.exists()):
        return False
    rec = json.loads(rj.read_text(encoding="utf-8"))
    return rec.get("ended_at") is not None and "error" not in rec and \
        sum(1 for x in rk.read_text(encoding="utf-8").splitlines() if x.strip()) == n


def cmd_run(rep: bool) -> None:
    import torch  # noqa: PLC0415

    from retrieve import load_queries, run  # noqa: PLC0415

    qs = load_queries(results_dir("E1-queries") / "queries.jsonl", SPLIT)
    assert qs and all(q["split"] == SPLIT for q in qs)
    sub = f"{SPLIT}-rep" if rep else SPLIT
    conds = run_conditions()
    ids = list(DETERMINISM) if rep else list(conds)
    for cid in ids:
        for k in KS:
            rid = f"{cid}-k{k}"
            d = _out(sub, rid)
            if _done(d, len(qs)):
                continue
            t0 = time.monotonic()
            rows = run(conds[cid], qs, k, run_id=rid, out_dir=d, labels_dir=LABELS_DIR, task_id="T05",
                       experiment="E2", notes="T05 dev" + (" (결정성 재실행)" if rep else ""))
            print(f"{sub}/{rid} {len(rows)} {time.monotonic() - t0:.0f}s", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _results_only(path: Path) -> list[str]:
    """지연(latency_ms)·run_id 를 뺀 순위 부분을 질의 순서대로 직렬화."""
    out = []
    for r in read_jsonl(path):
        out.append(json.dumps({"query_id": r["query_id"], "k": r["k"], "score_type": r["score_type"],
                               "results": r["results"]}, ensure_ascii=False, sort_keys=True))
    return out


def cmd_determinism() -> None:
    rep = {}
    ok = True
    for cid in DETERMINISM:
        for k in KS:
            a = _results_only(_out(SPLIT, f"{cid}-k{k}") / "rankings.jsonl")
            b = _results_only(_out(f"{SPLIT}-rep", f"{cid}-k{k}") / "rankings.jsonl")
            diff = [json.loads(x)["query_id"] for x, y in zip(a, b) if x != y]
            same = len(a) == len(b) and not diff
            ok &= same
            rep[f"{cid}-k{k}"] = {"n_queries": len(a), "identical": same, "differing_queries": diff[:20]}
    out = {"gate": "pass" if ok else "fail", "compared": "rankings.jsonl 의 query_id·k·score_type·results "
           "(rank·chunk_key·doc_id·chunk_index·score·부가 순위) 를 sort_keys JSON 으로 바이트 비교, latency_ms 제외",
           "runs": rep, "ts": now_iso()}
    write_json(_out() / "determinism-dev.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if not ok:
        sys.exit(1)


# ---- 지표 -----------------------------------------------------------------------
def cmd_metrics() -> None:
    from metrics import per_query, summarize, variant_chunk_keys  # noqa: PLC0415

    labels = Labels.load(LABELS_DIR, results_dir("E1-corpus") / "facts.jsonl")
    chunks = read_jsonl(results_dir(LABELS_DIR) / "chunks.jsonl")
    queries = {q["query_id"]: q for q in read_jsonl(results_dir("E1-queries") / "queries.jsonl")}
    by_var: dict[str, dict[str, list[str]]] = {}
    table = {}
    for cid, cond in run_conditions().items():
        if cond.variant_id not in by_var:
            by_var[cond.variant_id] = variant_chunk_keys([c for c in chunks if c["variant_id"] == cond.variant_id])
            assert by_var[cond.variant_id], f"chunks.jsonl 에 {cond.variant_id} 없음"
        mrr = {r["query_id"]: r for r in read_jsonl(_out(SPLIT, f"{cid}-k10") / "rankings.jsonl")}
        table[cid] = {"exploratory": cond.exploratory, "variant_id": cond.variant_id}
        for k in KS:
            d = _out(SPLIT, f"{cid}-k{k}")
            rankings = read_jsonl(d / "rankings.jsonl")
            meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
            rows = per_query(rankings, queries, labels, by_var[cond.variant_id], mrr)
            s = summarize(rows, k, f"E2-retrieval/{SPLIT}/{cid}-k10")
            s["ties"] = meta["config"]["ties"]
            write_json(d / "metrics.json", {"run_id": meta["run_id"], "condition_id": cid, "summary": s,
                                            "per_query": rows})
            table[cid][f"k{k}"] = {key: s.get(key) for key in (
                f"ndcg@{k}", f"fact_recall@{k}", f"hit@{k}", "mrr@10", "judged@k", "n_answerable",
                "unreachable", "unreachable_rate", f"ndcg@{k}_reachable_only")} | {"tie_rate": s["ties"]["tie_rate"]}
    for cid in REUSE_OF_B0:
        table[cid] = dict(table["B0"]) | {"reuses": "B0", "variant_id": conditions()[cid].variant_id,
                                          "exploratory": conditions()[cid].exploratory}
    write_json(_out() / "dev-table.json", table)
    print(f"{'조건':18} {'nDCG@5':>7} {'FR@5':>6} {'Hit@5':>6} {'MRR10':>6} {'도달불가':>6} {'동점':>5} {'nDCG@3':>7} {'nDCG@10':>7}")
    for cid, t in sorted(table.items(), key=lambda x: -(x[1]["k5"]["ndcg@5"] or 0)):
        k5 = t["k5"]
        print(f"{cid:18} {k5['ndcg@5']:7.3f} {k5['fact_recall@5']:6.3f} {k5['hit@5']:6.3f} {k5['mrr@10']:6.3f} "
              f"{k5['unreachable']:>6} {k5['tie_rate']:5.2f} {t['k3']['ndcg@3']:7.3f} {t['k10']['ndcg@10']:7.3f}"
              + (" (탐색)" if t["exploratory"] else "") + (" =B0" if t.get("reuses") else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "run", "determinism", "metrics"])
    ap.add_argument("--rep", action="store_true")
    a = ap.parse_args()
    {"index": cmd_index, "run": lambda: cmd_run(a.rep), "determinism": cmd_determinism,
     "metrics": cmd_metrics}[a.cmd]()


if __name__ == "__main__":
    main()
