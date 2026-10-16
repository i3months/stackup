"""P-025 실행 규칙 점검: 재순위 배치 분할(RAG_RERANK_BATCH)이 기본(한 배치) 결과와 같은지.

    RAG_RERANK_BATCH=4 python rerank_batch_check.py [--n 40]   # → E2-retrieval/rerank-batch-check.json

dev 질의 일부(pool 전부 + 나머지 앞에서부터, 합 n 개)로 R-RR-1·R-RR-2 를 다시 검색해, 기존 dev run(배치 기본값)과
상위 k 순위(chunk_key 순서)가 같고 점수 차가 1e-4 이하인지 단언한다. 결과 파일은 쓰지 않고 비교만 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
import retrieve  # noqa: E402
from common import now_iso, results_dir, write_json  # noqa: E402
from labels import read_jsonl  # noqa: E402

TOL = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    assert retrieve.RERANK_BATCH, "RAG_RERANK_BATCH 를 지정한다"
    qs = retrieve.load_queries(results_dir("E1-queries") / "queries.jsonl", "dev")
    pick = [q for q in qs if q["qtype"] == "pool"]
    pick += [q for q in qs if q["qtype"] != "pool"][: max(a.n - len(pick), 0)]
    out = {"ts": now_iso(), "rerank_batch": retrieve.RERANK_BATCH, "k": a.k, "n_queries": len(pick), "tolerance": TOL,
           "conditions": {}}
    ok_all = True
    for cid in ("R-RR-1", "R-RR-2"):
        cond = C.load(cid)
        base = {r["query_id"]: r for r in read_jsonl(results_dir("E2-retrieval", "dev", f"{cid}-k{a.k}") / "rankings.jsonl")}
        keys = retrieve.load_chunk_keys("E1-labels", cond.variant_id)
        s = retrieve.Searcher(cond)
        n_same, max_diff, bad = 0, 0.0, []
        for q in pick:
            row = retrieve.ranking_row("check", q, a.k, s.search(q["persona_id"], q["query_text"], a.k), keys)
            b = base[q["query_id"]]
            same = [x["chunk_key"] for x in row["results"]] == [x["chunk_key"] for x in b["results"]]
            d = max((abs(x["score"] - y["score"]) for x, y in zip(row["results"], b["results"])), default=0.0)
            max_diff = max(max_diff, d)
            n_same += same
            if not same or d > TOL:
                bad.append({"query_id": q["query_id"], "same_order": same, "max_score_diff": d})
        s._reranker = None
        s.emb.unload()
        ok = not bad
        ok_all &= ok
        out["conditions"][cid] = {"same_order": n_same, "max_score_diff": max_diff, "pass": ok, "failures": bad[:20]}
        print(cid, json.dumps(out["conditions"][cid], ensure_ascii=False), flush=True)
    out["pass"] = ok_all
    write_json(results_dir("E2-retrieval") / "rerank-batch-check.json", out)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
