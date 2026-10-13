"""19개 조건 전부를 소형 코퍼스(cases)로 인덱싱·검색해 실행되는지 확인한다 (T02 완료 기준 1·6).

    python all_conditions.py [--device cpu] → results/E0-harness/all-conditions.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import results_dir, write_json  # noqa: E402
from equivalence import OUT, TASK, cases_queries  # noqa: E402
from index import index  # noqa: E402
from retrieve import run, tie_stats  # noqa: E402

SCHEMA_KEYS = {"run_id", "query_id", "k", "score_type", "results", "latency_ms"}
RESULT_KEYS = {"rank", "chunk_key", "doc_id", "chunk_index", "score"}


def main() -> None:
    qs = cases_queries()
    out = {}
    for cid, cond in C.load_all().items():
        t0 = time.monotonic()
        try:
            info = index(cond, "cases", OUT)
            rid = f"all-{cid}-k5"
            rows = run(cond, qs, 5, run_id=rid, out_dir=results_dir(OUT, "all", rid), labels_dir=OUT, task_id=TASK,
                       experiment="E0", notes="조건 실행 확인 (cases)")
            assert all(SCHEMA_KEYS <= r.keys() and all(RESULT_KEYS <= x.keys() for x in r["results"]) for r in rows)
            if cond.reranker != "none":
                assert all("pre_rerank_rank" in x for r in rows for x in r["results"])
            out[cid] = {"ok": True, "table": info["table"], "n_chunks": info["n_chunks"], "ties": tie_stats(rows),
                        "mean_latency_ms": {k: round(sum(r["latency_ms"].get(k, 0) for r in rows) / len(rows), 1)
                                            for k in ("embed", "search", "rerank")},
                        "sec": round(time.monotonic() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            out[cid] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
        print(cid, json.dumps({k: v for k, v in out[cid].items() if k != "trace"}, ensure_ascii=False), flush=True)
    write_json(results_dir(OUT) / "all-conditions.json", out)


if __name__ == "__main__":
    main()
