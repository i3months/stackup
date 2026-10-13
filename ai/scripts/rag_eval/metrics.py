"""검색 지표 (SPEC §6.1): nDCG@k, Fact Recall@k, Hit@k, MRR@10, judged@k.

    python metrics.py --run E2-retrieval/B0-k5-dev [--mrr-run E2-retrieval/B0-k10-dev] --labels-dir E1-labels

- 지표@k 는 그 k 실행의 순위로, MRR@10 은 k=10 실행의 순위로 계산한다 (서로 다른 실행이면 출력에 표시).
- IDCG@k 는 해당 변형·페르소나의 **전체 청크** 등급으로 만든다 (검색 결과와 무관).
- judged@k < 100% 인 질의가 하나라도 있으면 지표를 내지 않고 오류 (SPEC §4.3-3).
- 정답 없음 질의(gold 없음)는 지표에서 빼고 따로 집계한다.
- k=10 지표는 범위 내 청크 수 ≥ 20 인 질의만 집계한다 (SPEC §6.1 작은 페르소나 규칙).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import results_dir, write_json
from labels import Labels, read_jsonl


class NotFullyJudged(RuntimeError):
    pass


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[int], all_grades: list[int], k: int) -> float:
    ideal = dcg(sorted(all_grades, reverse=True)[:k])
    if ideal == 0:
        return float("nan")  # 관련 청크가 하나도 없는 (질의, 변형) — 평균에서 빠지고 개수 보고
    return dcg(ranked[:k]) / ideal


def fact_recall_at_k(top_keys: list[str], gold: list[str], labels: Labels) -> float:
    hit = sum(1 for f in gold if any(labels.label.get((c, f)) == "full" for c in top_keys))
    return hit / len(gold)


def hit_at_k(ranked: list[int], k: int) -> int:
    return int(any(g == 2 for g in ranked[:k]))


def mrr_at_10(ranked: list[int]) -> float:
    for i, g in enumerate(ranked[:10]):
        if g == 2:
            return 1.0 / (i + 1)
    return 0.0


def variant_chunk_keys(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_persona: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        by_persona[c["persona_id"]].append(c["chunk_key"])
    return by_persona


def per_query(rankings: list[dict[str, Any]], queries: dict[str, dict[str, Any]], labels: Labels,
              persona_chunks: dict[str, list[str]], mrr_rankings: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out = []
    for r in rankings:
        q = queries[r["query_id"]]
        k = r["k"]
        top = [x["chunk_key"] for x in r["results"]]
        judged = sum(labels.judged(c, q["persona_id"]) for c in top)
        row: dict[str, Any] = {"query_id": r["query_id"], "persona_id": q["persona_id"], "qtype": q.get("qtype"),
                               "k": k, "n_scope_chunks": r["n_scope_chunks"], "judged_at_k": judged / max(len(top), 1),
                               "no_answer": not q["gold_fact_ids"]}
        if judged < len(top):
            row["error"] = "not_fully_judged"
            out.append(row)
            continue
        if row["no_answer"]:
            row["top1_score"] = r["results"][0]["score"] if r["results"] else None
            row["max_grade"] = 0
            out.append(row)
            continue
        gold = q["gold_fact_ids"]
        ranked = [labels.rel(c, gold) for c in top]
        all_g = [labels.rel(c, gold) for c in persona_chunks[q["persona_id"]]]
        row.update(grades=ranked, ndcg=ndcg_at_k(ranked, all_g, k), fact_recall=fact_recall_at_k(top, gold, labels),
                   hit=hit_at_k(ranked, k), n_relevant_in_scope=sum(g > 0 for g in all_g))
        if mrr_rankings is not None and r["query_id"] in mrr_rankings:
            m = mrr_rankings[r["query_id"]]
            row["mrr10"] = mrr_at_10([labels.rel(x["chunk_key"], gold) for x in m["results"]])
        out.append(row)
    return out


def summarize(rows: list[dict[str, Any]], k: int, mrr_source: str | None) -> dict[str, Any]:
    bad = [r for r in rows if r.get("error")]
    if bad:
        raise NotFullyJudged(f"judged@k < 100% 인 질의 {len(bad)}개 — 포함 판정을 먼저 끝낸다 (예: {bad[0]['query_id']})")
    ans = [r for r in rows if not r["no_answer"]]
    if k == 10:
        ans = [r for r in ans if r["n_scope_chunks"] >= 20]
    def mean(key: str, rs: list[dict[str, Any]]) -> float | None:
        v = [r[key] for r in rs if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return round(statistics.fmean(v), 4) if v else None
    out = {
        "k": k, "n_queries": len(rows), "n_answerable": len(ans),
        f"ndcg@{k}": mean("ndcg", ans), f"fact_recall@{k}": mean("fact_recall", ans), f"hit@{k}": mean("hit", ans),
        "ndcg_undefined": sum(1 for r in ans if math.isnan(r.get("ndcg", 0.0))),
        "judged@k": 1.0,
        "mrr@10": mean("mrr10", ans), "mrr_source_run": mrr_source,
        "no_answer": {"n": sum(r["no_answer"] for r in rows),
                      "top1_scores": [r["top1_score"] for r in rows if r["no_answer"]]},
        "by_qtype": {t: {f"ndcg@{k}": mean("ndcg", [r for r in ans if r["qtype"] == t]),
                         "n": sum(1 for r in ans if r["qtype"] == t)}
                     for t in sorted({r["qtype"] for r in ans if r["qtype"]})},
    }
    if k == 10:
        out["excluded_small_scope"] = sum(1 for r in rows if not r["no_answer"] and r["n_scope_chunks"] < 20)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/ 상대 경로 (rankings.jsonl 이 있는 디렉터리)")
    ap.add_argument("--mrr-run", default=None, help="MRR@10 을 계산할 k=10 실행 (없으면 MRR 생략)")
    ap.add_argument("--labels-dir", default="E1-labels")
    ap.add_argument("--queries", default=str(results_dir("E1-queries") / "queries.jsonl"))
    ap.add_argument("--facts", default=str(results_dir("E1-corpus") / "facts.jsonl"))
    a = ap.parse_args()
    run_dir = results_dir(a.run)
    rankings = read_jsonl(run_dir / "rankings.jsonl")
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    variant = run_meta["config"]["variant_id"]
    labels = Labels.load(a.labels_dir, Path(a.facts))
    chunks = [c for c in read_jsonl(results_dir(a.labels_dir) / "chunks.jsonl") if c["variant_id"] == variant]
    queries = {q["query_id"]: q for q in read_jsonl(Path(a.queries))}
    mrr = None
    if a.mrr_run:
        mr = read_jsonl(results_dir(a.mrr_run) / "rankings.jsonl")
        assert all(r["k"] == 10 for r in mr), "MRR@10 은 k=10 실행에서 계산한다"
        mrr = {r["query_id"]: r for r in mr}
    rows = per_query(rankings, queries, labels, variant_chunk_keys(chunks), mrr)
    k = rankings[0]["k"]
    summ = summarize(rows, k, a.mrr_run)
    write_json(run_dir / "metrics.json", {"run_id": run_meta["run_id"], "condition_id": run_meta["condition_id"],
                                          "summary": summ, "per_query": rows})
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
