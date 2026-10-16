"""동점 민감도 (P-020 §5, SPEC §5.1·§10.4): 운영 SQL 이 동점 순서를 정하지 않을 때 나올 수 있는 nDCG@k 범위.

    python ties.py dev-check --condition B0 --k 5     # dev: 구현 검증만 → E2-retrieval/ties/<조건>-k<k>-dev.jsonl

운영 searchHybrid 에서 결과를 바꿀 수 있는 것은 세 가지다.
  1. 벡터 브랜치 ROW_NUMBER 의 거리 동점 그룹 안 순서 (그룹이 LIMIT cand 경계에 걸리면 어떤 청크가 들어가는지도)
  2. FTS 브랜치 ROW_NUMBER 의 ts_rank_cd 동점 그룹 안 순서 (같음)
  3. 최종 RRF 동점 그룹 안 순서와 LIMIT 경계

계산 방법:
- 브랜치 결과(1·2)는 경우의 수 = Π P(g, m) (g = 그룹 크기, m = 그 그룹이 상위 cand 안에서 차지하는 자리 수) 다.
  두 브랜치 경우의 수의 곱이 10⁴ 이하면 전부 열거하고, 넘으면 무작위 2,000회로 근사한다 (`sampled: true`).
- 최종 RRF 동점(3)은 열거하지 않고 해석적으로 푼다. 한 동점 그룹이 차지하는 자리 집합은 정해져 있으므로,
  그 안에서 등급을 내림차순으로 놓으면 DCG 최대, 오름차순이면 최소다 (재배열 부등식). 경계에 걸린 그룹은
  최대일 때 높은 등급부터, 최소일 때 낮은 등급부터 고른다. 그래서 브랜치 열거와 합치면 정확한 최솟값·최댓값이다.
- 벡터 단독은 브랜치 하나 = 최종 순서라 3 과 같은 방법으로 바로 정확하다.
- 등급순 정렬(두 브랜치 모두 동점 안에서 관련 청크를 앞/뒤로)은 RRF 합 때문에 최댓값·최솟값을 보장하지 않는다.
  검증용으로만 같이 계산해 열거와 어긋나는 질의 수를 센다 (`gradesort_*`).
"""

from __future__ import annotations

import itertools
import zlib
import math
import random
from collections.abc import Hashable, Iterator, Sequence
from typing import Any

EXACT_MAX = 10_000
N_SAMPLES = 2_000
RRF_K = 60


# ---- 순위 조각 ---------------------------------------------------------------------
def tie_groups(scores: dict[Hashable, float], cand: int) -> list[tuple[list[Hashable], int]]:
    """점수 내림차순 동점 그룹 중 상위 cand 에 걸치는 것만. (그룹 원소, 상위 cand 안 자리 수). 원소는 정렬해 둔다."""
    order = sorted(scores, key=lambda x: (-scores[x], str(x)))
    out, i = [], 0
    while i < len(order) and i < cand:
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        out.append((order[i:j], min(j, cand) - i))
        i = j
    return out


def n_outcomes(groups: list[tuple[list[Hashable], int]]) -> int:
    return math.prod(math.perm(len(g), m) for g, m in groups)


def outcomes(groups: list[tuple[list[Hashable], int]]) -> Iterator[list[Hashable]]:
    """브랜치가 낼 수 있는 상위 cand 순서 전부."""
    for parts in itertools.product(*(itertools.permutations(g, m) for g, m in groups)):
        yield [x for p in parts for x in p]


def sample_outcome(groups: list[tuple[list[Hashable], int]], rng: random.Random) -> list[Hashable]:
    out = []
    for g, m in groups:
        out += rng.sample(g, m)
    return out


def rrf(branches: Sequence[list[Hashable]]) -> dict[Hashable, float]:
    s: dict[Hashable, float] = {}
    for br in branches:
        for n, x in enumerate(br):
            s[x] = s.get(x, 0.0) + 1.0 / (RRF_K + n + 1)
    return s


# ---- 지표 -------------------------------------------------------------------------
def dcg(grades: Sequence[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def final_minmax(scores: dict[Hashable, float], grade: dict[Hashable, int], limit: int) -> tuple[list[int], list[int]]:
    """최종 정렬의 동점을 해석적으로: (최대 DCG 등급열, 최소 DCG 등급열), 길이 ≤ limit."""
    best, worst = [], []
    for g, m in tie_groups(scores, limit):
        gs = sorted((grade.get(x, 0) for x in g), reverse=True)
        best += gs[:m]
        worst += sorted(gs)[:m]
    return best, worst


def ndcg(grades: Sequence[int], ideal: float, k: int) -> float:
    return dcg(list(grades)[:k]) / ideal if ideal > 0 else 0.0


def ideal_dcg(all_grades: Sequence[int], k: int) -> float:
    return dcg(sorted(all_grades, reverse=True)[:k])


# ---- 한 질의 -----------------------------------------------------------------------
def hybrid_range(vec: dict[Hashable, float], fts: dict[Hashable, float], grade: dict[Hashable, int],
                 all_grades: Sequence[int], *, cand: int, limit: int, k: int, seed: int = 0) -> dict[str, Any]:
    """vec·fts 는 '클수록 위' 점수 (거리는 음수로 넘긴다). fts 는 @@ 에 걸린 청크만."""
    ideal = ideal_dcg(all_grades, k)
    gv, gt = tie_groups(vec, cand), tie_groups(fts, cand)
    n_v, n_t = n_outcomes(gv), n_outcomes(gt)
    n_branch = n_v * n_t
    sampled = n_branch > EXACT_MAX
    if sampled:
        rng = random.Random(seed)
        pairs: Iterator[tuple[list, list]] = ((sample_outcome(gv, rng), sample_outcome(gt, rng))
                                               for _ in range(N_SAMPLES))
    else:
        pairs = ((a, b) for a in outcomes(gv) for b in list(outcomes(gt)))
    lo, hi, n_final_tied = math.inf, -math.inf, 0
    for a, b in pairs:
        s = rrf([a, b])
        best, worst = final_minmax(s, grade, limit)
        hi = max(hi, ndcg(best, ideal, k))
        lo = min(lo, ndcg(worst, ideal, k))
        n_final_tied += best != worst
    gs = gradesort(vec, fts, grade, all_grades, cand=cand, limit=limit, k=k)
    return {"ndcg_min": lo, "ndcg_max": hi, "width": hi - lo, "n_branch_outcomes": n_branch,
            "n_vec_outcomes": n_v, "n_fts_outcomes": n_t, "sampled": sampled,
            "n_evaluated": N_SAMPLES if sampled else n_branch} | gs


def single_range(scores: dict[Hashable, float], grade: dict[Hashable, int], all_grades: Sequence[int], *,
                 limit: int, k: int) -> dict[str, Any]:
    """브랜치 하나 (벡터 단독·BM25 단독): 최종 순서 = 브랜치 순서라 해석적으로 정확하다."""
    ideal = ideal_dcg(all_grades, k)
    best, worst = final_minmax(scores, grade, limit)
    lo, hi = ndcg(worst, ideal, k), ndcg(best, ideal, k)
    return {"ndcg_min": lo, "ndcg_max": hi, "width": hi - lo, "n_branch_outcomes": n_outcomes(tie_groups(scores, limit)),
            "sampled": False, "n_evaluated": 1}


def gradesort(vec: dict[Hashable, float], fts: dict[Hashable, float], grade: dict[Hashable, int],
              all_grades: Sequence[int], *, cand: int, limit: int, k: int) -> dict[str, float]:
    """검증용: 두 브랜치 모두 동점 안에서 등급 높은 청크를 앞(최대)/뒤(최소)로 두고, 최종 동점도 같은 방향."""
    ideal = ideal_dcg(all_grades, k)
    out = {}
    for name, sign in (("gradesort_max", -1), ("gradesort_min", 1)):
        brs = []
        for sc in (vec, fts):
            order = sorted(sc, key=lambda x: (-sc[x], sign * grade.get(x, 0), str(x)))[:cand]
            brs.append(order)
        s = rrf(brs)
        best, worst = final_minmax(s, grade, limit)
        out[name] = ndcg(best if sign < 0 else worst, ideal, k)
    return out


# ---- DB 에서 한 질의의 브랜치 점수 전부 --------------------------------------------------
def _branch_scores_sql(table: str) -> str:
    return f"""
WITH tq AS (SELECT replace(plainto_tsquery('simple', %(qtext)s)::text, ' & ', ' | ')::tsquery AS tsq)
SELECT e.document_id, e.chunk_index,
       (e.embedding <=> CAST(%(qvec)s AS vector)) AS distance,
       CASE WHEN e.chunk_text_tsv @@ (SELECT tsq FROM tq)
            THEN ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) END AS fts
FROM {table} e
JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
WHERE e.document_id = ANY(%(document_ids)s)
"""


def query_range(searcher: Any, q: dict[str, Any], k: int, labels: Any, chunk_keys: dict[tuple, str],
                all_keys: list[str]) -> dict[str, Any]:
    """운영 SQL 과 같은 식으로 계산한 점수 전부로 그 질의의 nDCG@k 범위. 벡터 점수는 -거리 (클수록 위)."""
    from index import vec_literal  # noqa: PLC0415

    cond = searcher.cond
    sc = searcher.scope(q["persona_id"])
    qvec = vec_literal(searcher.emb.encode([q["query_text"]], query=True)[0], searcher.dim)
    with searcher.conn.cursor() as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute(_branch_scores_sql(searcher.table), {"qvec": qvec, "qtext": q["query_text"],
                                                         "document_ids": sc["db_ids"]})
        rows = cur.fetchall()
    searcher.conn.commit()
    vec, fts = {}, {}
    for db_id, ci, dist, fr in rows:
        key = chunk_keys[searcher._key(sc, db_id, ci)]
        vec[key] = -float(dist)
        if fr is not None:
            fts[key] = float(fr)
    gold = q["gold_fact_ids"]
    grade = {c: labels.rel(c, gold) for c in vec} if gold else {}
    all_g = [labels.rel(c, gold) for c in all_keys] if gold else [0] * len(all_keys)
    cand, limit = cond.cand_for(k), cond.limit_for(k)
    if cond.retrieval == "hybrid_simple":
        r = hybrid_range(vec, fts, grade, all_g, cand=cand, limit=limit, k=k, seed=zlib.crc32(q["query_id"].encode()))
    elif cond.retrieval == "vector":
        r = single_range(vec, grade, all_g, limit=limit, k=k)
    else:
        raise ValueError(f"{cond.retrieval}: 동점 범위는 운영 SQL 경로(hybrid_simple·vector)만")
    return {"query_id": q["query_id"], "no_answer": not gold, **r}


def cmd_dev_check(condition_id: str, k: int) -> None:
    """dev 구현 검증: 하네스 순위(동점 처리 doc_id·chunk_index)의 nDCG 가 [min, max] 안에 있는지, 등급순 정렬과
    열거가 어긋나는 질의 수, 근사한 질의 비율."""
    import json  # noqa: PLC0415

    import conditions as C  # noqa: PLC0415
    from common import append_jsonl, results_dir, write_json  # noqa: PLC0415
    from labels import Labels, read_jsonl  # noqa: PLC0415
    from retrieve import Searcher, load_chunk_keys, load_queries  # noqa: PLC0415

    cond = C.load(condition_id)
    labels = Labels.load("E1-labels", results_dir("E1-corpus") / "facts.jsonl")
    keys = load_chunk_keys("E1-labels", cond.variant_id)
    by_p: dict[str, list[str]] = {}
    for (doc_id, _), ck in keys.items():
        by_p.setdefault(doc_id.split("-")[0], []).append(ck)
    qs = load_queries(results_dir("E1-queries") / "queries.jsonl", "dev")
    harness = {r["query_id"]: r for r in json.loads(
        (results_dir("E2-retrieval", "dev", f"{condition_id}-k{k}") / "metrics.json").read_text(encoding="utf-8"))["per_query"]}
    out = results_dir("E2-retrieval", "ties") / f"{condition_id}-k{k}-dev.jsonl"
    done = {r["query_id"] for r in read_jsonl(out)} if out.exists() else set()
    s = Searcher(cond)
    for q in qs:
        if q["query_id"] in done:
            continue
        r = query_range(s, q, k, labels, keys, by_p[q["persona_id"]])
        h = harness[q["query_id"]].get("ndcg")
        r["ndcg_harness"] = h
        r["harness_in_range"] = None if h is None else (r["ndcg_min"] - 1e-12 <= h <= r["ndcg_max"] + 1e-12)
        append_jsonl(out, [r])
    s.emb.unload()
    rows = [r for r in read_jsonl(out) if not r["no_answer"]]
    w = sorted(r["width"] for r in rows)
    summ = {"condition": condition_id, "k": k, "split": "dev", "n_answerable": len(rows),
            "harness_in_range": sum(bool(r["harness_in_range"]) for r in rows),
            "with_range": sum(r["width"] > 1e-12 for r in rows),
            "sampled": sum(r["sampled"] for r in rows), "sampled_rate": round(sum(r["sampled"] for r in rows) / len(rows), 4),
            "max_branch_outcomes": max(r["n_branch_outcomes"] for r in rows),
            "gradesort_mismatch": sum(1 for r in rows if "gradesort_max" in r and (
                abs(r["gradesort_max"] - r["ndcg_max"]) > 1e-9 or abs(r["gradesort_min"] - r["ndcg_min"]) > 1e-9)),
            "width_quantiles": [round(w[int(p * (len(w) - 1))], 4) for p in (0, .5, .75, .9, 1)],
            "mean_ndcg_min": round(sum(r["ndcg_min"] for r in rows) / len(rows), 4),
            "mean_ndcg_max": round(sum(r["ndcg_max"] for r in rows) / len(rows), 4),
            "mean_ndcg_harness": round(sum(r["ndcg_harness"] for r in rows) / len(rows), 4)}
    write_json(results_dir("E2-retrieval", "ties") / f"{condition_id}-k{k}-dev-summary.json", summ)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev-check"])
    ap.add_argument("--condition", default="B0")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    cmd_dev_check(a.condition, a.k)
