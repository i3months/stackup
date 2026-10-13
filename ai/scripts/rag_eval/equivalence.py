"""B0 운영 동등성 검증 + 결정성 + 예시 실행 (T02, SPEC §10-3·4).

    python equivalence.py all          # → results/E0-harness/

(a) 청크 바이트 일치: 운영 `chunk_embed_and_upsert`(가짜 임베더·가짜 Core 클라이언트로 페이로드를 가로챔)가
    만든 chunk_text 와 하네스 B0 변형의 (접두 + 본문)이 바이트 단위로 같은지. 문서 6개.
(b) 순위 일치: 같은 DB 데이터에서 운영 SQL 원문(동점 처리 없음)과 하네스 SQL(동점 처리 추가)의 상위 k.
    같은 RRF 점수 안의 순서 차이는 일치로 본다 (P-013). 그 밖의 불일치는 사례별로 원인을 남긴다.
(c) 결정성: 같은 조건을 2회 실행해 rankings 가 같은지.
질의 50개: ai/scripts/llm_eval/cases_v2.py 의 꼬리질문 40 (직전 질문 + "\\n\\n" + 답변) + 코칭 10 (질문 + "\\n" + 답변).
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import RunRecorder, results_dir, write_json, write_jsonl  # noqa: E402
from index import _cases_corpus, deterministic_prefix, index, variant_chunks  # noqa: E402
from retrieve import Searcher, load_chunk_keys, run, tie_stats  # noqa: E402

OUT = "E0-harness"
TASK = "T02"
K = 5


def cases_queries() -> list[dict[str, Any]]:
    import cases_v2  # noqa: PLC0415

    qs = []
    for c in cases_v2.FOLLOWUP_CASES:  # 운영 followup_consumer: f"{previous_question}\n\n{answer_text}"
        qs.append({"query_id": f"f-{c['id']}", "persona_id": "C00", "qtype": "followup",
                   "query_text": f"{c['previous_question']}\n\n{c['answer_text']}", "gold_fact_ids": []})
    for c in cases_v2.COACHING_CASES[:10]:  # 운영 feedback_consumer 코칭: f"{question}\n{answer}"
        qs.append({"query_id": f"c-{c['id']}", "persona_id": "C00", "qtype": "coaching",
                   "query_text": f"{c['question']}\n{c['answer']}", "gold_fact_ids": []})
    assert len(qs) == 50
    return qs


# ---- (a) ----------------------------------------------------------------------
class _FakeEmbedder:
    model = "fake"
    dim = 4

    async def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 1.0] for _ in texts]


class _FakeCore:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    async def upsert_embeddings(self, *, document_id: int, model: str, dim: int, chunks: list[Any]) -> int:
        self.payloads = chunks
        return len(chunks)


def check_chunks() -> dict[str, Any]:
    from ai_server.analyzer._embedding_step import chunk_embed_and_upsert
    from ai_server.config.settings import Settings
    from ai_server.rag.chunker import MarkdownChunker

    s = Settings(_env_file=None)
    cond = C.load("B0")
    assert (cond.chunk_size, cond.chunk_overlap) == (s.embedding_chunk_size, s.embedding_chunk_overlap)
    docs = _cases_corpus()
    rows = variant_chunks(cond, docs)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    report = []
    for d in docs:
        core = _FakeCore()
        n = asyncio.run(chunk_embed_and_upsert(
            document_id=1, markdown=d.analysis_md,
            chunker=MarkdownChunker(chunk_size=s.embedding_chunk_size, chunk_overlap=s.embedding_chunk_overlap),
            embedder=_FakeEmbedder(), core_client=core, summary=d.summary, log_prefix="equiv"))
        prod = [(p.chunk_index, p.chunk_text) for p in core.payloads]
        mine = [(r["part_chunk_index"], deterministic_prefix(r, d) + r["body"]) for r in by_doc[d.doc_id]]
        same = [a == b and a[1].encode() == b[1].encode() for a, b in zip(prod, mine)]
        report.append({"doc_id": d.doc_id, "n_prod": n, "n_harness": len(mine),
                       "byte_equal": len(prod) == len(mine) and all(same),
                       "bytes": sum(len(t.encode()) for _, t in prod)})
    return {"docs": report, "all_equal": all(r["byte_equal"] for r in report), "n_docs": len(report)}


# ---- (b) ----------------------------------------------------------------------
def _groups(hits: list[dict[str, Any]]) -> list[tuple[float, frozenset]]:
    out: list[tuple[float, set]] = []
    for h in hits:
        s = round(h["score"], 12)
        if out and out[-1][0] == s:
            out[-1][1].add(h["key"])
        else:
            out.append((s, {h["key"]}))
    return [(s, frozenset(k)) for s, k in out]


def compare(prod: list[dict[str, Any]], mine: list[dict[str, Any]]) -> str:
    if [h["key"] for h in prod] == [h["key"] for h in mine]:
        return "exact"
    gp, gm = _groups(prod), _groups(mine)
    if [s for s, _ in gp] == [s for s, _ in gm]:
        # 마지막 그룹은 LIMIT 으로 잘린 동점 그룹일 수 있어 크기만 같으면 된다
        if all(a == b for (_, a), (_, b) in zip(gp[:-1], gm[:-1])) and len(gp[-1][1]) == len(gm[-1][1]):
            return "tie_permutation"
    return "mismatch"


def explain(searcher: Searcher, q: dict[str, Any], prod: list[dict[str, Any]], mine: list[dict[str, Any]]) -> dict[str, Any]:
    """불일치 사례: 브랜치 원점수(거리·ts_rank_cd)를 보고, 차이가 브랜치 안 동점에서 왔는지 확인."""
    sc = searcher.scope(q["persona_id"])
    vec = searcher.emb.encode([q["query_text"]], query=True)[0]
    from index import vec_literal  # noqa: PLC0415

    with searcher.conn.cursor() as cur:
        cur.execute(
            f"""WITH tq AS (SELECT replace(plainto_tsquery('simple', %(q)s)::text, ' & ', ' | ')::tsquery AS tsq)
            SELECT e.document_id, e.chunk_index, (e.embedding <=> CAST(%(v)s AS vector)),
                   CASE WHEN e.chunk_text_tsv @@ (SELECT tsq FROM tq) THEN ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) END
            FROM {searcher.table} e WHERE e.document_id = ANY(%(ids)s)""",
            {"q": q["query_text"], "v": vec_literal(vec, searcher.dim), "ids": sc["db_ids"]})
        raw = cur.fetchall()
    searcher.conn.commit()
    dist = {(r[0], r[1]): round(r[2], 9) for r in raw}
    fts = {(r[0], r[1]): round(r[3], 9) for r in raw if r[3] is not None}

    def tie_range(scores: dict, key: Any, reverse: bool) -> tuple[int, int]:
        """그 원점수의 동점 그룹이 차지하는 순위 범위 [처음, 끝] (1부터)."""
        s = scores[key]
        better = sum(1 for v in scores.values() if (v > s if reverse else v < s))
        same = sum(1 for v in scores.values() if v == s)
        return better + 1, better + same

    inv = {(doc, ci): (db, pci) for db, (doc, first) in sc["map"].items() for pci in range(500) for ci in [first + pci]}
    ok = True
    for h in prod:  # 운영 결과의 브랜치 순위가 원점수 동점 그룹 범위 안에 있으면, 동점 순서만 다른 것이다
        db_key = inv[h["key"]]
        if h.get("vec_rank") is not None:
            lo, hi = tie_range(dist, db_key, reverse=False)
            ok &= lo <= h["vec_rank"] <= hi
        if h.get("fts_rank") is not None:
            lo, hi = tie_range(fts, db_key, reverse=True)
            ok &= lo <= h["fts_rank"] <= hi
    n_dist_ties = len(dist) - len(set(dist.values()))
    n_fts_ties = len(fts) - len(set(fts.values()))
    return {"query_id": q["query_id"],
            "prod": [(h["key"], round(h["score"], 6), h.get("vec_rank"), h.get("fts_rank")) for h in prod],
            "harness": [(h["key"], round(h["score"], 6), h.get("vec_rank"), h.get("fts_rank")) for h in mine],
            "distance_ties": n_dist_ties, "ts_rank_ties": n_fts_ties,
            "cause": "branch_tie_order" if ok else "unexplained"}


def check_ranking(queries: list[dict[str, Any]]) -> dict[str, Any]:
    cond = C.load("B0")
    prod_s = Searcher(cond, production_sql=True)
    mine_s = Searcher(cond)
    counts = {"exact": 0, "tie_permutation": 0, "mismatch": 0}
    cases = []
    for q in queries:
        p = prod_s.search(q["persona_id"], q["query_text"], K)["hits"]
        m = mine_s.search(q["persona_id"], q["query_text"], K)["hits"]
        verdict = compare(p, m)
        counts[verdict] += 1
        if verdict == "mismatch":
            cases.append(explain(mine_s, q, p, m))
    prod_s.emb.unload()
    mine_s.emb.unload()
    return {"k": K, "n_queries": len(queries), **counts,
            "match_rate": (counts["exact"] + counts["tie_permutation"]) / len(queries), "mismatches": cases}


# ---- (c) + 예시 실행 ------------------------------------------------------------
def example_runs(queries: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for cid in ("B0", "R-RET-hkiwi", "R-RR-1"):
        cond = C.load(cid)
        rid = f"{cid}-k{K}-cases"
        rows = run(cond, queries, K, run_id=rid, out_dir=results_dir(OUT, rid), labels_dir=OUT, task_id=TASK,
                   experiment="E0", notes="T02 예시 실행 (cases_v2 문서 6개·질의 50개, 라벨 없음 — 지표 미산출)")
        out[cid] = tie_stats(rows)
    # 결정성: B0 를 한 번 더 실행해 비교
    cond = C.load("B0")
    again = run(cond, queries, K, run_id="B0-k5-cases-rerun", out_dir=results_dir(OUT, "B0-k5-cases-rerun"),
                labels_dir=OUT, task_id=TASK, experiment="E0", notes="결정성 재실행")
    first = [json.loads(x) for x in (results_dir(OUT, f"B0-k{K}-cases") / "rankings.jsonl").read_text(
        encoding="utf-8").splitlines()]
    strip = lambda rs: [(r["query_id"], [(x["chunk_key"], x["doc_id"], x["chunk_index"], x["score"])
                                         for x in r["results"]]) for r in rs]  # noqa: E731
    out["determinism_B0_identical"] = strip(first) == strip(again)
    return out


def main() -> None:
    queries = cases_queries()
    write_jsonl(results_dir(OUT) / "queries.jsonl", queries)
    res: dict[str, Any] = {}
    with RunRecorder(results_dir(OUT), run_id="E0-harness-equivalence", task_id=TASK, experiment="E0",
                     condition_id="B0", config={"k": K, "corpus": "cases", "n_queries": len(queries)},
                     model_ids=["EMB-1", "RR-1"], filename="equivalence.run.json") as rr:
        res["chunks"] = check_chunks()
        for cid in ("B0", "R-RR-1", "R-RET-hkiwi"):  # 재순위·kiwi 는 B0 인덱스를 공유
            res.setdefault("index", {})[cid] = index(C.load(cid), "cases", OUT)
        res["ranking"] = check_ranking(queries)
        res["examples"] = example_runs(queries)
        rr.n_items = len(queries)
    write_json(results_dir(OUT) / "equivalence.json", res)
    print(json.dumps({"chunks_all_equal": res["chunks"]["all_equal"],
                      "ranking": {k: v for k, v in res["ranking"].items() if k != "mismatches"},
                      "examples": res["examples"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
