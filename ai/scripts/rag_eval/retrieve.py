"""검색: 조건 × 질의 × k → rankings.jsonl (SPEC §5.1, §9.2, §10).

    python retrieve.py --corpus E1 --condition B0 --split dev --k 5 [--queries path] [--out E2-retrieval]

- hybrid_simple·vector: 운영 JdbcDocumentEmbeddingRepository SQL 을 PG 에서 실행한다.
  하네스 SQL 은 동점 처리(점수 → doc_id → chunk_index)만 덧붙인다 (SPEC §10.4, P-013).
  `production_sql=True` 면 운영 SQL 원문(동점 처리 없음) — 동등성 검증용.
- 모든 검색은 트랜잭션 안에서 `SET LOCAL enable_indexscan = off` (전수 계산) 이고, 벡터 브랜치 행 수를 단언한다.
- hybrid_kiwi·bm25_kiwi: Kiwi 형태소 BM25 (파이썬). 하이브리드는 운영과 같은 RRF(k=60, 후보 = cand).
- 재순위화: 1단계 hybrid_simple cand = limit = 20 → 교차 인코더 → 상위 k. pre_rerank_rank 기록.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import conditions as C
from common import PG_DSN_DEFAULT, RRF_K, RunRecorder, append_jsonl, model_path, results_dir
from index import DEVICE, D2560_TABLE, Embedder, ensure_tables, vec_literal

# ---------------------------------------------------------------------------
_TIE_V = ", e.document_id, e.chunk_index"
_TIE_F = ", document_id, chunk_index"


def hybrid_sql(table: str, *, production_sql: bool = False) -> str:
    """운영 searchHybrid (filterByDoc=true). production_sql=False 면 각 ORDER BY 에 동점 처리를 덧붙인다."""
    tv = "" if production_sql else _TIE_V
    tf = "" if production_sql else ", document_id, chunk_index"
    return f"""
WITH tq AS (
    SELECT replace(plainto_tsquery('simple', %(qtext)s)::text, ' & ', ' | ')::tsquery AS tsq
),
v AS (
    SELECT e.document_id, e.chunk_index, e.chunk_text,
           (e.embedding <=> CAST(%(qvec)s AS vector)) AS distance,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> CAST(%(qvec)s AS vector){tv}) AS rnk
    FROM {table} e
    JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
    WHERE e.document_id = ANY(%(document_ids)s)
    ORDER BY e.embedding <=> CAST(%(qvec)s AS vector){tv}
    LIMIT %(cand)s
),
t AS (
    SELECT e.document_id, e.chunk_index, e.chunk_text,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) DESC{tv}
           ) AS rnk
    FROM {table} e
    JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
    WHERE e.chunk_text_tsv @@ (SELECT tsq FROM tq)
    AND e.document_id = ANY(%(document_ids)s)
    ORDER BY ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) DESC{tv}
    LIMIT %(cand)s
)
SELECT COALESCE(v.document_id, t.document_id) AS document_id,
       COALESCE(v.chunk_index, t.chunk_index) AS chunk_index,
       COALESCE(v.chunk_text, t.chunk_text)   AS chunk_text,
       COALESCE(v.distance, 1.0)              AS distance,
       COALESCE(1.0 / ({RRF_K} + v.rnk), 0) + COALESCE(1.0 / ({RRF_K} + t.rnk), 0) AS rrf,
       v.rnk AS vec_rank,
       t.rnk AS fts_rank
FROM v
FULL OUTER JOIN t
  ON v.document_id = t.document_id AND v.chunk_index = t.chunk_index
ORDER BY rrf DESC{tf}
LIMIT %(limit)s
"""


def vector_sql(table: str, *, production_sql: bool = False) -> str:
    """운영 searchVectorOnly (filterByDoc=true)."""
    tv = "" if production_sql else _TIE_V
    return f"""
SELECT e.document_id, e.chunk_index, e.chunk_text, (e.embedding <=> CAST(%(qvec)s AS vector)) AS distance
FROM {table} e
JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
WHERE e.document_id = ANY(%(document_ids)s)
ORDER BY e.embedding <=> CAST(%(qvec)s AS vector){tv}
LIMIT %(limit)s
"""


def vec_branch_sql(table: str) -> str:
    return f"""
SELECT e.document_id, e.chunk_index, e.chunk_text, (e.embedding <=> CAST(%(qvec)s AS vector)) AS distance
FROM {table} e
JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
WHERE e.document_id = ANY(%(document_ids)s)
ORDER BY e.embedding <=> CAST(%(qvec)s AS vector){_TIE_V}
LIMIT %(cand)s
"""


# ---- Kiwi BM25 ----------------------------------------------------------------
KIWI_POS = ("NNG", "NNP", "NNB", "NR", "SL", "SH", "SN", "XR", "VV", "VA", "MAG")
BM25_K1 = 1.2
BM25_B = 0.75
# 재순위 배치 크기 (P-025). 기본 None = CrossEncoder 기본(32). 동결 전 dev 단언(상위 k 순위 동일, 점수 차 ≤ 1e-4) 뒤에 정한다.
RERANK_BATCH = int(os.environ["RAG_RERANK_BATCH"]) if os.environ.get("RAG_RERANK_BATCH") else None
_kiwi = None


def kiwi_tokens(text: str) -> list[str]:
    """명사·외국어·숫자·한자·어근·용언 어간·일반 부사. 외국어는 소문자화 (PG simple 과 같은 대소문자 처리)."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi()
    out = []
    for t in _kiwi.tokenize(text):
        if t.tag in KIWI_POS:
            out.append(t.form.lower() if t.tag == "SL" else t.form)
    return out


class BM25:
    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {w: math.log((self.n - f + 0.5) / (f + 0.5) + 1.0) for w, f in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for w in q:  # 질의 토큰 중복은 그대로 (운영 OR tsquery 와 달리 BM25 표준)
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * len(d) / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------------------
class Searcher:
    def __init__(self, cond: C.Condition, dsn: str = PG_DSN_DEFAULT, *, production_sql: bool = False) -> None:
        import psycopg

        self.cond = cond
        self.production_sql = production_sql
        self.conn = psycopg.connect(dsn)
        self.emb = Embedder(cond)
        self.table = self.emb.table
        self.dim = 1536 if self.table == "document_embeddings" else 2560
        self._scope: dict[str, dict[str, Any]] = {}
        self._bm25: dict[str, tuple[BM25, list[tuple]]] = {}
        self._reranker = None
        with self.conn.cursor() as cur:
            ensure_tables(cur)
        self.conn.commit()

    def scope(self, persona_id: str) -> dict[str, Any]:
        """페르소나 = 사용자 범위. 그 인덱스 키의 DB 문서와 (db_id, part_chunk_index) → (doc_id, chunk_index) 매핑."""
        if persona_id not in self._scope:
            with self.conn.cursor() as cur:
                cur.execute("SELECT doc_id, part, first_chunk_index, db_document_id FROM rag_eval_docmap "
                            "WHERE index_key=%s AND persona_id=%s ORDER BY doc_id, first_chunk_index",
                            (self.cond.index_key, persona_id))
                rows = cur.fetchall()
                if not rows:
                    raise LookupError(f"{self.cond.index_key} / {persona_id} 인덱스 없음 — index.py 먼저")
                ids = [r[3] for r in rows]
                cur.execute(f"SELECT count(*) FROM {self.table} WHERE document_id = ANY(%s)", (ids,))
                n = cur.fetchone()[0]
            self.conn.commit()
            self._scope[persona_id] = {"db_ids": ids, "n_chunks": n,
                                       "map": {r[3]: (r[0], r[2]) for r in rows}}
        return self._scope[persona_id]

    def _key(self, sc: dict[str, Any], db_id: int, ci: int) -> tuple[str, int]:
        doc_id, first = sc["map"][db_id]
        return doc_id, first + ci

    def _bm25_for(self, persona_id: str) -> tuple[BM25, list[tuple]]:
        if persona_id not in self._bm25:
            sc = self.scope(persona_id)
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT document_id, chunk_index, chunk_text FROM {self.table} "
                            "WHERE document_id = ANY(%s) ORDER BY document_id, chunk_index", (sc["db_ids"],))
                rows = cur.fetchall()
            self.conn.commit()
            self._bm25[persona_id] = (BM25([kiwi_tokens(r[2]) for r in rows]), rows)
        return self._bm25[persona_id]

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(str(model_path(self.cond.reranker)), device=DEVICE)
        return self._reranker

    # ---- 한 질의 ----------------------------------------------------------------
    def search(self, persona_id: str, query_text: str, k: int) -> dict[str, Any]:
        cond = self.cond
        sc = self.scope(persona_id)
        cand, limit = cond.cand_for(k), cond.limit_for(k)
        lat: dict[str, float] = {}
        need_vec = cond.retrieval != "bm25_kiwi"
        qvec = None
        if need_vec:
            t0 = time.perf_counter()
            qvec = vec_literal(self.emb.encode([query_text], query=True)[0], self.dim)
            lat["embed"] = (time.perf_counter() - t0) * 1000
        params = {"qvec": qvec, "qtext": query_text, "document_ids": sc["db_ids"], "cand": cand, "limit": limit}
        t0 = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off")  # SPEC §2 전수 계산
            if cond.retrieval == "hybrid_simple":
                cur.execute(hybrid_sql(self.table, production_sql=self.production_sql), params)
                hits = [{"key": self._key(sc, r[0], r[1]), "text": r[2], "score": float(r[4]),
                         "distance": float(r[3]), "vec_rank": r[5], "fts_rank": r[6]} for r in cur.fetchall()]
                score_type = "rrf"
            elif cond.retrieval == "vector":
                cur.execute(vector_sql(self.table, production_sql=self.production_sql), params)
                hits = [{"key": self._key(sc, r[0], r[1]), "text": r[2], "score": 1.0 - float(r[3]),
                         "distance": float(r[3])} for r in cur.fetchall()]
                score_type = "cosine"
            else:
                bm, rows = self._bm25_for(persona_id)
                bs = bm.scores(kiwi_tokens(query_text))
                # BM25 브랜치: 점수 > 0 (운영 FTS 의 @@ 와 같은 '일치하는 청크만'), 점수 → doc_id → chunk_index
                order = sorted((i for i, s in enumerate(bs) if s > 0),
                               key=lambda i: (-bs[i], self._key(sc, rows[i][0], rows[i][1])))[:cand]
                if cond.retrieval == "bm25_kiwi":
                    hits = [{"key": self._key(sc, rows[i][0], rows[i][1]), "text": rows[i][2], "score": bs[i],
                             "bm25_rank": n + 1} for n, i in enumerate(order[:limit])]
                    score_type = "bm25"
                else:
                    cur.execute(vec_branch_sql(self.table), params)
                    vrows = cur.fetchall()
                    fused: dict[tuple, dict[str, Any]] = {}
                    for n, r in enumerate(vrows):
                        key = self._key(sc, r[0], r[1])
                        fused[key] = {"key": key, "text": r[2], "score": 1.0 / (RRF_K + n + 1), "vec_rank": n + 1,
                                      "bm25_rank": None, "distance": float(r[3])}
                    for n, i in enumerate(order):
                        key = self._key(sc, rows[i][0], rows[i][1])
                        h = fused.setdefault(key, {"key": key, "text": rows[i][2], "score": 0.0, "vec_rank": None,
                                                   "distance": 1.0})
                        h["score"] += 1.0 / (RRF_K + n + 1)
                        h["bm25_rank"] = n + 1
                    hits = sorted(fused.values(), key=lambda h: (-h["score"], h["key"]))[:limit]
                    score_type = "rrf"
            if need_vec:
                cur.execute(f"SELECT count(*) FROM ({vec_branch_sql(self.table)}) x", params)
                n_vec = cur.fetchone()[0]
                assert n_vec == min(cand, sc["n_chunks"]), f"벡터 브랜치 {n_vec} ≠ min({cand}, {sc['n_chunks']})"
        self.conn.commit()
        lat["search"] = (time.perf_counter() - t0) * 1000
        if cond.reranker != "none":
            t0 = time.perf_counter()
            rq = query_text
            if cond.rerank_query_tokens:  # R-RR-1-q512: 재순위 입력의 질의만 앞 N 토큰 (P-025)
                tok = self.reranker.tokenizer
                ids = tok(query_text, add_special_tokens=False)["input_ids"]
                if len(ids) > cond.rerank_query_tokens:
                    rq = tok.decode(ids[: cond.rerank_query_tokens], skip_special_tokens=True)
            # 배치 분할 (P-025 실행 규칙): CrossEncoder.predict 는 길이순으로 정렬해 batch_size 씩 나눈다. 기본 32 면
            # 후보 20쌍이 한 배치라 긴 pool 질의에서 GPU 메모리를 넘는다. RERANK_BATCH 로 나눈다 (None = 기본값 그대로).
            kw = {"batch_size": RERANK_BATCH} if RERANK_BATCH else {}
            scores = self.reranker.predict([(rq, h["text"]) for h in hits], **kw) if hits else []
            lat["rerank"] = (time.perf_counter() - t0) * 1000
            for n, (h, s) in enumerate(zip(hits, scores)):
                h["pre_rerank_rank"] = n + 1
                h["score"] = float(s)
            hits = sorted(hits, key=lambda h: (-h["score"], h["key"]))[:k]
            score_type = "rerank"
        return {"hits": hits, "score_type": score_type, "latency_ms": {k_: round(v, 2) for k_, v in lat.items()},
                "n_scope_chunks": sc["n_chunks"], "cand": cand}


def ranking_row(run_id: str, q: dict[str, Any], k: int, res: dict[str, Any], chunk_keys: dict[tuple, str]) -> dict[str, Any]:
    results = []
    for n, h in enumerate(res["hits"]):
        doc_id, ci = h["key"]
        r = {"rank": n + 1, "chunk_key": chunk_keys[(doc_id, ci)], "doc_id": doc_id, "chunk_index": ci,
             "score": h["score"]}
        if "pre_rerank_rank" in h:
            r["pre_rerank_rank"] = h["pre_rerank_rank"]
        for extra in ("vec_rank", "fts_rank", "bm25_rank"):
            if extra in h:
                r[extra] = h[extra]
        results.append(r)
    return {"run_id": run_id, "query_id": q["query_id"], "persona_id": q["persona_id"], "k": k,
            "score_type": res["score_type"], "cand": res["cand"], "n_scope_chunks": res["n_scope_chunks"],
            "results": results, "latency_ms": res["latency_ms"]}


def tie_stats(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    """질의별 동점 발생률: 상위 k 안에 같은 점수가 2개 이상인 질의 비율 (SPEC §10.4, P-013)."""
    n_tie = 0
    pairs = 0
    for r in rankings:
        scores = [round(x["score"], 12) for x in r["results"]]
        c = Counter(scores)
        if any(v > 1 for v in c.values()):
            n_tie += 1
        pairs += sum(v * (v - 1) // 2 for v in c.values())
    n = len(rankings)
    return {"n_queries": n, "queries_with_tie": n_tie, "tie_rate": round(n_tie / n, 4) if n else None,
            "tied_pairs": pairs}


def load_chunk_keys(out: str, variant_id: str) -> dict[tuple, str]:
    keys = {}
    for ln in (results_dir(out) / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if r["variant_id"] == variant_id:
            keys[(r["doc_id"], r["chunk_index"])] = r["chunk_key"]
    return keys


def run(cond: C.Condition, queries: list[dict[str, Any]], k: int, *, run_id: str, out_dir: Path, labels_dir: str,
        task_id: str, experiment: str, production_sql: bool = False, notes: str = "") -> list[dict[str, Any]]:
    model_ids = [cond.embedder] + ([cond.reranker] if cond.reranker != "none" else [])
    cfg = cond.as_dict() | {"k": k, "n_queries": len(queries), "production_sql": production_sql,
                            "labels_dir": labels_dir, "device": DEVICE, "rerank_batch": RERANK_BATCH, "bm25": {"k1": BM25_K1, "b": BM25_B, "pos": KIWI_POS}
                            if "kiwi" in cond.retrieval else None,
                            "llm_prefix_cache_prompt": ("요약 부분 false (T04), 원문 부분 true (P-026)" if cond.prefix == "llm"
                                                        else None)}
    keys = load_chunk_keys(labels_dir, cond.variant_id)
    with RunRecorder(out_dir, run_id=run_id, task_id=task_id, experiment=experiment, condition_id=cond.condition_id,
                     config=cfg, model_ids=model_ids, notes=notes) as rr:
        s = Searcher(cond, production_sql=production_sql)
        rows = []
        path = out_dir / "rankings.jsonl"
        path.unlink(missing_ok=True)  # run 단위로 다시 돈다 (끊긴 run 은 run.json ended_at 이 null)
        for q in queries:
            rows.append(ranking_row(run_id, q, k, s.search(q["persona_id"], q["query_text"], k), keys))
            append_jsonl(path, rows[-1:])
        s._reranker = None  # 재순위 모델도 GPU 에서 내린다 (emb.unload 가 gc·empty_cache)
        s.emb.unload()
        s.conn.close()
        rr.rec["config"]["ties"] = tie_stats(rows)
        rr.n_items = len(rows)
    return rows


def load_queries(path: Path, split: str | None) -> list[dict[str, Any]]:
    qs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if split:
        qs = [q for q in qs if q.get("split") == split]
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--queries", default=str(results_dir("E1-queries") / "queries.jsonl"))
    ap.add_argument("--labels-dir", default="E1-labels")
    ap.add_argument("--out", default="E2-retrieval")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--task-id", default="T05")
    a = ap.parse_args()
    cond = C.load(a.condition)
    run_id = a.run_id or f"{cond.condition_id}-k{a.k}-{a.split or 'all'}"
    out = results_dir(a.out, run_id)
    rows = run(cond, load_queries(Path(a.queries), a.split), a.k, run_id=run_id, out_dir=out, labels_dir=a.labels_dir,
               task_id=a.task_id, experiment=a.out.split("-")[0])
    print(json.dumps(tie_stats(rows), ensure_ascii=False), out)


if __name__ == "__main__":
    main()
