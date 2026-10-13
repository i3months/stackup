"""E0 스모크 테스트 6종 (T01, SPEC §7 E0 관문).

    python smoke.py chunk              # 1. 운영 청커 + 운영 접두로 이력서 1개 청킹
    python smoke.py emb1               # 2. EMB-1 → 0 패딩 → 운영 document_embeddings + 하이브리드 SQL
    python smoke.py emb4               # 3. EMB-4 → document_embeddings_d2560 + 같은 SQL
    python smoke.py rerank             # 4. RR-1 재순위화 1건 (1단계 cand = limit = 20)
    python smoke.py generate --profile M-S   # 5. 로컬 생성 모델로 운영 꼬리질문 체인 1건
    python smoke.py generate --profile M-M
    python smoke.py claude             # 6. Haiku 5.5 일반 1건(구조화 출력) + Batch 1건, 원장 2줄 확인

결과: stackup-lab/results/E0-env/smoke/<번호>-<이름>.jsonl + .run.json (SPEC §9.3).
입력은 전부 ai/scripts/llm_eval/cases*.py 의 합성 데이터다. 운영 DB·로그는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 고정 리비전의 로컬 사본만 쓴다
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODELS,
    PG_DSN_DEFAULT,
    PROD_EMBED_DIM,
    RRF_K,
    RunRecorder,
    model_path,
    results_dir,
    write_jsonl,
)

TASK_ID = "T01"
EXP = "E0"
CASE_RESUME = "RESUME_BACKEND"  # cases.py (cases_v2 가 import 해 재사용하는 백엔드 이력서)
CASE_FOLLOWUP = "f-strong-backend"
TOP_K = 5  # 운영 꼬리질문 k (cand = topK)
RERANK_CAND = 20  # SPEC §5.1 재순위화 1단계 cand = limit = 20


def smoke_dir() -> Path:
    return results_dir(EXP + "-env", "smoke")


def recorder(name: str, **kw: Any) -> RunRecorder:
    return RunRecorder(
        smoke_dir(), run_id=f"E0-smoke-{name}", task_id=TASK_ID, experiment=EXP,
        filename=f"{name}.run.json", **kw,
    )


# ---------------------------------------------------------------------------
# 운영 하이브리드 검색 SQL (JdbcDocumentEmbeddingRepository.searchHybrid, filterByDoc=true).
# 차이: 바인딩 문법(`IN (:documentIds)` → `= ANY(%(document_ids)s)`), 테이블 이름 치환(SPEC §2 차원 규칙),
# 진단용 vec_rank·fts_rank 열 추가. 순위 계산식은 같다.
def hybrid_sql(table: str) -> str:
    return f"""
WITH tq AS (
    SELECT replace(plainto_tsquery('simple', %(qtext)s)::text, ' & ', ' | ')::tsquery AS tsq
),
v AS (
    SELECT e.document_id, e.chunk_index, e.chunk_text,
           (e.embedding <=> CAST(%(qvec)s AS vector)) AS distance,
           ROW_NUMBER() OVER (ORDER BY e.embedding <=> CAST(%(qvec)s AS vector)) AS rnk
    FROM {table} e
    JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
    WHERE e.document_id = ANY(%(document_ids)s)
    ORDER BY e.embedding <=> CAST(%(qvec)s AS vector)
    LIMIT %(cand)s
),
t AS (
    SELECT e.document_id, e.chunk_index, e.chunk_text,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) DESC
           ) AS rnk
    FROM {table} e
    JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
    WHERE e.chunk_text_tsv @@ (SELECT tsq FROM tq)
    AND e.document_id = ANY(%(document_ids)s)
    ORDER BY ts_rank_cd(e.chunk_text_tsv, (SELECT tsq FROM tq)) DESC
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
ORDER BY rrf DESC
LIMIT %(limit)s
"""


def vec_branch_count_sql(table: str) -> str:
    return f"""
SELECT count(*) FROM (
    SELECT e.id FROM {table} e
    JOIN analyzed_documents d ON d.id = e.document_id AND d.is_deleted = FALSE
    WHERE e.document_id = ANY(%(document_ids)s)
    ORDER BY e.embedding <=> CAST(%(qvec)s AS vector)
    LIMIT %(cand)s
) x
"""


D2560_DDL = """
CREATE TABLE IF NOT EXISTS document_embeddings_d2560 (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES analyzed_documents (id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(2560) NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', chunk_text)) STORED,
    CONSTRAINT uq_document_embeddings_d2560_document_chunk UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_d2560_document_id
    ON document_embeddings_d2560 (document_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_d2560_tsv
    ON document_embeddings_d2560 USING GIN (chunk_text_tsv);
-- HNSW 없음: pgvector HNSW 는 2000차원까지 (SPEC §2).
"""

COLUMNS_SQL = """
SELECT column_name, data_type, is_nullable, column_default IS NOT NULL, is_generated
FROM information_schema.columns WHERE table_name = %s ORDER BY column_name
"""


# ---------------------------------------------------------------------------
def _resume_text() -> str:
    import cases_v2  # noqa: PLC0415  (common.py 가 llm_eval 경로를 sys.path 에 넣는다)

    return getattr(cases_v2, CASE_RESUME)


def _followup_case() -> dict[str, Any]:
    import cases_v2  # noqa: PLC0415

    return next(c for c in cases_v2.FOLLOWUP_CASES if c["id"] == CASE_FOLLOWUP)


def _query_text() -> str:
    c = _followup_case()
    return f"{c['previous_question']}\n\n{c['answer_text']}"  # 운영 꼬리질문 질의 형식


def _settings(**overrides: Any) -> Any:
    from ai_server.config.settings import Settings

    # .env 를 읽지 않는다 — 운영 키·엔드포인트가 섞이지 않게.
    return Settings(_env_file=None, **overrides)


def _summary_stand_in(markdown: str) -> str:
    """운영은 분석 LLM 이 만든 summary 를 접두에 쓴다. 스모크에서는 LLM 없이
    이력서 '## 요약' 절의 불릿을 이어붙여 대신한다 (E1a 에서 M-M 분석 요약으로 교체)."""
    lines, on = [], False
    for ln in markdown.splitlines():
        if ln.startswith("## "):
            on = ln.strip() == "## 요약"
            continue
        if on and ln.strip().startswith("-"):
            lines.append(ln.strip()[1:].strip())
    return " ".join(lines)


def contextualized_chunks() -> tuple[list[Any], list[str], str]:
    from ai_server.analyzer._embedding_step import _contextualize
    from ai_server.rag.chunker import MarkdownChunker

    s = _settings()
    md = _resume_text()
    chunker = MarkdownChunker(chunk_size=s.embedding_chunk_size, chunk_overlap=s.embedding_chunk_overlap)
    chunks = chunker.split(md)
    summary = _summary_stand_in(md)
    ctx = [_contextualize(c.text, c.heading_path, summary) for c in chunks]
    return chunks, ctx, summary


# ---------------------------------------------------------------------------
def cmd_chunk(_: argparse.Namespace) -> None:
    import hashlib

    s = _settings()
    cfg = {
        "case": CASE_RESUME,
        "chunk_size": s.embedding_chunk_size,
        "chunk_overlap": s.embedding_chunk_overlap,
        "summary_source": "resume '## 요약' bullets (stand-in for LLM summary)",
    }
    with recorder("1-chunk", condition_id="B0-chunking", config=cfg) as rr:
        chunks, ctx, summary = contextualized_chunks()
        rows = [
            {
                "variant_id": "smoke-b0",
                "doc_id": CASE_RESUME,
                "chunk_index": c.index,
                "chunk_key": hashlib.sha256(c.text.encode()).hexdigest(),
                "heading_path": c.heading_path,
                "body_chars": len(c.text),
                "prefix_text": ctx[i][: len(ctx[i]) - len(c.text)],
                "body": c.text,
            }
            for i, c in enumerate(chunks)
        ]
        write_jsonl(smoke_dir() / "1-chunk.jsonl", rows)
        rr.n_items = len(rows)
    print(f"chunks={len(chunks)} summary_chars={len(summary)}")
    for r in rows:
        print(f"  #{r['chunk_index']} [{r['heading_path']}] body {r['body_chars']}자")
    assert chunks, "청크 0개"
    assert all(ctx[i].endswith(c.text) for i, c in enumerate(chunks)), "접두 + 본문 형식 불일치"


# ---------------------------------------------------------------------------
def _pad(vec: list[float], dim: int) -> list[float]:
    # 0 패딩은 내적·노름을 바꾸지 않으므로 코사인 거리가 원래 차원과 같다 (SPEC §2).
    if len(vec) > dim:
        raise ValueError(f"임베딩 차원 {len(vec)} > 컬럼 차원 {dim}")
    return list(vec) + [0.0] * (dim - len(vec))


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _seed_document(cur: Any, variant_id: str, summary: str) -> int:
    """(variant_id, doc_id) → DB document_id. 같은 변형을 다시 넣으면 이전 행을 지운다 (CASCADE)."""
    path = f"rag-eval/{variant_id}/{CASE_RESUME}.md"
    cur.execute("DELETE FROM analyzed_documents WHERE document_path = %s", (path,))
    cur.execute(
        # 합성 사용자 1명. uq_users_github_id_active 가 부분 유니크 인덱스라 ON CONFLICT 에 WHERE 를 맞춘다.
        "INSERT INTO users (github_id, github_username, encrypted_github_access_token, display_name) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (github_id) WHERE is_deleted = false "
        "DO UPDATE SET github_username = EXCLUDED.github_username RETURNING id",
        (900000001, "rag-eval-synthetic", "none", "rag-eval-synthetic"),
    )
    uid = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO resumes (user_id, original_filename, file_path, file_type, status) "
        "VALUES (%s, %s, %s, 'PDF', 'ANALYZED') RETURNING id",
        (uid, f"{variant_id}-{CASE_RESUME}.pdf", f"rag-eval/{variant_id}"),
    )
    rid = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO analyzed_documents (resume_id, document_path, summary) VALUES (%s, %s, %s) RETURNING id",
        (rid, path, summary[:2000]),
    )
    return cur.fetchone()[0]


def _load_embedder(mid: str) -> tuple[Any, str]:
    import torch
    from sentence_transformers import SentenceTransformer

    # 4B 는 fp32 로 12GB 에 안 들어가 bf16. 나머지는 fp32.
    dtype = torch.bfloat16 if mid == "EMB-4" else torch.float32
    m = SentenceTransformer(str(model_path(mid)), device="cuda", model_kwargs={"dtype": dtype})
    return m, str(dtype).replace("torch.", "")


def _encode(model: Any, texts: list[str], *, query: bool) -> Any:
    kw: dict[str, Any] = {"normalize_embeddings": True, "batch_size": 8}
    if query and "query" in (getattr(model, "prompts", None) or {}):
        kw["prompt_name"] = "query"  # Qwen3-Embedding 등 모델 카드 기본 질의 지시문
    return model.encode(texts, **kw)


def search(cur: Any, table: str, qvec: str, qtext: str, document_ids: list[int], cand: int, limit: int,
           n_in_scope: int) -> tuple[list[Any], dict[str, Any]]:
    """SPEC §2: 전수 계산 강제 + 벡터 브랜치 행 수 단언. (cur 는 트랜잭션 안이어야 SET LOCAL 이 유효)"""
    params = {"qvec": qvec, "qtext": qtext, "document_ids": document_ids, "cand": cand, "limit": limit}
    cur.execute("SET LOCAL enable_indexscan = off")
    cur.execute("EXPLAIN (FORMAT JSON) " + hybrid_sql(table), params)
    plan = json.dumps(cur.fetchone()[0])
    t0 = time.perf_counter()
    cur.execute(hybrid_sql(table), params)
    hits = cur.fetchall()
    sql_ms = (time.perf_counter() - t0) * 1000
    cur.execute(vec_branch_count_sql(table), params)
    n_vec = cur.fetchone()[0]
    assert n_vec == min(cand, n_in_scope), f"벡터 브랜치 {n_vec}행 ≠ min({cand}, {n_in_scope})"
    diag = {
        "sql_ms": round(sql_ms, 2),
        "vec_branch_rows": n_vec,
        "plan_uses_hnsw_index": "hnsw" in plan,
        "plan_index_scans": plan.count('"Index Scan"'),
    }
    assert not diag["plan_uses_hnsw_index"], "HNSW 인덱스가 계획에 들어갔다"
    return hits, diag


def _embed_and_search(mid: str, table: str, dim_col: int) -> None:
    import numpy as np
    import psycopg

    variant = f"smoke-{mid.lower()}"
    name = {"EMB-1": "2-emb1", "EMB-4": "3-emb4"}[mid]
    cfg = {"embed_model": mid, "table": table, "column_dim": dim_col, "variant_id": variant,
           "cand": TOP_K, "limit": TOP_K, "query_case": CASE_FOLLOWUP, "enable_indexscan": "off"}
    with recorder(name, condition_id=f"smoke-{mid}", config=cfg, model_ids=[mid]) as rr:
        chunks, ctx, summary = contextualized_chunks()
        query = _query_text()
        t0 = time.perf_counter()
        model, dtype = _load_embedder(mid)
        load_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        doc_vecs = _encode(model, ctx, query=False)
        q_vec = _encode(model, [query], query=True)[0]
        embed_sec = time.perf_counter() - t0
        dim = int(doc_vecs.shape[1])
        assert dim == MODELS[mid]["dim"], f"{mid} 차원 {dim}"

        with psycopg.connect(PG_DSN_DEFAULT) as conn, conn.cursor() as cur:
            if table != "document_embeddings":
                cur.execute(D2560_DDL)
                cur.execute(COLUMNS_SQL, ("document_embeddings",))
                prod_cols = cur.fetchall()
                cur.execute(COLUMNS_SQL, (table,))
                assert cur.fetchall() == prod_cols, "실험 테이블 컬럼이 운영과 다르다"
            doc_id = _seed_document(cur, variant, summary)
            for c, text, vec in zip(chunks, ctx, doc_vecs):
                cur.execute(
                    f"INSERT INTO {table} (document_id, chunk_index, chunk_text, embedding, model) "
                    "VALUES (%s, %s, %s, CAST(%s AS vector), %s)",
                    (doc_id, c.index, text, _vec_literal(_pad(vec.tolist(), dim_col)), MODELS[mid]["repo"]),
                )
            hits, diag = search(cur, table, _vec_literal(_pad(q_vec.tolist(), dim_col)), query,
                                [doc_id], TOP_K, TOP_K, len(chunks))
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            pgv = cur.fetchone()[0]
            cur.execute("SHOW server_version")
            pg_ver = cur.fetchone()[0]
            conn.commit()

        # 내적이 아니라 코사인으로 비교한다 — bf16 로 정규화한 벡터는 노름이 1 에서 ~1e-3 벗어난다.
        d64 = np.asarray(doc_vecs, dtype=np.float64)
        q64 = np.asarray(q_vec, dtype=np.float64)
        sims = (d64 @ q64) / (np.linalg.norm(d64, axis=1) * np.linalg.norm(q64))
        norm_dev = float(np.max(np.abs(np.linalg.norm(d64, axis=1) - 1.0)))
        max_abs_diff = max(
            (abs((1 - float(sims[h[1]])) - float(h[3])) for h in hits if h[5] is not None), default=0.0
        )
        rr.rec["config"].update(embed_dim=dim, dtype=dtype, document_id=doc_id, postgres=pg_ver, pgvector=pgv)
        meta = {"query": query, "model_load_sec": round(load_sec, 2), "embed_sec": round(embed_sec, 3),
                "cosine_max_abs_diff_numpy_vs_pg": max_abs_diff,
                "doc_vec_norm_max_dev": norm_dev, **diag}
        rows = [meta] + [
            {"rank": i + 1, "doc_id": CASE_RESUME, "db_document_id": h[0], "chunk_index": h[1],
             "score_type": "rrf", "score": float(h[4]), "distance": float(h[3]),
             "vec_rank": h[5], "fts_rank": h[6], "chunk_head": h[2][:120]}
            for i, h in enumerate(hits)
        ]
        write_jsonl(smoke_dir() / f"{name}.jsonl", rows)
        rr.n_items = 1
    if mid == "EMB-1":  # 생성 스모크에 넘길 운영 형식 컨텍스트
        context = "\n---\n".join(f"[doc#{h[0]} chunk#{h[1]}] {h[2]}" for h in hits)
        (smoke_dir() / "2-emb1.context.txt").write_text(context, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1)[:800])
    for r in rows[1:]:
        print(f"  {r['rank']}. chunk#{r['chunk_index']} rrf={r['score']:.5f} dist={r['distance']:.4f} "
              f"vec={r['vec_rank']} fts={r['fts_rank']}")
    assert hits, "검색 결과 0건"
    assert max_abs_diff < 1e-4, f"코사인 불일치 {max_abs_diff}"


def cmd_emb1(_: argparse.Namespace) -> None:
    _embed_and_search("EMB-1", "document_embeddings", PROD_EMBED_DIM)


def cmd_emb4(_: argparse.Namespace) -> None:
    _embed_and_search("EMB-4", "document_embeddings_d2560", 2560)


# ---------------------------------------------------------------------------
def cmd_rerank(_: argparse.Namespace) -> None:
    import psycopg
    from sentence_transformers import CrossEncoder

    cfg = {"embed_model": "EMB-1", "reranker": "RR-1", "stage1_cand": RERANK_CAND,
           "stage1_limit": RERANK_CAND, "k": TOP_K, "rerank_input": "prefixed chunk_text (운영 저장 텍스트)"}
    with recorder("4-rerank", condition_id="smoke-RR-1", config=cfg, model_ids=["EMB-1", "RR-1"]) as rr:
        model, _ = _load_embedder("EMB-1")
        query = _query_text()
        q = _encode(model, [query], query=True)[0]
        del model
        with psycopg.connect(PG_DSN_DEFAULT) as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM analyzed_documents WHERE document_path = %s",
                        (f"rag-eval/smoke-emb-1/{CASE_RESUME}.md",))
            row = cur.fetchone()
            assert row, "2-emb1 를 먼저 실행"
            doc_id = row[0]
            cur.execute("SELECT count(*) FROM document_embeddings WHERE document_id = %s", (doc_id,))
            n_scope = cur.fetchone()[0]
            hits, diag = search(cur, "document_embeddings", _vec_literal(_pad(q.tolist(), PROD_EMBED_DIM)),
                                query, [doc_id], RERANK_CAND, RERANK_CAND, n_scope)
            conn.commit()
        t0 = time.perf_counter()
        ce = CrossEncoder(str(model_path("RR-1")), device="cuda")
        load_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = ce.predict([(query, h[2]) for h in hits])
        rerank_ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))[:TOP_K]
        rows = [{"stage1_n": len(hits), "model_load_sec": round(load_sec, 2), "rerank_ms": round(rerank_ms, 1),
                 **diag}] + [
            {"rank": r + 1, "doc_id": CASE_RESUME, "chunk_index": hits[i][1], "score_type": "rerank",
             "score": float(scores[i]), "pre_rerank_rank": i + 1, "chunk_head": hits[i][2][:120]}
            for r, i in enumerate(order)
        ]
        write_jsonl(smoke_dir() / "4-rerank.jsonl", rows)
        rr.n_items = 1
    for r in rows[1:]:
        print(f"  {r['rank']}. chunk#{r['chunk_index']} score={r['score']:.4f} (1단계 {r['pre_rerank_rank']}위)")
    assert len(rows) - 1 == min(TOP_K, len(hits))


# ---------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    from servers.llama_server import base_url, server_record, start, stop

    from ai_server.chain.followup_generation_chain import build_streaming_followup_generator

    profile = args.profile
    name = f"5-generate-{profile}"
    case = dict(_followup_case())
    ctx_path = smoke_dir() / "2-emb1.context.txt"
    assert ctx_path.exists(), "2-emb1 를 먼저 실행"
    case["context"] = ctx_path.read_text(encoding="utf-8")

    started = False
    if not args.keep_server:
        st = start(profile, args.n_cpu_moe)
        started = True
        print(f"{profile} up: load {st['load_sec']}s, VRAM {st['vram_after_mib']} MiB")
    try:
        # 운영 Flash 설정(temperature·max_tokens·프롬프트·파서)은 그대로, 엔드포인트·모델 이름만 로컬로.
        # 타임아웃·재시도만 실험용으로 바꾼다 (운영 10초는 원격 Flash 기준, 숨은 재시도는 지연을 흐린다).
        settings = _settings(
            llm_flash_model=profile,
            llm_flash_base_url=f"{base_url(profile)}/v1",
            llm_flash_api_key="local",
            llm_flash_timeout_sec=args.timeout,
            llm_max_retries=0,
        )
        gen = build_streaming_followup_generator(settings)
        gen._llm.seed = args.seed  # noqa: SLF001  (OpenAI 호환 seed 필드 — llama-server 가 받는다)
        cfg = {"case": CASE_FOLLOWUP, "context_from": "2-emb1 (B0 hybrid top-5)",
               "temperature": settings.llm_flash_temperature, "max_tokens": settings.llm_flash_max_tokens,
               "timeout_sec": args.timeout, "max_retries": 0}
        with recorder(name, condition_id=f"smoke-G-B0-{profile}", config=cfg, model_ids=[profile],
                      server=server_record(profile), seed=args.seed) as rr:
            kwargs = {k: case[k] for k in ("job_category", "mode", "previous_question", "answer_text",
                                            "context", "parent_category", "expected_signal", "history")}
            first: list[float] = []
            t0 = time.perf_counter()

            def on_tok(_d: str) -> None:
                if not first:
                    first.append(time.perf_counter() - t0)

            res = asyncio.run(gen.stream(on_question_token=on_tok, **kwargs))
            latency = time.perf_counter() - t0
            rec = {
                "latency_sec": round(latency, 3),
                "ttft_question_sec": round(first[0], 3) if first else None,
                "answer_intent": res.answer_intent,
                "followup_question": res.followup_question,
                "answer_evaluation": res.answer_evaluation.model_dump() if res.answer_evaluation else None,
            }
            write_jsonl(smoke_dir() / f"{name}.jsonl", [rec])
            rr.n_items = 1
    finally:
        if started:
            stop(profile)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    assert res.followup_question, "꼬리질문이 비었다"
    assert res.answer_evaluation is not None, "meta 파싱 실패 (answer_evaluation 없음)"


# ---------------------------------------------------------------------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "reason"],
    "additionalProperties": False,
}


def cmd_claude(args: argparse.Namespace) -> None:
    from llm_client import LabClient, cached_text, json_of

    run_id = "E0-smoke-6-claude"
    lab = LabClient(task_id=TASK_ID, budget_usd=1.0, run_id=run_id)
    ledger_before = (lab.ledger_path.read_text(encoding="utf-8").splitlines()
                     if lab.ledger_path.exists() else [])
    case = _followup_case()
    gen_path = smoke_dir() / "5-generate-M-S.jsonl"
    fq = (json.loads(gen_path.read_text(encoding="utf-8").splitlines()[0])["followup_question"]
          if gen_path.exists() else "멱등 키 테이블은 언제 정리하셨나요?")
    system = [cached_text(
        "당신은 한국어 면접 꼬리질문을 보는 평가자입니다. 꼬리질문이 지원자 답변에 실제로 나온 내용만 "
        "전제하는지 판정하세요. grounded 는 참/거짓, reason 은 한 문장입니다."
    )]
    params = {
        "model": args.model,
        "max_tokens": 1024,
        "output_config": {"effort": "low"},
        "system": system,
        "messages": [{"role": "user", "content": f"[지원자 답변]\n{case['answer_text']}\n\n[꼬리질문]\n{fq}"}],
    }
    cfg = {"model": args.model, "effort": "low", "structured_output": True, "budget_usd": 1.0}
    with recorder("6-claude", condition_id="smoke-claude", config=cfg) as rr:
        msg = lab.call(estimate_usd=0.01, json_schema=JUDGE_SCHEMA, **params)
        sync_out = json_of(msg)
        bid = lab.submit_batch([("smoke-1", params)], estimate_usd=0.01, json_schema=JUDGE_SCHEMA)
        print(f"batch {bid} 제출 — 완료 대기")
        lab.wait_batch(bid, poll_sec=20, timeout_sec=args.batch_timeout)
        res = lab.collect_batch(bid)
        assert res["smoke-1"]["type"] == "succeeded", res["smoke-1"]
        batch_out = json_of(res["smoke-1"]["message"])
        rr.n_items = 2
    lines = lab.ledger_path.read_text(encoding="utf-8").splitlines()
    new = [json.loads(x) for x in lines[len(ledger_before):]]
    out = {
        "followup_question": fq,
        "sync": {"stop_reason": msg.stop_reason, "parsed": sync_out},
        "batch": {"batch_id": bid, "parsed": batch_out},
        "ledger_new_rows": new,
        "task_spent_usd": lab.spent_usd(),
    }
    write_jsonl(smoke_dir() / "6-claude.jsonl", [out])
    print(json.dumps(out, ensure_ascii=False, indent=1))
    assert len(new) == 2, f"원장 새 줄 {len(new)} ≠ 2"
    assert [r["batch"] for r in new] == [False, True]
    assert new[0]["request_id"] and new[1]["batch_id"] == bid and new[1]["custom_id"] == "smoke-1"
    assert lab.spent_usd() < 1.0


def cmd_env(_: argparse.Namespace) -> None:
    """E0 환경 기록 results/E0-env/run.json — 확정 모델 8종, 서버 2종 설정, 스모크 결과 요약."""
    from servers.llama_server import CTX, build_args

    from common import llama_build

    smoke = {}
    for p in sorted(smoke_dir().glob("*.run.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        smoke[p.name.removesuffix(".run.json")] = {
            "ok": r.get("ended_at") is not None and "error" not in r,
            "stackup_commit": r["stackup_commit"], "ended_at": r["ended_at"]}
    cfg = {"pg_dsn": PG_DSN_DEFAULT, "prod_embed_dim": PROD_EMBED_DIM, "rrf_k": RRF_K, "ctx": CTX, "smoke": smoke}
    server = {"binary_build": llama_build(), "n_parallel": 1,
              "args": {p: build_args(p, None) for p in ("M-S", "M-M")}}
    with RunRecorder(results_dir(EXP + "-env"), run_id="E0-env", task_id=TASK_ID, experiment=EXP,
                     condition_id=None, config=cfg, model_ids=list(MODELS), server=server,
                     notes="T01 환경 기록. 스모크별 run.json 은 smoke/ 에 있다.") as rr:
        rr.n_items = len(smoke)
    print(json.dumps(smoke, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    for n in ("chunk", "emb1", "emb4", "rerank"):
        sub.add_parser(n)
    g = sub.add_parser("generate")
    g.add_argument("--profile", choices=("M-S", "M-M"), required=True)
    g.add_argument("--n-cpu-moe", type=int, default=None)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--timeout", type=float, default=120.0)
    g.add_argument("--keep-server", action="store_true", help="이미 띄운 서버를 쓴다 (시작·중지 안 함)")
    c = sub.add_parser("claude")
    c.add_argument("--model", default="claude-haiku-5-5")
    c.add_argument("--batch-timeout", type=float, default=3600)
    a = ap.parse_args()
    {"env": cmd_env, "chunk": cmd_chunk, "emb1": cmd_emb1, "emb4": cmd_emb4, "rerank": cmd_rerank,
     "generate": cmd_generate, "claude": cmd_claude}[a.cmd](a)


if __name__ == "__main__":
    main()
