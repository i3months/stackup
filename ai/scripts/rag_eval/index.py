"""인덱싱: 코퍼스 → 변형 청크 → 접두 → 임베딩(캐시) → 실험 DB (SPEC §2, §4.3, §9.2, §10).

    python index.py --corpus E1 --condition B0 [--out E1-labels]
    python index.py --corpus cases --condition B0 --out E0-harness

- 청킹·결정적 접두는 운영 모듈(`MarkdownChunker`, `_contextualize`)을 그대로 import 한다.
- chunks.jsonl 기본 키 (variant_id, doc_id, chunk_index), chunk_key = 접두를 뺀 본문의 SHA-256.
- 임베딩 캐시 키 = (embed_id, 텍스트 SHA-256, 문서/질의 구분). ~/rag-lab/cache/embeddings.sqlite
- DB: 인덱스 키마다 페르소나 사용자 1명 + 문서 행을 따로 만든다 (운영 UNIQUE(document_id, chunk_index) 때문,
  SPEC §2 변형 저장). 매핑은 rag_eval_docmap 테이블. 문서는 doc_id 순서로 넣어 DB id 순서 = doc_id 순서.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import MODELS, PG_DSN_DEFAULT, PROD_EMBED_DIM, model_path, results_dir, write_jsonl  # noqa: E402

CACHE_DB = Path(os.environ.get("RAG_CACHE_DIR", Path.home() / "rag-lab" / "cache")) / "embeddings.sqlite"
D2560_TABLE = "document_embeddings_d2560"
DEVICE = os.environ.get("RAG_DEVICE", "cuda")  # GPU 를 M-M 이 쓰는 동안 검증용으로 cpu 가능 (run.json 에 기록)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---- 코퍼스 -------------------------------------------------------------------
@dataclass(frozen=True)
class Doc:
    persona_id: str
    doc_id: str
    doc_type: str
    raw_text: str
    analysis_md: str
    summary: str


def _read_raw(p: Path) -> str:
    """문서 원문을 바이트 그대로 읽는다. read_text 는 CRLF 를 LF 로 바꾸는데, 레포 원문(운영 추출기 출력)에는
    CRLF 파일이 있다 (E1a P29-repo). 사실 오프셋도 이 원문 기준이다 (T03 보고서 §5-5)."""
    with p.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def load_corpus(name: str) -> list[Doc]:
    if name == "cases":
        return _cases_corpus()
    corpus = results_dir("E1-corpus")
    docs = []
    for p in sorted((corpus / "docs").glob("*/*.md")):
        doc_id = p.stem
        js = corpus / "analysis" / f"{doc_id}.json"
        if not js.exists():
            continue
        a = json.loads(js.read_text(encoding="utf-8"))
        docs.append(Doc(persona_id=p.parent.name, doc_id=doc_id, doc_type=doc_id.split("-", 1)[1],
                        raw_text=_read_raw(p),
                        analysis_md=(corpus / "analysis" / f"{doc_id}.md").read_text(encoding="utf-8"),
                        summary=a["summary"]))
    return docs


def _summary_stand_in(md: str) -> str:
    lines, on = [], False
    for ln in md.splitlines():
        if ln.startswith("## "):
            on = ln.strip() in ("## 요약", "## 개요")
            continue
        if on and ln.strip():
            lines.append(ln.strip().lstrip("-").strip())
    return " ".join(lines)


CASE_DOCS = ["RESUME_BACKEND", "REPO_FRONTEND", "RESUME_INFRA_DBA", "COVER_LETTER", "RESUME_FRONTEND_JUNIOR",
             "RESUME_DBA_DATA"]


def _cases_corpus() -> list[Doc]:
    """운영 동등성 검증용 소형 코퍼스: ai/scripts/llm_eval 의 합성 문서 6개를 페르소나 C00 하나로.
    분석 요약이 따로 없으므로 원문 = 분석 마크다운, 요약 = '## 요약'·'## 개요' 절."""
    import cases_v2  # noqa: PLC0415

    out = []
    for i, name in enumerate(CASE_DOCS):
        text = getattr(cases_v2, name)
        out.append(Doc(persona_id="C00", doc_id=f"C00-{i:02d}-{name.lower()}", doc_type=name.split("_")[0].lower(),
                       raw_text=text, analysis_md=text, summary=_summary_stand_in(text)))
    return out


# ---- 변형 청크 ----------------------------------------------------------------
def make_chunker(cond: C.Condition) -> Any:
    from ai_server.rag.chunker import MarkdownChunker

    return MarkdownChunker(chunk_size=cond.chunk_size, chunk_overlap=cond.chunk_overlap)


def variant_chunks(cond: C.Condition, docs: list[Doc]) -> list[dict[str, Any]]:
    """변형의 전 청크. both 는 요약 부분 다음에 원문 부분을 이어 붙이고 chunk_index 를 이어서 매긴다."""
    chunker = make_chunker(cond)
    parts = {"summary": ["summary"], "raw": ["raw"], "both": ["summary", "raw"]}[cond.index_source]
    rows = []
    for d in sorted(docs, key=lambda x: x.doc_id):
        idx = 0
        for part in parts:
            text = d.analysis_md if part == "summary" else d.raw_text
            for c in chunker.split(text):
                rows.append({
                    "variant_id": cond.variant_id, "persona_id": d.persona_id, "doc_id": d.doc_id,
                    "chunk_index": idx, "part": part, "part_chunk_index": c.index,
                    "chunk_key": sha256(c.text), "heading_path": c.heading_path, "body": c.text,
                })
                idx += 1
    return rows


def deterministic_prefix(row: dict[str, Any], doc: Doc) -> str:
    from ai_server.analyzer._embedding_step import _contextualize

    full = _contextualize(row["body"], row["heading_path"], doc.summary)
    assert full.endswith(row["body"])
    return full[: len(full) - len(row["body"])]


LLM_PREFIX_SYSTEM = "당신은 검색 색인을 돕는 사람입니다. 지시대로 짧은 맥락 문장만 답합니다."
LLM_PREFIX_USER = (  # Anthropic Contextual Retrieval 프롬프트의 한국어 번역
    "<document>\n{doc}\n</document>\n"
    "다음은 위 문서 전체 안에서 위치를 설명해야 하는 청크입니다.\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "검색 성능을 높이기 위해, 전체 문서 맥락 안에서 이 청크가 무엇에 관한 것인지 설명하는 짧은 맥락을 "
    "한국어 1~2문장으로 쓰세요. 맥락 문장만 답하고 다른 말은 하지 마세요."
)


def llm_prefix_cache() -> dict[str, dict[str, Any]]:
    """key → {"prefix", "cache_prompt"}. cache_prompt 가 없는 옛 레코드는 캐시 끔으로 만든 것이다 (T04·T05 P-026 이전)."""
    cache_p = CACHE_DB.parent / "llm_prefix.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    if cache_p.exists():
        for ln in cache_p.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            cache[r["key"]] = {"prefix": r["prefix"], "cache_prompt": r.get("cache_prompt", False)}
    return cache


def llm_prefixes(rows: list[dict[str, Any]], docs: dict[str, Doc], *, seed: int = 4242) -> dict[tuple, str]:
    """PFX-llm: M-M 이 청크별 1~2문장. (문서 해시, 본문 해시) 로 캐시.

    프롬프트 캐시 (P-026): 요약 부분은 cache_prompt=false (T04). 원문 부분은 문서 전체(평균 2만여 토큰)가 프롬프트에
    들어가 캐시 끔이면 청크당 30초여서 cache_prompt=true 로 만든다. 이때 출력이 서버 캐시 상태에 따라 달라질 수 있어
    (P-018) 재현 단위는 '같은 순서의 전체 실행'이다. rows 는 doc_id → chunk_index 순이라 같은 문서의 청크가 연속이고,
    실행 순서를 E1-labels/llm_prefix_runs.jsonl 에 남긴다. 레코드마다 cache_prompt 를 기록한다."""
    from common import append_jsonl, now_iso, results_dir  # noqa: PLC0415
    from local_llm import LocalLLM  # noqa: PLC0415
    from servers.llama_server import base_url  # noqa: PLC0415

    cache_p = CACHE_DB.parent / "llm_prefix.jsonl"
    cache = llm_prefix_cache()
    log = CACHE_DB.parent / "llm_prefix_calls.jsonl"
    llms = {False: LocalLLM(base_url("M-M"), model="M-M", log_path=log, cache_prompt=False),
            True: LocalLLM(base_url("M-M"), model="M-M", log_path=log, cache_prompt=True)}
    order_p = results_dir("E1-labels") / "llm_prefix_runs.jsonl"
    out = {}
    for r in rows:
        d = docs[r["doc_id"]]
        src = d.analysis_md if r["part"] == "summary" else d.raw_text
        key = sha256(src) + ":" + r["chunk_key"]
        if key not in cache:
            cp = r["part"] != "summary"
            msgs = [{"role": "system", "content": LLM_PREFIX_SYSTEM},
                    {"role": "user", "content": LLM_PREFIX_USER.format(doc=src, chunk=r["body"])}]
            text = llms[cp].chat(msgs, seed=seed, temperature=0.0, max_tokens=200, tag="pfx-llm").strip()
            cache[key] = {"prefix": text, "cache_prompt": cp}
            append_jsonl(cache_p, [{"key": key, "prefix": text, "cache_prompt": cp}])
            append_jsonl(order_p, [{"variant_id": r["variant_id"], "doc_id": r["doc_id"], "chunk_index": r["chunk_index"],
                                    "key": key, "cache_prompt": cp, "ts": now_iso()}])
        out[(r["doc_id"], r["chunk_index"])] = cache[key]["prefix"] + "\n\n"
    return out


# ---- 임베딩 -------------------------------------------------------------------
class Embedder:
    """조건의 embedder·dtype·query_instruction·truncate_dim 을 따른다. (embed_id, kind, sha) 캐시."""

    def __init__(self, cond: C.Condition) -> None:
        self.cond = cond
        self.embed_id = cond.embed_id
        self._model = None
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (embed_id TEXT, kind TEXT, sha TEXT, vec BLOB, "
                        "PRIMARY KEY (embed_id, kind, sha))")

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[self.cond.dtype]
            self._model = SentenceTransformer(str(model_path(self.cond.embedder)), device=DEVICE,
                                              model_kwargs={"dtype": dtype})
        return self._model

    def unload(self) -> None:
        if self._model is not None:
            import gc

            import torch

            self._model = None
            gc.collect()  # sentence-transformers 모델은 순환 참조가 있어 바로 안 풀린다 (T05 R-BEST: 12GB 가 차서 10배 느려짐)
            torch.cuda.empty_cache()

    def _kind(self, query: bool) -> str:
        return "query-" + self.cond.query_instruction if query else "doc"

    def encode(self, texts: list[str], *, query: bool) -> "Any":
        import numpy as np

        kind = self._kind(query) + ("" if DEVICE == "cuda" else f"@{DEVICE}")  # 장치가 다르면 캐시도 분리
        shas = [sha256(t) for t in texts]
        got: dict[str, Any] = {}
        for i in range(0, len(shas), 500):
            batch = shas[i:i + 500]
            q = f"SELECT sha, vec FROM emb WHERE embed_id=? AND kind=? AND sha IN ({','.join('?' * len(batch))})"
            for sha, blob in self.db.execute(q, [self.embed_id, kind, *batch]):
                got[sha] = np.frombuffer(blob, dtype=np.float32)
        todo = [(s, t) for s, t in zip(shas, texts) if s not in got]
        todo = list({s: (s, t) for s, t in todo}.values())
        if todo:
            kw: dict[str, Any] = {"normalize_embeddings": True, "batch_size": 8, "convert_to_numpy": True}
            if query and self.cond.query_instruction == "query":
                kw["prompt_name"] = "query"
            vecs = self.model.encode([t for _, t in todo], **kw).astype(np.float32)
            if self.cond.truncate_dim:
                vecs = vecs[:, : self.cond.truncate_dim]
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # MRL 절단 후 재정규화
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?,?)",
                                [(self.embed_id, kind, s, v.tobytes()) for (s, _), v in zip(todo, vecs)])
            self.db.commit()
            for (s, _), v in zip(todo, vecs):
                got[s] = v
        return np.stack([got[s] for s in shas])

    @property
    def dim(self) -> int:
        return self.cond.truncate_dim or MODELS[self.cond.embedder]["dim"]

    @property
    def table(self) -> str:
        return "document_embeddings" if self.dim <= PROD_EMBED_DIM else D2560_TABLE


def vec_literal(v: Iterable[float], dim: int) -> str:
    vals = [float(x) for x in v]
    vals += [0.0] * (dim - len(vals))  # 0 패딩 (SPEC §2)
    return "[" + ",".join(repr(x) for x in vals) + "]"


# ---- DB ---------------------------------------------------------------------
DOCMAP_DDL = """
CREATE TABLE IF NOT EXISTS rag_eval_docmap (
    index_key TEXT NOT NULL, persona_id TEXT NOT NULL, doc_id TEXT NOT NULL, part TEXT NOT NULL,
    first_chunk_index INT NOT NULL, db_document_id BIGINT NOT NULL,
    PRIMARY KEY (index_key, doc_id, part)
);
CREATE TABLE IF NOT EXISTS rag_eval_users (persona_id TEXT PRIMARY KEY, user_id BIGINT NOT NULL);
"""


def ensure_tables(cur: Any) -> None:
    from smoke import D2560_DDL  # noqa: PLC0415

    cur.execute(DOCMAP_DDL)
    cur.execute(D2560_DDL)


def persona_user(cur: Any, persona_id: str) -> int:
    cur.execute("SELECT user_id FROM rag_eval_users WHERE persona_id=%s", (persona_id,))
    r = cur.fetchone()
    if r:
        return r[0]
    gid = 910_000_000 + int(sha256(persona_id)[:6], 16)
    cur.execute(
        "INSERT INTO users (github_id, github_username, encrypted_github_access_token, display_name) "
        "VALUES (%s,%s,'none',%s) RETURNING id", (gid, f"rag-eval-{persona_id}", f"rag-eval-{persona_id}"))
    uid = cur.fetchone()[0]
    cur.execute("INSERT INTO rag_eval_users VALUES (%s,%s)", (persona_id, uid))
    return uid


def load_to_db(cond: C.Condition, rows: list[dict[str, Any]], texts: list[str], vecs: Any, dsn: str) -> dict[str, Any]:
    import psycopg

    emb_table = "document_embeddings" if (cond.truncate_dim or MODELS[cond.embedder]["dim"]) <= PROD_EMBED_DIM \
        else D2560_TABLE
    dim = PROD_EMBED_DIM if emb_table == "document_embeddings" else 2560
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        ensure_tables(cur)
        cur.execute("SELECT db_document_id FROM rag_eval_docmap WHERE index_key=%s", (cond.index_key,))
        old = [r[0] for r in cur.fetchall()]
        if old:
            cur.execute("DELETE FROM analyzed_documents WHERE id = ANY(%s)", (old,))  # CASCADE → 임베딩
            cur.execute("DELETE FROM rag_eval_docmap WHERE index_key=%s", (cond.index_key,))
        groups: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault((r["doc_id"], r["part"]), []).append(i)
        for (doc_id, part), idxs in groups.items():  # rows 는 doc_id·part 순서
            r0 = rows[idxs[0]]
            uid = persona_user(cur, r0["persona_id"])
            cur.execute("INSERT INTO resumes (user_id, original_filename, file_path, file_type, status) "
                        "VALUES (%s,%s,%s,'PDF','ANALYZED') RETURNING id", (uid, f"{doc_id}.{part}", "rag-eval"))
            rid = cur.fetchone()[0]
            cur.execute("INSERT INTO analyzed_documents (resume_id, document_path, summary) VALUES (%s,%s,%s) "
                        "RETURNING id", (rid, f"rag-eval/{cond.index_key}/{doc_id}/{part}", ""))
            db_id = cur.fetchone()[0]
            cur.execute("INSERT INTO rag_eval_docmap VALUES (%s,%s,%s,%s,%s,%s)",
                        (cond.index_key, r0["persona_id"], doc_id, part, r0["chunk_index"], db_id))
            with cur.copy(f"COPY {emb_table} (document_id, chunk_index, chunk_text, embedding, model) FROM STDIN") as cp:
                for i in idxs:
                    cp.write_row((db_id, rows[i]["part_chunk_index"], texts[i], vec_literal(vecs[i], dim),
                                  f"{MODELS[cond.embedder]['repo']}:{cond.embed_id}"))
        conn.commit()
    return {"table": emb_table, "n_docs": len(groups), "n_chunks": len(rows)}


# ---- 진입점 -------------------------------------------------------------------
def merge_jsonl(path: Path, rows: list[dict[str, Any]], key: str | tuple[str, ...], value: str | tuple[str, ...]) -> None:
    """key 값이 value 인 기존 행을 rows 로 바꾼다 (여러 필드면 튜플)."""
    keys = key if isinstance(key, tuple) else (key,)
    vals = value if isinstance(value, tuple) else (value,)
    keep = []
    if path.exists():
        keep = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        keep = [r for r in keep if tuple(r.get(k) for k in keys) != vals]
    # 다른 변형의 행도 같이 다시 쓰는 파일이라, 쓰다 끊기면 라벨 전체가 깨진다 → 임시 파일 + fsync + 교체 (T05)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, keep + rows)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def chunks_and_prefixes(cond: C.Condition, corpus: str, out: str) -> tuple[list[dict[str, Any]], dict[tuple, str], str | None]:
    """변형 청크와 접두를 만들고 chunks.jsonl·prefixes.jsonl (SPEC §9.2) 에 쓴다. 임베딩 없음."""
    docs = load_corpus(corpus)
    by_id = {d.doc_id: d for d in docs}
    rows = variant_chunks(cond, docs)
    if cond.prefix == "deterministic":
        prefixes = {(r["doc_id"], r["chunk_index"]): deterministic_prefix(r, by_id[r["doc_id"]]) for r in rows}
        gen = "deterministic"
    elif cond.prefix == "llm":
        prefixes = llm_prefixes(rows, by_id)
        gen = "M-M"
    else:
        prefixes = {(r["doc_id"], r["chunk_index"]): "" for r in rows}
        gen = None
    outdir = results_dir(out)
    merge_jsonl(outdir / "chunks.jsonl", rows, "variant_id", cond.variant_id)
    if cond.prefix != "none":
        pref_rows = [{"variant_id": r["variant_id"], "doc_id": r["doc_id"], "chunk_index": r["chunk_index"],
                      "prefix_id": cond.prefix, "prefix_text": prefixes[(r["doc_id"], r["chunk_index"])],
                      "generator": gen} for r in rows]
        if cond.prefix == "llm":  # P-026: 레코드마다 프롬프트 캐시 설정
            cache = llm_prefix_cache()
            for r, pr in zip(rows, pref_rows):
                d = by_id[r["doc_id"]]
                src = d.analysis_md if r["part"] == "summary" else d.raw_text
                pr["cache_prompt"] = cache[sha256(src) + ":" + r["chunk_key"]]["cache_prompt"]
        merge_jsonl(outdir / "prefixes.jsonl", pref_rows, ("variant_id", "prefix_id"), (cond.variant_id, cond.prefix))
    return rows, prefixes, gen


def index(cond: C.Condition, corpus: str, out: str, dsn: str = PG_DSN_DEFAULT) -> dict[str, Any]:
    t0 = time.perf_counter()
    rows, prefixes, _ = chunks_and_prefixes(cond, corpus, out)
    texts = [prefixes[(r["doc_id"], r["chunk_index"])] + r["body"] for r in rows]
    t1 = time.perf_counter()
    emb = Embedder(cond)
    vecs = emb.encode(texts, query=False)
    t2 = time.perf_counter()
    info = load_to_db(cond, rows, texts, vecs, dsn)
    emb.unload()
    info.update(condition=cond.condition_id, index_key=cond.index_key, corpus=corpus,
                chunk_sec=round(t1 - t0, 2), embed_sec=round(t2 - t1, 2), load_sec=round(time.perf_counter() - t2, 2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="E1")
    ap.add_argument("--condition", required=True, help="조건 ID 또는 쉼표 목록")
    ap.add_argument("--out", default="E1-labels")
    ap.add_argument("--dsn", default=PG_DSN_DEFAULT)
    ap.add_argument("--chunks-only", action="store_true", help="청크·접두만 쓰고 임베딩·적재는 하지 않는다 (T04 라벨링)")
    a = ap.parse_args()
    for cid in a.condition.split(","):
        if a.chunks_only:
            rows, _, gen = chunks_and_prefixes(C.load(cid), a.corpus, a.out)
            print(json.dumps({"condition": cid, "variant_id": C.load(cid).variant_id, "n_chunks": len(rows),
                              "prefix": gen}, ensure_ascii=False))
        else:
            print(json.dumps(index(C.load(cid), a.corpus, a.out, a.dsn), ensure_ascii=False))


if __name__ == "__main__":
    main()
