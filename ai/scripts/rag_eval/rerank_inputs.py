"""재순위 입력 길이·잘림 실측 (P-024 ⑥): RR-1·RR-2 의 max_length, 잘림 방식, 질의별 잘림 비율.

    python rerank_inputs.py        # → E2-retrieval/rerank-inputs-dev.json (dev 질의만, GPU 안 씀)

- 모델 설정은 실제로 CrossEncoder 를 CPU 에 올려 `max_length` 와 토크나이저 `truncation_side` 를 읽는다.
- 잘림 방식: sentence-transformers CrossEncoder.predict 는 (질의, 청크) 쌍을 `truncation=True` 로 토큰화한다.
  HF 토크나이저에서 쌍 입력의 `True` 는 'longest_first' 다 — 두 쪽 중 긴 쪽에서 한 토큰씩 깎는다.
- 입력: 재순위 1단계 후보 = hybrid_simple cand = limit = 20. 요약 변형은 페르소나 청크가 20개 이하라 전 청크가
  후보다. 원문 헤딩 변형(R-BEST 후보)은 R-BEST dev run 의 1단계 후보를 그대로 쓸 수 없어(상위 k 만 저장),
  그 페르소나 전 청크로 상한을 본다. 청크 텍스트는 DB 에 들어가는 것과 같은 '접두 + 본문'이다.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import model_path, now_iso, results_dir, write_json  # noqa: E402
from labels import read_jsonl  # noqa: E402

VARIANTS = ("summary-c1000o200", "raw-h2000o0")


def chunk_texts(variant: str) -> dict[str, list[str]]:
    pre = {(r["doc_id"], r["chunk_index"]): r["prefix_text"] for r in read_jsonl(results_dir("E1-labels") / "prefixes.jsonl")
           if r["variant_id"] == variant and r["prefix_id"] == "deterministic"}
    out: dict[str, list[str]] = defaultdict(list)
    for c in read_jsonl(results_dir("E1-labels") / "chunks.jsonl"):
        if c["variant_id"] == variant:
            out[c["persona_id"]].append(pre[(c["doc_id"], c["chunk_index"])] + c["body"])
    return out


def main() -> None:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    qs = [q for q in read_jsonl(results_dir("E1-queries") / "queries.jsonl") if q["split"] == "dev"]
    texts = {v: chunk_texts(v) for v in VARIANTS}
    out: dict[str, Any] = {"ts": now_iso(), "method": __doc__.strip().splitlines()[2:9], "models": {}}
    for rr in ("RR-1", "RR-2"):
        ce = CrossEncoder(str(model_path(rr)), device="cpu")
        tok = ce.tokenizer
        max_len = ce.max_length or tok.model_max_length
        n_special = tok.num_special_tokens_to_add(pair=True)
        info: dict[str, Any] = {"max_length": max_len, "tokenizer_model_max_length": tok.model_max_length,
                                "truncation_side": tok.truncation_side, "pair_special_tokens": n_special,
                                "truncation_strategy": "longest_first (truncation=True, 쌍 입력)", "variants": {}}
        for v in VARIANTS:
            per_q, q_tok_all, c_tok_all, total_all = [], [], [], []
            for q in qs:
                qn = len(tok(q["query_text"], add_special_tokens=False)["input_ids"])
                cn = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts[v][q["persona_id"]]]
                tot = [qn + c + n_special for c in cn]
                trunc = [t > max_len for t in tot]
                q_tok_all.append(qn)
                c_tok_all += cn
                total_all += tot
                per_q.append({"query_id": q["query_id"], "qtype": q["qtype"], "query_tokens": qn,
                              "n_pairs": len(cn), "truncated_pairs": sum(trunc),
                              "max_pair_tokens": max(tot) if tot else 0})
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in per_q:
                by_type[r["qtype"]].append(r)
            info["variants"][v] = {
                "queries_with_any_truncation": sum(r["truncated_pairs"] > 0 for r in per_q),
                "n_queries": len(per_q),
                "pairs_truncated": sum(r["truncated_pairs"] for r in per_q),
                "n_pairs": sum(r["n_pairs"] for r in per_q),
                "query_tokens": {t: {"median": st.median([r["query_tokens"] for r in rs]),
                                     "max": max(r["query_tokens"] for r in rs)} for t, rs in by_type.items()},
                "chunk_tokens": {"median": st.median(c_tok_all), "max": max(c_tok_all)},
                "pair_tokens": {"median": st.median(total_all), "max": max(total_all)},
                "query_share_of_pair_median_pool": round(st.median(
                    [r["query_tokens"] / r["max_pair_tokens"] for r in by_type["pool"]]), 3),
            }
            print(rr, v, json.dumps(info["variants"][v], ensure_ascii=False), flush=True)
        out["models"][rr] = info
        del ce
    write_json(results_dir("E2-retrieval") / "rerank-inputs-dev.json", out)


if __name__ == "__main__":
    main()
