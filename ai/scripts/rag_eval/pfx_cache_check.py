"""P-026 점검: 원문 PFX-llm 처음 20청크를 캐시 켬으로 다시 만들어, 캐시 끔 출력과 질적으로 같은 종류인지 본다.

    python pfx_cache_check.py      # M-M 필요 → E1-labels/pfx-llm-cache-check.json (본 캐시에는 쓰지 않음)

기록: 청크별 두 출력, 길이, 바이트 동일 여부, 그 문서 사실의 근거 문장 속 숫자·영문 토큰이 접두에 나오는지(간이 '사실 포함'),
      완전 일치 비율. 정량 비교가 아니라 같은 종류의 출력인지 보는 용도다.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402
from common import now_iso, results_dir, write_json  # noqa: E402
from index import LLM_PREFIX_SYSTEM, LLM_PREFIX_USER, load_corpus, llm_prefix_cache, sha256, variant_chunks  # noqa: E402
from labels import read_jsonl  # noqa: E402

N = 20
_TOK = re.compile(r"[0-9][0-9.,%]*|[A-Za-z][A-Za-z0-9+#.\-]{2,}")


def distinctive(text: str) -> set[str]:
    return set(_TOK.findall(text))


def main() -> None:
    from local_llm import LocalLLM  # noqa: PLC0415
    from servers.llama_server import base_url  # noqa: PLC0415

    docs = {d.doc_id: d for d in load_corpus("E1")}
    cond = dataclasses.replace(C.load("R-SRC-raw"), prefix="llm")
    rows = variant_chunks(cond, list(docs.values()))
    cache = llm_prefix_cache()
    facts = read_jsonl(results_dir("E1-corpus") / "facts.jsonl")
    off = [(r, cache[sha256(docs[r["doc_id"]].raw_text) + ":" + r["chunk_key"]]) for r in rows
           if (sha256(docs[r["doc_id"]].raw_text) + ":" + r["chunk_key"]) in cache][:N]
    assert all(not c["cache_prompt"] for _, c in off), "점검 대상은 캐시 끔으로 만든 접두"
    llm = LocalLLM(base_url("M-M"), model="M-M", log_path=results_dir("E1-labels") / "pfx-llm-cache-check-calls.jsonl",
                   cache_prompt=True)
    out = []
    for r, c in off:
        d = docs[r["doc_id"]]
        msgs = [{"role": "system", "content": LLM_PREFIX_SYSTEM},
                {"role": "user", "content": LLM_PREFIX_USER.format(doc=d.raw_text, chunk=r["body"])}]
        on = llm.chat(msgs, seed=4242, temperature=0.0, max_tokens=200, tag="pfx-llm-cache-check").strip()
        fact_toks = set()
        for f in facts:
            if f["doc_id"] == r["doc_id"]:
                fact_toks |= distinctive(f["evidence_text"])
        out.append({"doc_id": r["doc_id"], "chunk_index": r["chunk_index"], "cache_off": c["prefix"], "cache_on": on,
                    "identical": on == c["prefix"], "len_off": len(c["prefix"]), "len_on": len(on),
                    "fact_tokens_off": sorted(distinctive(c["prefix"]) & fact_toks),
                    "fact_tokens_on": sorted(distinctive(on) & fact_toks)})
        print(json.dumps({k: out[-1][k] for k in ("doc_id", "chunk_index", "identical", "len_off", "len_on")}), flush=True)
    summ = {"n": len(out), "identical": sum(o["identical"] for o in out),
            "mean_len_off": round(sum(o["len_off"] for o in out) / len(out), 1),
            "mean_len_on": round(sum(o["len_on"] for o in out) / len(out), 1),
            "with_fact_tokens_off": sum(bool(o["fact_tokens_off"]) for o in out),
            "with_fact_tokens_on": sum(bool(o["fact_tokens_on"]) for o in out)}
    write_json(results_dir("E1-labels") / "pfx-llm-cache-check.json", {"ts": now_iso(), "decision": "P-026", "summary": summ,
                                                                       "items": out})
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
