"""사실 기반 관련성 라벨 (SPEC §4.3).

    python labels.py auto --labels-dir E1-labels --variant raw-c1000o200      # 원문 변형 자동 full 판정
    python labels.py unjudged --labels-dir E1-labels --variant summary-c1000o200 > todo.jsonl   # T04 포함 판정 입력

- containment.jsonl: chunk_key, fact_id, label(full·partial·none), labeler, ts
- 판정 대상은 접두를 뺀 본문(chunk_key = 본문 SHA-256).
- 자동 full: 원문 부분(part=raw)의 본문에 evidence_text 가 그대로 있고, 그 근거가 문서에서 한 번만 나오며 15자 이상일 때만.
  그 밖(청크 경계에 걸림·짧음·중복·요약 부분)은 LLM 판정으로 보낸다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import results_dir, write_jsonl

AUTO_MIN_CHARS = 15
LABELS = ("full", "partial", "none")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Labels:
    """containment 조회: (chunk_key, fact_id) → label. 같은 쌍이 여러 번 있으면 마지막(사람 > LLM > auto 순으로 덧씀)."""

    def __init__(self, containment: Iterable[dict[str, Any]], facts: Iterable[dict[str, Any]]) -> None:
        self.label: dict[tuple[str, str], str] = {}
        for r in containment:
            if r["label"] not in LABELS:
                raise ValueError(f"알 수 없는 라벨 {r['label']}")
            self.label[(r["chunk_key"], r["fact_id"])] = r["label"]
        self.facts = {f["fact_id"]: f for f in facts}
        self.persona_facts: dict[str, list[str]] = defaultdict(list)
        for f in self.facts.values():
            self.persona_facts[f["persona_id"]].append(f["fact_id"])

    @classmethod
    def load(cls, labels_dir: str, facts_path: Path | None = None) -> "Labels":
        d = results_dir(labels_dir)
        facts_path = facts_path or results_dir("E1-corpus") / "facts.jsonl"
        return cls(read_jsonl(d / "containment.jsonl"), read_jsonl(facts_path))

    def judged(self, chunk_key: str, persona_id: str) -> bool:
        """그 페르소나의 모든 사실에 대해 판정이 있어야 판정 완료 (IDCG 가 전 청크·전 사실 기준이므로)."""
        return all((chunk_key, f) in self.label for f in self.persona_facts[persona_id])

    def rel(self, chunk_key: str, gold_fact_ids: list[str]) -> int:
        """rel(q, chunk) (SPEC §4.3-2):
        2 = 근거 사실 중 하나 이상 full / 1 = 근거 사실 partial, 또는 근거 사실과 같은 경험의 다른 사실 full / 0."""
        if not gold_fact_ids:
            return 0
        labs = [self.label.get((chunk_key, f)) for f in gold_fact_ids]
        if any(x is None for x in labs):
            raise KeyError(f"미판정 (chunk {chunk_key[:12]}…, 근거 사실)")
        if "full" in labs:
            return 2
        if "partial" in labs:
            return 1
        exps = {self.facts[f]["experience_id"] for f in gold_fact_ids if self.facts[f]["experience_id"]}
        persona = self.facts[gold_fact_ids[0]]["persona_id"]
        for f in self.persona_facts[persona]:
            if f in gold_fact_ids or self.facts[f]["experience_id"] not in exps:
                continue
            lab = self.label.get((chunk_key, f))
            if lab is None:
                raise KeyError(f"미판정 (chunk {chunk_key[:12]}…, 같은 경험 사실 {f})")
            if lab == "full":
                return 1
        return 0


def auto_full(chunks: list[dict[str, Any]], facts: list[dict[str, Any]], raw_docs: dict[str, str]) -> list[dict[str, Any]]:
    """원문 부분 청크의 자동 full 판정 (SPEC §4.3-1). 나머지 쌍은 판정하지 않는다 (LLM 으로)."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        by_doc[f["doc_id"]].append(f)
    out, seen = [], set()
    for c in chunks:
        if c["part"] != "raw":
            continue
        for f in by_doc.get(c["doc_id"], []):
            ev = f["evidence_text"]
            if len(ev) < AUTO_MIN_CHARS or raw_docs[c["doc_id"]].count(ev) != 1:
                continue
            if ev in c["body"] and (c["chunk_key"], f["fact_id"]) not in seen:
                seen.add((c["chunk_key"], f["fact_id"]))
                out.append({"chunk_key": c["chunk_key"], "fact_id": f["fact_id"], "label": "full", "labeler": "auto",
                            "ts": ts})
    return out


def unjudged(chunks: list[dict[str, Any]], labels: Labels) -> list[dict[str, Any]]:
    """판정이 빠진 (청크, 사실) 묶음. 청크 1개 + 그 페르소나의 미판정 사실 목록 = LLM 호출 1회 (SPEC §4.3-1)."""
    out, seen = [], set()
    for c in chunks:
        if c["chunk_key"] in seen:
            continue
        missing = [f for f in labels.persona_facts[c["persona_id"]] if (c["chunk_key"], f) not in labels.label]
        if missing:
            seen.add(c["chunk_key"])
            out.append({"chunk_key": c["chunk_key"], "persona_id": c["persona_id"], "body": c["body"],
                        "fact_ids": missing})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["auto", "unjudged"])
    ap.add_argument("--labels-dir", default="E1-labels")
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    d = results_dir(a.labels_dir)
    chunks = [c for c in read_jsonl(d / "chunks.jsonl") if c["variant_id"] == a.variant]
    facts = read_jsonl(results_dir("E1-corpus") / "facts.jsonl")
    if a.cmd == "auto":
        raw_docs = {p.stem: p.read_text(encoding="utf-8") for p in (results_dir("E1-corpus") / "docs").glob("*/*.md")}
        new = auto_full(chunks, facts, raw_docs)
        old = read_jsonl(d / "containment.jsonl")
        have = {(r["chunk_key"], r["fact_id"]) for r in old}
        add = [r for r in new if (r["chunk_key"], r["fact_id"]) not in have]
        write_jsonl(d / "containment.jsonl", old + add)
        print(f"auto full {len(add)} 쌍 추가 (후보 {len(new)})")
    else:
        lab = Labels.load(a.labels_dir)
        for r in unjudged(chunks, lab):
            sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
