"""검색 조건 설정 (conditions/*.yaml) 로더 — 조건 추가는 파일 추가만으로 된다 (SPEC §10-1).

- 변형 ID(variant_id) = 인덱싱 대상 + 청킹 설정. 같은 변형이면 청크 본문이 같다.
- 인덱스 키(index_key) = 변형 + 접두 + 임베딩(정밀도·절단). 같은 키면 DB 의 같은 임베딩을 재사용한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

COND_DIR = Path(__file__).resolve().parent / "conditions"

_ENUMS = {
    "index_source": {"summary", "raw", "both"},
    "chunk_mode": {"recursive", "heading"},
    "prefix": {"deterministic", "none", "llm"},
    "embedder": {"EMB-1", "EMB-2", "EMB-3", "EMB-4"},
    "query_instruction": {"none", "query"},
    "dtype": {"fp32", "bf16"},
    "retrieval": {"hybrid_simple", "vector", "hybrid_kiwi", "bm25_kiwi"},
    "reranker": {"none", "RR-1", "RR-2"},
}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    description: str
    index_source: str
    chunk_size: int
    chunk_overlap: int
    chunk_mode: str
    prefix: str
    embedder: str
    query_instruction: str
    dtype: str
    truncate_dim: int | None
    retrieval: str
    cand: str | int
    reranker: str
    rerank_stage1: int
    exploratory: bool

    @property
    def variant_id(self) -> str:
        mode = "c" if self.chunk_mode == "recursive" else "h"
        return f"{self.index_source}-{mode}{self.chunk_size}o{self.chunk_overlap}"

    @property
    def embed_id(self) -> str:
        s = f"{self.embedder}-{self.dtype}"
        return s + (f"-t{self.truncate_dim}" if self.truncate_dim else "")

    @property
    def index_key(self) -> str:
        return f"{self.variant_id}|{self.prefix}|{self.embed_id}"

    def cand_for(self, k: int) -> int:
        """운영 의미: 브랜치별 후보 수 = k (SPEC §5.1). 재순위 조건은 1단계 cand = limit = rerank_stage1."""
        if self.reranker != "none":
            return self.rerank_stage1
        if self.cand == "k":
            return k
        if self.cand == "4k":
            return 4 * k
        return int(self.cand)

    def limit_for(self, k: int) -> int:
        return self.rerank_stage1 if self.reranker != "none" else k

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"variant_id": self.variant_id, "index_key": self.index_key}


def validate(d: dict[str, Any], src: str) -> Condition:
    missing = set(Condition.__dataclass_fields__) - set(d)
    extra = set(d) - set(Condition.__dataclass_fields__)
    if missing or extra:
        raise ValueError(f"{src}: 필드 누락 {sorted(missing)} / 알 수 없는 필드 {sorted(extra)}")
    for k, allowed in _ENUMS.items():
        if d[k] not in allowed:
            raise ValueError(f"{src}: {k}={d[k]!r} 는 {sorted(allowed)} 중 하나여야 한다")
    if not (isinstance(d["cand"], int) or d["cand"] in ("k", "4k")):
        raise ValueError(f"{src}: cand={d['cand']!r}")
    if not 0 <= d["chunk_overlap"] < d["chunk_size"]:
        raise ValueError(f"{src}: chunk_overlap 범위")
    if d["truncate_dim"] is not None and d["embedder"] != "EMB-4":
        raise ValueError(f"{src}: truncate_dim 은 EMB-4 에만")
    return Condition(**d)


def load(condition_id: str) -> Condition:
    p = COND_DIR / f"{condition_id}.yaml"
    return validate(yaml.safe_load(p.read_text(encoding="utf-8")), p.name)


def load_all() -> dict[str, Condition]:
    out = {}
    for p in sorted(COND_DIR.glob("*.yaml")):
        c = validate(yaml.safe_load(p.read_text(encoding="utf-8")), p.name)
        if c.condition_id != p.stem:
            raise ValueError(f"{p.name}: condition_id {c.condition_id} ≠ 파일명")
        out[c.condition_id] = c
    return out
