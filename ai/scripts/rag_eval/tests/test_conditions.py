"""조건 설정 파일 검증 — 19개, OFAT, 필드 형식."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditions as C  # noqa: E402

# 모델에 묶인 속성 — EMB 요인 수준을 바꾸면 함께 바뀐다
MODEL_BOUND = {"query_instruction", "dtype", "truncate_dim"}
META = {"condition_id", "description", "exploratory"}


def test_nineteen_conditions():
    allc = C.load_all()
    assert len(allc) == 19
    assert sum(not c.exploratory for c in allc.values()) == 17
    assert {"R-EMB-4m", "R-EMB-1-bf16"} == {k for k, c in allc.items() if c.exploratory}


def test_one_factor_at_a_time():
    allc = C.load_all()
    b0 = asdict(allc["B0"])
    for cid, c in allc.items():
        diff = {k for k, v in asdict(c).items() if b0[k] != v} - META
        if cid == "B0":
            assert not diff
            continue
        factors = diff - MODEL_BOUND
        # 청킹은 size·overlap·mode 를 한 요인으로 본다
        if factors & {"chunk_size", "chunk_overlap", "chunk_mode"}:
            factors = (factors - {"chunk_size", "chunk_overlap", "chunk_mode"}) | {"chunking"}
        assert len(factors) <= 1, (cid, diff)
        if cid == "R-EMB-1-bf16":
            assert diff == {"dtype"}


def test_query_instruction_policy():
    allc = C.load_all()
    for c in allc.values():
        want = "query" if c.embedder in ("EMB-3", "EMB-4") else "none"
        assert c.query_instruction == want, c.condition_id
        if c.embedder == "EMB-4":
            assert c.dtype == "bf16"


def test_variant_and_index_keys():
    allc = C.load_all()
    assert allc["B0"].variant_id == "summary-c1000o200"
    assert allc["R-CH-head"].variant_id == "summary-h2000o0"
    assert allc["R-RR-1"].index_key == allc["B0"].index_key  # 재순위는 인덱스 재사용
    assert allc["R-EMB-4m"].index_key != allc["R-EMB-4"].index_key
    assert allc["B0"].cand_for(5) == 5 and allc["R-CAND-4x"].cand_for(5) == 20
    assert allc["R-RR-1"].cand_for(3) == 20 and allc["R-RR-1"].limit_for(3) == 20


def test_validation_rejects_bad_field(tmp_path):
    d = asdict(C.load("B0")) | {"retrieval": "magic"}
    with pytest.raises(ValueError):
        C.validate(d, "x")
    d = asdict(C.load("B0"))
    d.pop("cand")
    with pytest.raises(ValueError):
        C.validate(d, "x")
