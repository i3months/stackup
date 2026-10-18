"""지표·라벨 손 계산 단위 테스트 (SPEC §10-7).

    cd ai/scripts/rag_eval && python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labels import Labels, auto_full, unjudged  # noqa: E402
from metrics import (  # noqa: E402
    NotFullyJudged,
    dcg,
    fact_recall_at_k,
    hit_at_k,
    mrr_at_10,
    ndcg_at_k,
    per_query,
    summarize,
)

L2_3 = math.log2(3)


# ---- nDCG 손 계산 ---------------------------------------------------------------
def test_dcg_hand():
    # (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) = 3 + 0 + 0.5
    assert dcg([2, 0, 1]) == pytest.approx(3.5)


def test_ndcg_mixed_grades():
    # 순위 [2,0,1], 전체 청크 등급 [2,2,1,0,0], k=3
    # IDCG = 3 + 3/log2(3) + 1/2 = 5.392789
    assert ndcg_at_k([2, 0, 1], [2, 2, 1, 0, 0], 3) == pytest.approx(3.5 / (3 + 3 / L2_3 + 0.5))


def test_ndcg_swapped():
    # 순위 [1,2], 전체 [2,1], k=2 → (1 + 3/log2(3)) / (3 + 1/log2(3))
    assert ndcg_at_k([1, 2], [2, 1], 2) == pytest.approx((1 + 3 / L2_3) / (3 + 1 / L2_3))


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([2, 1, 0], [0, 1, 2, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], [2, 0, 0, 0], 3) == 0.0


def test_idcg_boundary_fewer_chunks_than_k():
    # 범위 내 청크가 k 보다 적으면 IDCG 도 그 청크들로만 — 다 찾으면 1
    assert ndcg_at_k([2], [2], 5) == pytest.approx(1.0)
    # 관련 청크가 k 밖에만 있어도 IDCG 는 전체 청크 기준 (검색 결과와 무관)
    assert ndcg_at_k([0, 0], [0, 0, 2], 2) == 0.0


def test_idcg_all_zero_is_zero():
    # SPEC v1.3 §6.1: 근거 도달 불가 (질의, 변형) 은 nDCG 0
    assert ndcg_at_k([0, 0], [0, 0, 0], 2) == 0.0


def test_hit_and_mrr():
    assert hit_at_k([1, 1, 2], 3) == 1 and hit_at_k([1, 1, 2], 2) == 0
    assert mrr_at_10([0, 1, 2]) == pytest.approx(1 / 3)
    assert mrr_at_10([0] * 10 + [2]) == 0.0  # 상위 10 밖이면 0
    assert mrr_at_10([1, 1]) == 0.0  # 등급 1 은 세지 않음


# ---- 라벨 -----------------------------------------------------------------------
FACTS = [
    {"fact_id": "F1", "persona_id": "P", "doc_id": "P-resume", "experience_id": "E1", "evidence_text": "정산 배치를 5시간에서 40분으로 줄였다"},
    {"fact_id": "F2", "persona_id": "P", "doc_id": "P-resume", "experience_id": "E1", "evidence_text": "QueryDSL 튜닝과 청크 단위 처리"},
    {"fact_id": "F3", "persona_id": "P", "doc_id": "P-resume", "experience_id": "E2", "evidence_text": "Redis"},
]


def _c(chunk, fact, label):
    return {"chunk_key": chunk, "fact_id": fact, "label": label, "labeler": "test", "ts": ""}


def _labels():
    rows = []
    table = {
        "c1": {"F1": "full", "F2": "none", "F3": "none"},
        "c2": {"F1": "partial", "F2": "none", "F3": "none"},
        "c3": {"F1": "none", "F2": "full", "F3": "none"},  # 같은 경험(E1)의 다른 사실 full
        "c4": {"F1": "none", "F2": "none", "F3": "full"},  # 다른 경험
        "c5": {"F1": "none", "F2": "none", "F3": "none"},
    }
    for c, d in table.items():
        rows += [_c(c, f, lab) for f, lab in d.items()]
    return Labels(rows, FACTS)


def test_rel_grades():
    lab = _labels()
    assert lab.rel("c1", ["F1"]) == 2
    assert lab.rel("c2", ["F1"]) == 1
    assert lab.rel("c3", ["F1"]) == 1
    assert lab.rel("c4", ["F1"]) == 0
    assert lab.rel("c4", ["F1", "F3"]) == 2
    assert lab.rel("c5", []) == 0


def test_rel_unjudged_raises():
    lab = Labels([_c("x", "F1", "none")], FACTS)
    with pytest.raises(KeyError):
        lab.rel("x", ["F1", "F3"])
    with pytest.raises(KeyError):  # 같은 경험 사실 F2 가 미판정
        lab.rel("x", ["F1"])
    assert not lab.judged("x", "P")


def test_fact_recall():
    lab = _labels()
    assert fact_recall_at_k(["c1", "c5"], ["F1", "F3"], lab) == pytest.approx(0.5)
    assert fact_recall_at_k(["c2"], ["F1"], lab) == 0.0  # partial 은 회수로 안 셈


def test_auto_full_rules():
    doc = "정산 배치를 5시간에서 40분으로 줄였다. QueryDSL 튜닝과 청크 단위 처리. Redis 캐시. Redis 락."
    chunks = [
        {"chunk_key": "k1", "doc_id": "P-resume", "part": "raw", "body": "정산 배치를 5시간에서 40분으로 줄였다."},
        {"chunk_key": "k2", "doc_id": "P-resume", "part": "raw", "body": "Redis 캐시. Redis 락."},
        {"chunk_key": "k3", "doc_id": "P-resume", "part": "summary", "body": "정산 배치를 5시간에서 40분으로 줄였다"},
        {"chunk_key": "k4", "doc_id": "P-resume", "part": "raw", "body": "QueryDSL 튜닝과 청크"},  # 경계에 걸림
    ]
    got = {(r["chunk_key"], r["fact_id"]) for r in auto_full(chunks, FACTS, {"P-resume": doc})}
    assert got == {("k1", "F1")}  # F3 'Redis' 는 짧고 중복, k3 는 요약 부분, k4 는 근거가 잘림


def test_unjudged_groups_by_chunk():
    lab = Labels([_c("c1", "F1", "full")], FACTS)
    todo = unjudged([{"chunk_key": "c1", "persona_id": "P", "body": "x"},
                     {"chunk_key": "c1", "persona_id": "P", "body": "x"}], lab)
    assert len(todo) == 1 and todo[0]["fact_ids"] == ["F2", "F3"]


# ---- 질의 단위 집계 -----------------------------------------------------------------
def test_per_query_and_summary():
    lab = _labels()
    queries = {"q1": {"query_id": "q1", "persona_id": "P", "qtype": "followup", "gold_fact_ids": ["F1"]},
               "q2": {"query_id": "q2", "persona_id": "P", "qtype": "followup", "gold_fact_ids": []}}
    rankings = [
        {"query_id": "q1", "k": 2, "n_scope_chunks": 5, "results": [{"chunk_key": "c2", "score": 0.5}, {"chunk_key": "c1", "score": 0.4}]},
        {"query_id": "q2", "k": 2, "n_scope_chunks": 5, "results": [{"chunk_key": "c4", "score": 0.3}, {"chunk_key": "c5", "score": 0.2}]},
    ]
    rows = per_query(rankings, queries, lab, {"P": ["c1", "c2", "c3", "c4", "c5"]})
    q1 = rows[0]
    # 순위 [1,2], 전체 [2,1,1,0,0] → IDCG@2 = 3 + 1/log2(3)
    assert q1["grades"] == [1, 2]
    assert q1["ndcg"] == pytest.approx((1 + 3 / L2_3) / (3 + 1 / L2_3))
    assert q1["fact_recall"] == 1.0 and q1["hit"] == 1
    s = summarize(rows, 2, None)
    assert s["n_answerable"] == 1 and s["no_answer"]["n"] == 1


def test_summary_refuses_unjudged():
    lab = Labels([_c("c1", "F1", "full")], FACTS)  # F2·F3 미판정 → judged 아님
    queries = {"q1": {"query_id": "q1", "persona_id": "P", "qtype": "followup", "gold_fact_ids": ["F1"]}}
    rows = per_query([{"query_id": "q1", "k": 1, "n_scope_chunks": 1, "results": [{"chunk_key": "c1", "score": 1}]}],
                     queries, lab, {"P": ["c1"]})
    with pytest.raises(NotFullyJudged):
        summarize(rows, 1, None)


def test_k10_excludes_small_scope():
    lab = _labels()
    queries = {"q1": {"query_id": "q1", "persona_id": "P", "qtype": "followup", "gold_fact_ids": ["F1"]}}
    rows = per_query([{"query_id": "q1", "k": 10, "n_scope_chunks": 5, "results": [{"chunk_key": "c1", "score": 1}]}],
                     queries, lab, {"P": ["c1", "c2", "c3", "c4", "c5"]})
    s = summarize(rows, 10, None)
    assert s["n_answerable"] == 0 and s["excluded_small_scope"] == 1


def test_labeler_priority_human_over_auto_over_llm():
    def r(label, labeler):
        return {"chunk_key": "c", "fact_id": "F1", "label": label, "labeler": labeler, "ts": ""}

    # 원문 변형: LLM 이 none 이라고 해도 자동 full 이 덮어쓴다 (행 순서와 무관)
    assert Labels([r("full", "auto"), r("none", "claude-haiku-5-5")], FACTS).label[("c", "F1")] == "full"
    assert Labels([r("none", "claude-haiku-5-5"), r("full", "auto")], FACTS).label[("c", "F1")] == "full"
    # 사람 판정이 가장 우선
    assert Labels([r("full", "auto"), r("partial", "human:a")], FACTS).label[("c", "F1")] == "partial"
    # 같은 등급의 다른 판정자면 나중 행
    assert Labels([r("none", "m"), r("partial", "n")], FACTS).label[("c", "F1")] == "partial"


def test_duplicate_rows_first_wins_and_counted():
    """같은 (chunk_key, fact_id, labeler, rubric_version) 가 다시 나오면 첫 행만 쓰고 센다 (P-030)."""
    def r(label, rubric="v2"):
        return {"chunk_key": "c", "fact_id": "F1", "label": label, "labeler": "m", "rubric_version": rubric, "ts": ""}

    lab = Labels([r("none"), r("partial"), r("none")], FACTS)
    assert lab.label[("c", "F1")] == "none" and lab.n_duplicates == 2
    # 루브릭이 다르면 중복이 아니다 (같은 등급이라 나중 행)
    lab = Labels([r("none", "v1"), r("partial", "v2")], FACTS)
    assert lab.label[("c", "F1")] == "partial" and lab.n_duplicates == 0
