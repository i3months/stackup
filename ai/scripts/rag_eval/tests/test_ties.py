"""동점 민감도 손 계산 단위 테스트 (P-020 §5).

    cd ai/scripts/rag_eval && python -m pytest tests/test_ties.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ties  # noqa: E402

L2_3 = math.log2(3)


def test_tie_groups_and_count():
    # 점수 a=3, b=c=2, d=1, cand=2 → a 1자리 + {b,c} 중 1자리 → 경우의 수 1 × P(2,1)=2
    g = ties.tie_groups({"a": 3, "b": 2, "c": 2, "d": 1}, 2)
    assert g == [(["a"], 1), (["b", "c"], 1)]
    assert ties.n_outcomes(g) == 2
    assert sorted(ties.outcomes(g)) == [["a", "b"], ["a", "c"]]
    # 그룹이 통째로 들어가면 순서만: P(3,3) = 6
    assert ties.n_outcomes(ties.tie_groups({"a": 1, "b": 1, "c": 1}, 5)) == 6


def test_final_minmax_rearrangement():
    # 동점 그룹 {x,y,z} 가 상위 2자리를 차지: 최대 = 높은 등급 2개 내림차순, 최소 = 낮은 등급 2개 오름차순
    best, worst = ties.final_minmax({"x": 1.0, "y": 1.0, "z": 1.0, "w": 0.5}, {"x": 2, "y": 0, "z": 1, "w": 2}, 2)
    assert best == [2, 1] and worst == [0, 1]


def test_single_branch_range():
    # 벡터 단독, a·b 동점, k=1: a(등급 2) 가 앞이면 1, b(0) 가 앞이면 0
    r = ties.single_range({"a": 0.9, "b": 0.9}, {"a": 2, "b": 0}, [2, 0], limit=1, k=1)
    assert (r["ndcg_min"], r["ndcg_max"]) == (0.0, 1.0)


def test_hybrid_hand_enumeration():
    # 벡터: a 단독 1위, {b,c} 동점이 2위 1자리 / FTS: {b,c} 동점 (둘 다 들어감). cand = limit = k = 2.
    # 등급 b=2, a=c=0. IDCG@2 = 3.
    #  [a,b]·[b,c]: b=1/62+1/61 → b,a → [2,0] → 1
    #  [a,b]·[c,b]: b=2/62, a=c=1/61 동점 → [2,0] → 1
    #  [a,c]·[b,c]: c=2/62, a=b=1/61 동점 1자리 → 최대 [0,2] = (3/log2 3)/3, 최소 [0,0] = 0
    #  [a,c]·[c,b]: c=1/62+1/61, a=1/61 > b=1/62 → [0,0] → 0
    vec = {"a": -0.1, "b": -0.2, "c": -0.2}
    fts = {"b": 1.0, "c": 1.0}
    r = ties.hybrid_range(vec, fts, {"a": 0, "b": 2, "c": 0}, [0, 2, 0], cand=2, limit=2, k=2)
    assert r["n_branch_outcomes"] == 4 and not r["sampled"]
    assert r["ndcg_min"] == 0.0 and r["ndcg_max"] == pytest.approx(1.0)
    # 등급순 정렬: 최대 쪽은 b 를 두 브랜치 모두 앞에 → 1, 최소 쪽은 c 를 앞에 → 0
    assert r["gradesort_max"] == pytest.approx(1.0) and r["gradesort_min"] == 0.0


def test_hybrid_no_ties_is_point():
    vec = {"a": -0.1, "b": -0.2, "c": -0.3}
    fts = {"b": 1.0}  # b = 1/62 + 1/61 > a = 1/61 > c = 0 — 최종 동점도 없음
    r = ties.hybrid_range(vec, fts, {"a": 0, "b": 1, "c": 2}, [0, 1, 2], cand=2, limit=2, k=2)
    assert r["n_branch_outcomes"] == 1 and r["width"] == 0.0


def test_sampling_above_limit(monkeypatch):
    monkeypatch.setattr(ties, "EXACT_MAX", 3)
    monkeypatch.setattr(ties, "N_SAMPLES", 50)
    vec = {"a": -0.1, "b": -0.2, "c": -0.2}
    fts = {"b": 1.0, "c": 1.0}
    r = ties.hybrid_range(vec, fts, {"a": 0, "b": 2, "c": 0}, [0, 2, 0], cand=2, limit=2, k=2, seed=1)
    assert r["sampled"] and r["n_evaluated"] == 50
    assert 0.0 <= r["ndcg_min"] <= r["ndcg_max"] <= 1.0


def test_unreachable_is_zero():
    r = ties.single_range({"a": 1.0, "b": 1.0}, {}, [0, 0], limit=2, k=2)
    assert r["ndcg_max"] == 0.0
