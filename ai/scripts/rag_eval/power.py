"""검정력 점검 (SPEC §8, T05 B-4): dev 의 질의 단위 nDCG@5 쌍대 차이로 test 표본이 H2a~H2e 에 충분한지.

    python power.py            # → E2-retrieval/power.json

두 가지를 같이 낸다.
1. 해석적 (설계 효과): 페르소나 ICC 로 설계 효과 deff = 1 + (m̄ − 1)·ICC, 유효 표본 n/deff. 쌍대 차이 SD 로
   정규 근사 검정력 (Wilcoxon 의 점근 상대 효율 3/π ≈ 0.955 를 곱한 유효 표본). 최소 검출 효과(MDE, 검정력 0.8) 도.
2. 모의실험 (군집 부트스트랩): dev 12명에서 페르소나 28명을 복원 추출해 그 페르소나의 질의 차이를 그대로 쓰고,
   Wilcoxon 부호순위(양측)를 α 에서 검정한 기각 비율. 효과는 dev 관측값 그대로다 (차이를 옮기지 않는다 —
   nDCG 차이는 0 이 많아 상수를 더하면 0 이 사라져 검정력이 부풀려진다).
α 는 Holm 5검정의 가장 엄한 단계 0.05/5 = 0.01 과, 가장 느슨한 단계 0.05 를 둘 다 본다.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import now_iso, results_dir, write_json  # noqa: E402
from labels import read_jsonl  # noqa: E402

H2 = {"H2a": ("R-SRC-raw", "B0"), "H2b": ("B0", "R-PFX-none"), "H2c": ("R-RET-hkiwi", "B0"),
      "H2d": ("R-RR-1", "B0"), "H2e": ("R-CAND-4x", "B0")}  # (높다고 가설한 쪽, 비교 대상)
ALPHAS = (0.01, 0.05)
N_SIM = 2000
SEED = 2024
ARE = 3 / math.pi


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _z(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def per_query(cid: str) -> dict[str, dict[str, Any]]:
    m = json.loads((results_dir("E2-retrieval", "dev", f"{cid}-k5") / "metrics.json").read_text(encoding="utf-8"))
    return {r["query_id"]: r for r in m["per_query"] if not r["no_answer"]}


def icc_oneway(groups: list[list[float]]) -> float:
    """ANOVA 추정 ICC(1). 불균형 군집은 n0 보정."""
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / n
    ssb = sum(len(g) * (st.fmean(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((x - st.fmean(g)) ** 2 for x in g) for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    s2b = max((msb - msw) / n0, 0.0)
    return s2b / (s2b + msw) if s2b + msw > 0 else 0.0


def analytic(mean: float, sd: float, n: int, deff: float, alpha: float) -> dict[str, float]:
    n_eff = n / deff * ARE
    za = _z(1 - alpha / 2)
    pw = _phi(abs(mean) / sd * math.sqrt(n_eff) - za) if sd > 0 else 1.0
    mde = (za + _z(0.8)) * sd / math.sqrt(n_eff) if sd > 0 else 0.0
    return {"power": round(pw, 3), "mde_power0.8": round(mde, 4), "n_eff": round(n_eff, 1)}


def simulate(groups: list[list[float]], n_personas: int, alpha: float, rng: random.Random) -> float:
    from scipy.stats import wilcoxon  # noqa: PLC0415

    rej = 0
    for _ in range(N_SIM):
        d = [x for _ in range(n_personas) for x in rng.choice(groups)]
        if all(x == 0 for x in d):
            continue
        rej += wilcoxon(d).pvalue < alpha
    return rej / N_SIM


def main() -> None:
    qs = read_jsonl(results_dir("E1-queries") / "queries.jsonl")
    test = [q for q in qs if q["split"] == "test" and q["gold_fact_ids"]]
    test_personas = sorted({q["persona_id"] for q in test})
    n_test = len(test)  # 정답 있는 test 질의 수 (질의 텍스트·결과는 보지 않고 개수만)
    m_bar = n_test / len(test_personas)
    rng = random.Random(SEED)
    out: dict[str, Any] = {"method": __doc__.strip().splitlines()[2:13], "n_test_answerable": n_test,
                           "n_test_personas": len(test_personas), "m_bar": round(m_bar, 2), "n_sim": N_SIM,
                           "seed": SEED, "ts": now_iso(), "hypotheses": {}}
    for h, (hi, lo) in H2.items():
        a, b = per_query(hi), per_query(lo)
        assert a.keys() == b.keys()
        by_p: dict[str, list[float]] = defaultdict(list)
        for qid in sorted(a):
            by_p[a[qid]["persona_id"]].append(a[qid]["ndcg"] - b[qid]["ndcg"])
        groups = [by_p[p] for p in sorted(by_p)]
        d = [x for g in groups for x in g]
        mean, sd = st.fmean(d), st.stdev(d)
        icc = icc_oneway(groups)
        deff = 1 + (m_bar - 1) * icc
        row = {"comparison": f"{hi} − {lo}", "dev_n": len(d), "dev_mean_diff": round(mean, 4), "dev_sd": round(sd, 4),
               "dev_zero_share": round(sum(x == 0 for x in d) / len(d), 3), "icc": round(icc, 4), "deff": round(deff, 3)}
        for al in ALPHAS:
            row[f"alpha{al}"] = {"analytic": analytic(mean, sd, n_test, deff, al),
                                 "sim_cluster_bootstrap_power": round(simulate(groups, len(test_personas), al, rng), 3)}
        out["hypotheses"][h] = row
        print(h, json.dumps(row, ensure_ascii=False), flush=True)
    write_json(results_dir("E2-retrieval") / "power.json", out)


if __name__ == "__main__":
    main()
