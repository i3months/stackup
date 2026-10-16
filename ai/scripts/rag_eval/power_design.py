"""표본 확대 설계 (P-024 ①): test 페르소나 수 × 페르소나당 질의 수 격자에서 H2a·H2e 검정력.

    python power_design.py        # → E2-retrieval/power-design.json

- 현재 test = 28명 × 정답 있는 질의 13.75 (15개 중 정답 없음 제외). 격자: 추가 페르소나 P ∈ {0, 20, 40, 60}
  × 페르소나당 질의 m ∈ {15, 25} (정답 있는 비율은 지금과 같게 13.75/15). 기존 28명도 m 까지 늘린다고 둔다.
- 모의실험 (군집 2단 부트스트랩): 페르소나를 dev 12명에서 복원 추출하고, 그 페르소나의 질의 차이에서 질의를
  복원 추출해 m_ans 개를 만든다. Wilcoxon 부호순위 양측, α = 0.01 (Holm 최엄 단계), 2,000회.
  효과는 dev 관측값 그대로다 (H2a +0.044 → "검정력 ≥ 0.8" 이 곧 "MDE ≤ 0.045" 에 해당).
- ICC 불확실성: dev 12명 페르소나 부트스트랩으로 ICC 분포를 구하고, 그 80% 분위 ICC 에서도 계산한다. 이때는
  페르소나 평균의 편차를 λ 배 해 ICC 를 맞춘다 (d' = d − d̄_p + d̄ + λ(d̄_p − d̄)).
- 해석식도 같이: n_eff = N·m/(1 + (m − 1)ICC) · 3/π, 상한 (m → ∞) = N/ICC · 3/π.
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
from power import ARE, H2, _phi, _z, icc_oneway, per_query  # noqa: E402

BASE_TEST = 28
ADD = (0, 20, 40, 60)
MS = (15, 25)
ANS_RATE = 385 / 420
ALPHA = 0.01
N_SIM = 2000
N_ICC_BOOT = 2000
SEED = 2025


def groups_for(h: str) -> list[list[float]]:
    hi, lo = H2[h]
    a, b = per_query(hi), per_query(lo)
    by_p: dict[str, list[float]] = defaultdict(list)
    for qid in sorted(a):
        by_p[a[qid]["persona_id"]].append(a[qid]["ndcg"] - b[qid]["ndcg"])
    return [by_p[p] for p in sorted(by_p)]


def with_icc(groups: list[list[float]], target: float) -> tuple[list[list[float]], float]:
    """페르소나 평균 편차를 λ 배 해 ICC 를 target 에 맞춘다 (이분 탐색)."""
    grand = st.fmean([x for g in groups for x in g])

    def scaled(lam: float) -> list[list[float]]:
        return [[x - st.fmean(g) + grand + lam * (st.fmean(g) - grand) for x in g] for g in groups]

    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if icc_oneway(scaled(mid)) < target else (lo, mid)
    return scaled(hi), hi


def sim_power(groups: list[list[float]], n_personas: int, m_ans: int, rng: random.Random) -> float:
    from scipy.stats import wilcoxon  # noqa: PLC0415

    rej = 0
    for _ in range(N_SIM):
        d = []
        for _ in range(n_personas):
            g = rng.choice(groups)
            d += [rng.choice(g) for _ in range(m_ans)]
        if any(x != 0 for x in d):
            rej += wilcoxon(d).pvalue < ALPHA
    return rej / N_SIM


def analytic(mean: float, sd: float, n_personas: int, m_ans: float, icc: float) -> dict[str, float]:
    n_eff = n_personas * m_ans / (1 + (m_ans - 1) * icc) * ARE
    za = _z(1 - ALPHA / 2)
    return {"n_eff": round(n_eff, 1), "power": round(_phi(abs(mean) / sd * math.sqrt(n_eff) - za), 3),
            "mde_power0.8": round((za + _z(0.8)) * sd / math.sqrt(n_eff), 4),
            "n_eff_cap_m_inf": round(n_personas / icc * ARE, 1) if icc > 0 else None}


def main() -> None:
    rng = random.Random(SEED)
    out: dict[str, Any] = {"method": __doc__.strip().splitlines()[2:12], "alpha": ALPHA, "n_sim": N_SIM, "seed": SEED,
                           "ans_rate": round(ANS_RATE, 4), "ts": now_iso(), "hypotheses": {}}
    for h in ("H2a", "H2e"):
        groups = groups_for(h)
        d = [x for g in groups for x in g]
        mean, sd = st.fmean(d), st.stdev(d)
        icc0 = icc_oneway(groups)
        boots = sorted(icc_oneway([rng.choice(groups) for _ in groups]) for _ in range(N_ICC_BOOT))
        icc80 = boots[int(0.8 * (N_ICC_BOOT - 1))]
        g80, lam = with_icc(groups, icc80) if icc80 > icc0 else (groups, 1.0)
        row: dict[str, Any] = {"dev_mean_diff": round(mean, 4), "dev_sd": round(sd, 4), "icc_dev": round(icc0, 4),
                               "icc_boot_p80": round(icc80, 4), "icc_boot_p50": round(boots[N_ICC_BOOT // 2], 4),
                               "lambda_p80": round(lam, 3), "grid": []}
        for add in ADD:
            for m in MS:
                n_p = BASE_TEST + add
                m_ans = round(m * ANS_RATE)
                cell = {"add_personas": add, "n_test_personas": n_p, "m_per_persona": m, "m_answerable": m_ans,
                        "n_answerable": n_p * m_ans,
                        "icc_dev": {"sim_power": round(sim_power(groups, n_p, m_ans, rng), 3),
                                    **analytic(mean, sd, n_p, m_ans, icc0)},
                        "icc_p80": {"sim_power": round(sim_power(g80, n_p, m_ans, rng), 3),
                                    **analytic(mean, sd, n_p, m_ans, icc80)}}
                row["grid"].append(cell)
                print(h, json.dumps(cell, ensure_ascii=False), flush=True)
        out["hypotheses"][h] = row
    write_json(results_dir("E2-retrieval") / "power-design.json", out)


if __name__ == "__main__":
    main()
