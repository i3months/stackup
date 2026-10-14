"""페르소나 슬롯(직군·연차) 40개와 dev/test 분할 — 문서·질의 생성 전에 고정한다 (SPEC §4.1·§4.2).

    python -m corpus.slots      # results/E1-corpus/personas.csv (generator 빈칸) + results/E1-queries/split.json
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import now_iso, results_dir, stackup_commit  # noqa: E402

SPLIT_SEED = 20261014

# SPEC §4.1 직군 분포. job_role 은 운영 JobCategory 값 (model/messages/job_category.py).
JOB_PLAN: list[tuple[str, str, int]] = [
    ("backend", "BACKEND", 10),
    ("frontend", "FRONTEND", 7),
    ("infra", "INFRA", 6),
    ("data", "DATA_AI", 6),
    ("mobile", "MOBILE", 5),
    ("non_eng", "PLANNING", 2),
    ("non_eng", "MARKETING", 2),
    ("non_eng", "DESIGN", 2),
]
YEARS_BANDS = ["신입", "1~3", "4~7", "8+"]
# dev 12 / test 28, 직군(job_family) 층화
DEV_PER_FAMILY = {"backend": 3, "frontend": 2, "infra": 2, "data": 2, "mobile": 1, "non_eng": 2}
ENGINEERING = {"backend", "frontend", "infra", "data", "mobile"}


def build_slots() -> list[dict[str, str]]:
    slots = []
    i = 0
    for family, role, n in JOB_PLAN:
        for _ in range(n):
            # 연차는 전체 순서로 순환 배정 → 밴드마다 정확히 10명, 직군 안에서도 고르게
            slots.append({
                "persona_id": f"P{i + 1:02d}",
                "job_family": family,
                "job_role": role,
                "years_band": YEARS_BANDS[i % 4],
            })
            i += 1
    rng = random.Random(SPLIT_SEED)
    by_family: dict[str, list[dict[str, str]]] = {}
    for s in slots:
        by_family.setdefault(s["job_family"], []).append(s)
    for fam, members in by_family.items():
        dev = set(p["persona_id"] for p in rng.sample(members, DEV_PER_FAMILY[fam]))
        for p in members:
            p["split"] = "dev" if p["persona_id"] in dev else "test"
    return slots


def main() -> None:
    slots = build_slots()
    assert len(slots) == 40 and sum(s["split"] == "dev" for s in slots) == 12
    corpus = results_dir("E1-corpus")
    with (corpus / "personas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["persona_id", "job_family", "job_role", "years_band", "split", "generator"])
        w.writeheader()
        for s in slots:
            w.writerow({**s, "generator": ""})
    split = {
        "seed": SPLIT_SEED,
        "rule": "페르소나 단위, job_family 층화 (SPEC §4.2). dev 12 / test 28",
        "created_at": now_iso(),
        "script": "ai/scripts/rag_eval/corpus/slots.py",
        "stackup_commit": stackup_commit(),
        "dev": [s["persona_id"] for s in slots if s["split"] == "dev"],
        "test": [s["persona_id"] for s in slots if s["split"] == "test"],
    }
    out = results_dir("E1-queries") / "split.json"
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from collections import Counter
    print(Counter((s["job_family"], s["split"]) for s in slots))
    print(Counter(s["years_band"] for s in slots))
    print(out)


if __name__ == "__main__":
    main()
