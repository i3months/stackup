"""사람 검증 시트 (T04, SPEC §4.3-5) 와 채점.

    python human_sheets.py make  [--n-pairs 300] [--n-queries 50] [--rankings E2-retrieval/B0-k5-dev]
    python human_sheets.py score --rater <이름>      # human/<이름>/ 에 채운 CSV → κ·누락 비율

(a) 포함 판정 300쌍: 변형 × LLM 라벨 × 사실 유형으로 층화. 시트에는 청크 본문과 사실 문장만 보이고 LLM 판정은
    정답 키 파일(answer-key-*.jsonl, 평가자에게 주지 않는다)에만 있다.
(b) 누락 관련성: 질의 50개 × B0 상위 5 청크. "근거 사실 외에 이 질의에 관련된 정보가 있는가".
시트는 UTF-8 BOM CSV (엑셀에서 바로 열림) + 안내 1쪽(README.md). 평가자 라벨은 human/<평가자>/ 아래에 둔다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import results_dir, write_json, write_jsonl  # noqa: E402
from labels import LABELS, labeler_rank, read_jsonl  # noqa: E402

SEED = 5150
HUMAN = "human"

GUIDE = """# 사람 검증 안내 (E1b, 약 2~3시간)

검색 실험의 정답 라벨을 사람이 확인하는 작업입니다. 지원자 문서는 모두 **합성 데이터**(가상 인물)입니다.
두 시트를 각자 독립적으로 채워 주세요. 다른 평가자나 AI 판정을 보지 마세요.

## 시트 A — `sheet-a-containment.csv` (300행, 약 1.5시간)

각 행에는 문서 조각(`chunk_text`)과 지원자에 관한 사실 한 문장(`fact`)이 있습니다.
`label` 칸에 아래 셋 중 하나를 쓰세요.

| label | 뜻 | 예 (사실: "쿼리 응답을 450ms 에서 120ms 로 줄였다") |
|---|---|---|
| `full` | 조각이 사실의 핵심 정보(수치·이름·도구·설정값·결과)를 담고 있다. 다른 말로 바꿔 쓴 것도 포함 | "쿼리 응답 시간을 450ms→120ms 로 단축" |
| `partial` | 조각이 **바로 그 구체적인 행동·결과**를 말하지만 핵심 값이 빠졌다 | "쿼리 응답 속도를 크게 개선했다" |
| `none` | 그 외. 같은 프로젝트·회사·도구·주제만 나오고 그 행동·결과는 말하지 않으면 none | "QueryDSL 을 사용했다" |

- 조각에 쓰인 것만 보고 판단합니다. 애매하면 `none` 쪽으로.
- 확신이 없으면 `note` 칸에 이유를 적어 주세요 (라벨은 꼭 고릅니다).

## 시트 B — `sheet-b-omission.csv` (질의 50개 × 5조각 = 250행, 약 1시간)

각 행에는 면접 상황의 질의(`query`), 그 질의가 원래 노린 사실(`target_facts`), 검색된 문서 조각(`chunk_text`)이 있습니다.
`other_relevant` 칸에 `yes` 또는 `no` 를 쓰세요.

- `yes`: 조각에 **target_facts 말고도** 이 질의에 답하거나 꼬리질문을 만드는 데 쓸 만한 지원자 정보가 있다.
- `no`: 그런 정보가 없다 (target_facts 만 있거나, 관련 없는 내용).
- 질의가 "정답 없음"(target_facts 가 비어 있음)이면, 조각에 질의와 관련된 지원자 정보가 있는지만 봅니다.

## 제출

채운 CSV 두 개를 `human/<본인 이름>/` 폴더에 그대로 저장해 주세요. 파일 이름과 열은 바꾸지 마세요.
"""


def _dir() -> Path:
    return results_dir("E1-labels", HUMAN)


def _llm_labels() -> dict[tuple[str, str], tuple[str, str]]:
    """(chunk_key, fact_id) → (label, labeler) — Labels 와 같은 우선순위 (사람 라벨 제외)."""
    best: dict[tuple[str, str], tuple[str, str, int]] = {}
    for r in read_jsonl(results_dir("E1-labels") / "containment.jsonl"):
        if r["labeler"].startswith("human:"):
            continue
        k, rk = (r["chunk_key"], r["fact_id"]), labeler_rank(r["labeler"])
        if rk >= best.get(k, ("", "", 0))[2]:
            best[k] = (r["label"], r["labeler"], rk)
    return {k: (v[0], v[1]) for k, v in best.items()}


def make_sheet_a(n: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts = {f["fact_id"]: f for f in read_jsonl(results_dir("E1-corpus") / "facts.jsonl")}
    labels = _llm_labels()
    chunks = read_jsonl(results_dir("E1-labels") / "chunks.jsonl")
    first_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for c in chunks:
        first_variant.setdefault((c["persona_id"], c["chunk_key"]), c)
    cells: dict[tuple, list] = defaultdict(list)
    for (ck, fid), (lab, labeler) in labels.items():
        f = facts[fid]
        c = first_variant.get((f["persona_id"], ck))
        if c is None:
            continue
        cells[(c["variant_id"].split("-")[0], lab, f["fact_type"])].append((c, f, lab, labeler))
    # 층화: (부분 summary/raw) × 라벨 × 사실 유형 칸에서 고르게. none 이 압도적이라 라벨은 균등 배분.
    by_lab: dict[str, list[tuple]] = defaultdict(list)
    for key in cells:
        by_lab[key[1]].append(key)
    pick = []
    per_lab = n // len(LABELS)
    for lab in LABELS:
        keys = sorted(by_lab[lab])
        quota = per_lab + (n - per_lab * len(LABELS) if lab == "none" else 0)
        i = 0
        pools = {k: rng.sample(cells[k], len(cells[k])) for k in keys}
        while quota > 0 and any(pools.values()):
            k = keys[i % len(keys)]
            if pools[k]:
                pick.append(pools[k].pop())
                quota -= 1
            i += 1
    rng.shuffle(pick)
    sheet, key = [], []
    for i, (c, f, lab, labeler) in enumerate(pick, 1):
        sheet.append({"item_id": f"A{i:03d}", "chunk_text": c["body"], "fact": f["statement"], "label": "", "note": ""})
        key.append({"item_id": f"A{i:03d}", "chunk_key": c["chunk_key"], "fact_id": f["fact_id"], "variant_id": c["variant_id"],
                    "fact_type": f["fact_type"], "llm_label": lab, "llm_labeler": labeler,
                    "generator": f.get("generator")})
    return sheet, key


def make_sheet_b(n_q: int, rankings: str, rng: random.Random) -> tuple[list[dict], list[dict]]:
    facts = {f["fact_id"]: f for f in read_jsonl(results_dir("E1-corpus") / "facts.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(results_dir("E1-queries") / "queries.jsonl")}
    chunks = {(c["doc_id"], c["chunk_index"]): c for c in read_jsonl(results_dir("E1-labels") / "chunks.jsonl")
              if c["variant_id"] == "summary-c1000o200"}
    rows = read_jsonl(results_dir(rankings) / "rankings.jsonl")
    # 질의 유형·정답 없음 비율을 유지하도록 층화
    by = defaultdict(list)
    for r in rows:
        q = queries[r["query_id"]]
        by[(q["qtype"], q["no_answer"])].append(r)
    total = sum(len(v) for v in by.values())
    pick = []
    for k in sorted(by):
        m = max(1, round(n_q * len(by[k]) / total))
        pick += rng.sample(by[k], min(m, len(by[k])))
    pick = rng.sample(pick, min(n_q, len(pick)))
    sheet, key = [], []
    for r in pick:
        q = queries[r["query_id"]]
        tf = "\n".join(f"- {facts[f]['statement']}" for f in q["gold_fact_ids"]) or "(정답 없음)"
        for x in r["results"]:
            c = chunks[(x["doc_id"], x["chunk_index"])]
            iid = f"B-{q['query_id']}-{x['rank']}"
            sheet.append({"item_id": iid, "query": q["query_text"][:2000], "target_facts": tf, "chunk_text": c["body"],
                          "other_relevant": "", "note": ""})
            key.append({"item_id": iid, "query_id": q["query_id"], "rank": x["rank"], "chunk_key": c["chunk_key"],
                        "qtype": q["qtype"], "no_answer": q["no_answer"], "generator": q.get("generator")})
    return sheet, key


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make(n_pairs: int, n_queries: int, rankings: str) -> dict[str, Any]:
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    a, ak = make_sheet_a(n_pairs, rng)
    b, bk = make_sheet_b(n_queries, rankings, rng)
    _write_csv(d / "sheet-a-containment.csv", a)
    _write_csv(d / "sheet-b-omission.csv", b)
    write_jsonl(d / "answer-key-a.jsonl", ak)
    write_jsonl(d / "answer-key-b.jsonl", bk)
    (d / "README.md").write_text(GUIDE, encoding="utf-8")
    info = {"sheet_a": len(a), "sheet_a_llm_labels": dict(Counter(k["llm_label"] for k in ak)),
            "sheet_a_variants": dict(Counter(k["variant_id"].split("-")[0] for k in ak)),
            "sheet_a_fact_types": dict(Counter(k["fact_type"] for k in ak)),
            "sheet_b_rows": len(b), "sheet_b_queries": len({k["query_id"] for k in bk}), "rankings": rankings, "seed": SEED}
    write_json(d / "manifest.json", info)
    return info


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def score(rater: str) -> dict[str, Any]:
    from labels_judge import cohen_kappa  # noqa: PLC0415

    d = _dir()
    out: dict[str, Any] = {"rater": rater}
    ak = {k["item_id"]: k for k in read_jsonl(d / "answer-key-a.jsonl")}
    a = [r for r in _read_csv(d / rater / "sheet-a-containment.csv") if r["label"].strip()]
    bad = [r["item_id"] for r in a if r["label"].strip().lower() not in LABELS]
    pairs = [(r["label"].strip().lower(), ak[r["item_id"]]["llm_label"]) for r in a if r["item_id"] not in bad]
    if pairs:
        h, m = [x for x, _ in pairs], [y for _, y in pairs]
        out["a"] = {"n": len(pairs), "invalid": bad, "kappa_3class": round(cohen_kappa(h, m), 3),
                    "kappa_full_vs_rest": round(cohen_kappa([x == "full" for x in h], [y == "full" for y in m]), 3),
                    "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3),
                    "confusion_human_to_llm": dict(Counter(f"{x}->{y}" for x, y in pairs))}
        # 라벨러와 같은 Anthropic 계열이 만든 자료는 따로 (SPEC §4.3-5). E1a 문서는 M-M 생성이라 해당 없음이 기본.
        anth = [(x, y) for (x, y), r in zip(pairs, a) if "claude" in str(ak[r["item_id"]].get("generator") or "")]
        out["a"]["anthropic_generated_n"] = len(anth)
    bk = {k["item_id"]: k for k in read_jsonl(d / "answer-key-b.jsonl")}
    b = [r for r in _read_csv(d / rater / "sheet-b-omission.csv") if r["other_relevant"].strip()]
    yes = [r for r in b if r["other_relevant"].strip().lower() in ("yes", "y", "예")]
    by_q = defaultdict(list)
    for r in b:
        by_q[bk[r["item_id"]]["query_id"]].append(r["other_relevant"].strip().lower() in ("yes", "y", "예"))
    out["b"] = {"n_rows": len(b), "yes_rows": len(yes), "omission_rate_rows": round(len(yes) / max(len(b), 1), 3),
                "queries": len(by_q), "queries_with_any_omission": sum(any(v) for v in by_q.values()),
                "omission_rate_queries": round(sum(any(v) for v in by_q.values()) / max(len(by_q), 1), 3)}
    write_json(d / rater / "score.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--rankings", default="E2-retrieval/B0-k5-dev")
    ap.add_argument("--rater", default="")
    a = ap.parse_args()
    if a.cmd == "make":
        print(json.dumps(make(a.n_pairs, a.n_queries, a.rankings), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(score(a.rater), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
