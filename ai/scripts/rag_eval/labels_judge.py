"""포함 판정 (T04, SPEC §4.3-1·3·4). LLM 판정 1회 = 텍스트 1개(청크 본문 또는 접두) + 그 페르소나 사실 전부.

    python labels_judge.py chunks   --variants summary-c1000o200,raw-c1000o200 [--personas P03,P26] [--model haiku] [--batch]
    python labels_judge.py prefixes [--personas ...] [--model haiku] [--batch]
    python labels_judge.py auto     --variants raw-c1000o200,...      # 원문 변형 자동 full (LLM 판정 뒤에 덮어쓴다)
    python labels_judge.py sample-kappa --personas P03,P26,P37 --n 180  # Haiku–Sonnet κ 묶음 (파일럿)

- 같은 (페르소나, chunk_key) 는 변형이 달라도 한 번만 판정한다 (본문이 같으면 같은 판정).
- 루브릭과 그 페르소나 사실 목록을 고정 앞부분(캐시)으로 두고 텍스트를 뒤에 붙인다. 구조화 출력, effort low.
- 결과: E1-labels/containment.jsonl (labeler = 모델 ID 또는 auto), prefix_containment.jsonl.
  주 판정자가 아닌 모델(κ 비교용 Sonnet)은 containment-<model>.jsonl 에 따로 둔다 (Labels 가 섞어 읽지 않게).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl, results_dir, write_jsonl  # noqa: E402
from labels import LABELS, auto_full, read_jsonl  # noqa: E402

TASK_ID = "T04"
BUDGET = 20.0
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
PRIMARY = MODELS["haiku"]
SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact_id": {"type": "string"}, "label": {"type": "string", "enum": list(LABELS)}},
        "required": ["fact_id", "label"], "additionalProperties": False}}},
    "required": ["judgments"], "additionalProperties": False,
}
RUBRIC_V1 = (  # T04 파일럿 1차 (Haiku–Sonnet κ 0.525: Haiku partial 의 72% 가 Sonnet none)
    "당신은 검색 평가 라벨러입니다. 한국어 지원자 문서에서 잘라 낸 텍스트 조각 하나와, 그 지원자에 관한 사실 목록이 "
    "주어집니다. 사실마다 그 텍스트가 사실을 담고 있는지 판정하세요.\n"
    "- full: 텍스트가 사실의 핵심 정보(수치·이름·도구·설정값·결과 등)를 담고 있다. 다른 말로 바꿔 쓴 것도 포함한다.\n"
    "- partial: 같은 사실(같은 일·같은 대상)을 언급하지만 핵심 값이 빠졌다.\n"
    "- none: 그 사실을 언급하지 않는다. 같은 도구나 비슷한 수치가 나와도 다른 경험·다른 일이면 none 이다.\n"
    "텍스트에 있는 것만 보고 판정하세요. 텍스트 밖의 지식이나 다른 사실로 추론하지 마세요. "
    "모든 fact_id 에 대해 정확히 한 번씩 답하세요."
)
RUBRIC_V2 = (  # partial 을 좁힌다: 같은 '구체적 행동·결과'를 가리켜야 하고, 같은 주제·도구·프로젝트만으로는 none
    "당신은 검색 평가 라벨러입니다. 한국어 지원자 문서에서 잘라 낸 텍스트 조각 하나와, 그 지원자에 관한 사실 목록이 "
    "주어집니다. 사실마다 그 텍스트가 사실을 담고 있는지 판정하세요.\n"
    "- full: 텍스트가 사실의 핵심 정보(수치·이름·도구·설정값·결과 등)를 담고 있다. 다른 말로 바꿔 쓴 것도 포함한다.\n"
    "- partial: 텍스트가 **바로 그 구체적인 행동이나 결과**(같은 대상에 같은 일을 한 것)를 가리키지만, 사실의 핵심 값"
    "(수치·이름·설정값 등)이 빠졌다. 예: 사실 '쿼리 응답을 450ms 에서 120ms 로 줄였다' ↔ 텍스트 '쿼리 응답 속도를 크게 "
    "개선했다'.\n"
    "- none: 위에 해당하지 않는다. 같은 프로젝트·같은 회사·같은 도구·같은 주제가 나오기만 하고 그 구체적인 행동·결과를 "
    "말하지 않으면 none 이다. 예: 사실 'Redis 분산 락으로 중복 예약을 막았다' ↔ 텍스트 'Redis 로 세션을 관리했다' 는 none.\n"
    "텍스트에 있는 것만 보고 판정하세요. 텍스트 밖의 지식이나 다른 사실로 추론하지 마세요. 애매하면 none 쪽으로 판정하세요. "
    "모든 fact_id 에 대해 정확히 한 번씩 답하세요."
)
RUBRICS = {"v1": RUBRIC_V1, "v2": RUBRIC_V2}
RUBRIC = RUBRIC_V2


def _labels_dir() -> Path:
    return results_dir("E1-labels")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def load_facts() -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in read_jsonl(results_dir("E1-corpus") / "facts.jsonl"):
        by[f["persona_id"]].append(f)
    return by


def facts_block(facts: list[dict[str, Any]]) -> str:
    return "[사실 목록]\n" + "\n".join(f"- {f['fact_id']}: {f['statement']}" for f in facts)


def request(model: str, facts: list[dict[str, Any]], text: str, kind: str, rubric: str = "v2") -> dict[str, Any]:
    from llm_client import cached_text  # noqa: PLC0415

    return {
        "model": model, "max_tokens": 4096, "output_config": {"effort": "low"},
        "system": [cached_text(RUBRICS[rubric])],
        "messages": [{"role": "user", "content": [
            cached_text(facts_block(facts)),  # 페르소나마다 같은 앞부분
            {"type": "text", "text": f"[{kind}]\n{text}"},
        ]}],
    }


def _short(model: str) -> str:
    return "h" if "haiku" in model else "s"


def run_requests(items: list[dict[str, Any]], model: str, *, batch: bool, run_id: str, rubric: str = "v2") -> dict[str, Any]:
    """items: {cid, persona_id, text, kind, fact_ids}. 반환: cid → {fact_id: label} (빠진 판정은 없음)."""
    from llm_client import LabClient, json_of  # noqa: PLC0415

    facts = load_facts()
    lab = LabClient(task_id=TASK_ID, budget_usd=BUDGET, run_id=run_id)
    reqs = [(it["cid"], request(model, [f for f in facts[it["persona_id"]] if f["fact_id"] in it["fact_ids"]],
                                it["text"], it["kind"], rubric)) for it in items]
    out: dict[str, dict[str, str]] = {}
    per = 0.0004 if "haiku" in model else 0.008  # 보수적 추정 (Batch 할인 반영 전)
    if batch:
        for i in range(0, len(reqs), 5000):
            part = reqs[i:i + 5000]
            bid = lab.submit_batch(part, estimate_usd=per * len(part) * 0.5, json_schema=SCHEMA)
            # 기다리기 전에 batch id 를 디스크에 남긴다 — 중간에 죽어도 `recover --batches` 로 재제출 없이 회수 (T04 복구)
            append_jsonl(_labels_dir() / "batches.jsonl", [{"batch_id": bid, "run_id": run_id, "model": model,
                                                           "rubric": rubric, "n": len(part), "ts": _now()}])
            print(f"batch {bid} ({len(part)}건) 대기", flush=True)
            lab.wait_batch(bid, poll_sec=30)
            res = lab.collect_batch(bid)
            for cid, _ in part:
                if res.get(cid, {}).get("type") == "succeeded":
                    out[cid] = {j["fact_id"]: j["label"] for j in json_of(res[cid]["message"])["judgments"]}
    else:
        for cid, r in reqs:
            msg = lab.call(estimate_usd=per, json_schema=SCHEMA, **r)
            out[cid] = {j["fact_id"]: j["label"] for j in json_of(msg)["judgments"]}
    return {"labels": out, "spent_task_usd": round(lab.spent_usd(), 6)}


# ---- 청크 판정 --------------------------------------------------------------------
def chunk_items(variants: list[str], personas: set[str] | None, model: str, done: set[tuple[str, str]],
                fact_ids_of: dict[str, list[str]]) -> list[dict[str, Any]]:
    seen, items = set(), []
    for c in read_jsonl(_labels_dir() / "chunks.jsonl"):
        if c["variant_id"] not in variants or (personas and c["persona_id"] not in personas):
            continue
        key = (c["persona_id"], c["chunk_key"])
        if key in seen:
            continue
        seen.add(key)
        missing = [f for f in fact_ids_of[c["persona_id"]] if (c["chunk_key"], f) not in done]
        if missing:
            items.append({"cid": f"{_short(model)}-{c['persona_id']}-{c['chunk_key'][:48]}", "persona_id": c["persona_id"],
                          "chunk_key": c["chunk_key"], "text": c["body"], "kind": "텍스트 조각", "fact_ids": missing})
    return items


def judge_chunks(variants: list[str], personas: set[str] | None, model: str, *, batch: bool,
                 out_name: str = "containment.jsonl", items: list[dict[str, Any]] | None = None,
                 rubric: str = "v2") -> dict[str, Any]:
    path = _labels_dir() / out_name
    old = read_jsonl(path)
    done = {(r["chunk_key"], r["fact_id"]) for r in old if r["labeler"] == model}
    facts = load_facts()
    fact_ids_of = {p: [f["fact_id"] for f in fs] for p, fs in facts.items()}
    items = items if items is not None else chunk_items(variants, personas, model, done, fact_ids_of)
    if not items:
        return {"requests": 0}
    res = run_requests(items, model, batch=batch, run_id=f"E1b-contain-{_short(model)}", rubric=rubric)
    ts, rows, missing = _now(), [], 0
    for it in items:
        got = res["labels"].get(it["cid"])
        if got is None:
            missing += len(it["fact_ids"])
            continue
        for f in it["fact_ids"]:
            if got.get(f) in LABELS:
                rows.append({"chunk_key": it["chunk_key"], "fact_id": f, "label": got[f], "labeler": model,
                             "persona_id": it["persona_id"], "rubric_version": rubric, "ts": ts})
            else:
                missing += 1
    append_jsonl(path, rows)  # 기존 행은 그대로 두고 덧붙인다
    return {"requests": len(items), "pairs": len(rows), "missing_pairs": missing, "task_spent_usd": res["spent_task_usd"]}


def recover_chunks(variants: list[str], model: str, batch_ids: list[str], rubric: str = "v2") -> dict[str, Any]:
    """제출은 끝났는데 containment 를 쓰기 전에 프로세스가 죽은 경우 (T04: 2026-10-14 전원 꺼짐).
    같은 미판정 묶음을 다시 만들어 custom_id 로 이미 끝난 Batch 결과와 맞춘다. 재제출 없음. collect_batch 는 원장에
    빠진 (batch_id, custom_id) 만 기록한다."""
    from llm_client import LabClient, json_of  # noqa: PLC0415

    path = _labels_dir() / "containment.jsonl"
    old = read_jsonl(path)
    done = {(r["chunk_key"], r["fact_id"]) for r in old if r["labeler"] == model}
    facts = load_facts()
    items = {it["cid"]: it for it in chunk_items(variants, None, model, done,
                                                 {p: [f["fact_id"] for f in fs] for p, fs in facts.items()})}
    lab = LabClient(task_id=TASK_ID, budget_usd=BUDGET, run_id=f"E1b-contain-{_short(model)}")
    ts, rows, matched, missing = _now(), [], 0, 0
    for bid in batch_ids:
        res = lab.collect_batch(bid)
        for cid, r in res.items():
            it = items.get(cid)
            if it is None or r["type"] != "succeeded":
                continue
            matched += 1
            got = {j["fact_id"]: j["label"] for j in json_of(r["message"])["judgments"]}
            for f in it["fact_ids"]:
                if got.get(f) in LABELS:
                    rows.append({"chunk_key": it["chunk_key"], "fact_id": f, "label": got[f], "labeler": model,
                                 "persona_id": it["persona_id"], "rubric_version": rubric, "ts": ts})
                else:
                    missing += 1
    append_jsonl(path, rows)  # 기존 행은 그대로 두고 덧붙인다
    return {"unjudged_items": len(items), "matched": matched, "pairs": len(rows), "missing_pairs": missing,
            "task_spent_usd": round(lab.spent_usd(), 6)}


def auto_pass(variants: list[str]) -> dict[str, Any]:
    """원문 변형 자동 full (SPEC §4.3-1). LLM 판정 뒤에 덮어쓰는 행으로 추가한다 (Labels 는 auto 를 LLM 보다 우선)."""
    from index import load_corpus  # noqa: PLC0415  (newline="" 원문)

    raw_docs = {d.doc_id: d.raw_text for d in load_corpus("E1")}
    chunks = [c for c in read_jsonl(_labels_dir() / "chunks.jsonl") if c["variant_id"] in variants]
    facts = read_jsonl(results_dir("E1-corpus") / "facts.jsonl")
    path = _labels_dir() / "containment.jsonl"
    old = read_jsonl(path)
    have = {(r["chunk_key"], r["fact_id"]) for r in old if r["labeler"] == "auto"}
    add = [r for r in auto_full(chunks, facts, raw_docs) if (r["chunk_key"], r["fact_id"]) not in have]
    append_jsonl(path, add)
    return {"auto_full_added": len(add)}


# ---- 접두 판정 (SPEC §4.3-4) -------------------------------------------------------
def judge_prefixes(personas: set[str] | None, model: str, *, batch: bool) -> dict[str, Any]:
    """결정적 접두의 사실을 담는 부분은 문서 요약(≤200자)이라 문서마다 한 번 판정한다 (헤딩 경로는 청크마다 다르지만
    사실을 담지 않는다). PFX-llm 접두는 청크마다 판정한다."""
    facts = load_facts()
    corpus = results_dir("E1-corpus")
    path = _labels_dir() / "prefix_containment.jsonl"
    old = read_jsonl(path)
    done = {(r["prefix_id"], r["variant_id"], r["doc_id"], r["chunk_index"]) for r in old if r["labeler"] == model}
    items = []
    for a in sorted((corpus / "analysis").glob("P*.json")):
        d = json.loads(a.read_text(encoding="utf-8"))
        pid = d["doc_id"].split("-")[0]
        if personas and pid not in personas or ("deterministic", "*", d["doc_id"], None) in done:
            continue
        items.append({"cid": f"{_short(model)}-pd-{d['doc_id']}", "persona_id": pid, "text": d["summary"],
                      "kind": "문서 요약 접두", "fact_ids": [f["fact_id"] for f in facts[pid]],
                      "key": ("deterministic", "*", d["doc_id"], None)})
    for r in read_jsonl(_labels_dir() / "prefixes.jsonl"):
        if r["prefix_id"] != "llm":
            continue
        pid = r["doc_id"].split("-")[0]
        k = ("llm", r["variant_id"], r["doc_id"], r["chunk_index"])
        if personas and pid not in personas or k in done:
            continue
        items.append({"cid": f"{_short(model)}-pl-{r['variant_id'][:20]}-{r['doc_id']}-{r['chunk_index']}",
                      "persona_id": pid, "text": r["prefix_text"].strip(), "kind": "청크 맥락 접두",
                      "fact_ids": [f["fact_id"] for f in facts[pid]], "key": k})
    if not items:
        return {"requests": 0}
    res = run_requests(items, model, batch=batch, run_id=f"E1b-prefix-{_short(model)}", rubric="v2")
    ts, rows = _now(), []
    for it in items:
        got = res["labels"].get(it["cid"]) or {}
        pfx, var, doc, ci = it["key"]
        rows += [{"prefix_id": pfx, "variant_id": var, "doc_id": doc, "chunk_index": ci, "fact_id": f, "label": got[f],
                  "labeler": model, "rubric_version": "v2", "ts": ts} for f in it["fact_ids"] if got.get(f) in LABELS]
    append_jsonl(path, rows)  # 기존 행은 그대로 두고 덧붙인다
    return {"requests": len(items), "pairs": len(rows), "task_spent_usd": res["spent_task_usd"]}


# ---- κ 표본 ----------------------------------------------------------------------
def kappa_sample(personas: set[str], n: int, seed: int = 4041) -> list[dict[str, Any]]:
    """주 판정자(Haiku) 라벨이 있는 (청크, 사실) 쌍에서 full/partial/none 을 고르게 층화해 뽑는다.
    Sonnet 은 그 쌍이 든 청크를 통째로 판정한다 (같은 입력 형식) — 비교는 뽑힌 쌍만."""
    rng = random.Random(seed)
    rows = [r for r in read_jsonl(_labels_dir() / "containment.jsonl")
            if r["labeler"] == PRIMARY and r.get("persona_id") in personas]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    pick = []
    for lab in LABELS:
        pick += rng.sample(by[lab], min(len(by[lab]), n // 3))
    return pick


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_rerun(rubric: str, *, batch: bool) -> dict[str, Any]:
    """kappa-sample.jsonl 의 쌍이 든 청크를 Haiku·Sonnet 모두 같은 루브릭으로 다시 판정해 κ 를 낸다 (파일럿 루브릭 조정)."""
    pick = read_jsonl(_labels_dir() / "kappa-sample.jsonl")
    chunks = {c["chunk_key"]: c for c in read_jsonl(_labels_dir() / "chunks.jsonl")}
    facts = load_facts()
    keys = sorted({(r["persona_id"], r["chunk_key"]) for r in pick})
    out, labs = {}, {}
    for short in ("haiku", "sonnet"):
        m = MODELS[short]
        name = f"containment-{short}-{rubric}-kappa.jsonl"
        items = [{"cid": f"{_short(m)}{rubric}-{pid}-{ck[:44]}", "persona_id": pid, "chunk_key": ck,
                  "text": chunks[ck]["body"], "kind": "텍스트 조각", "fact_ids": [f["fact_id"] for f in facts[pid]]}
                 for pid, ck in keys]
        done = {r["chunk_key"] for r in read_jsonl(_labels_dir() / name)}
        items = [it for it in items if it["chunk_key"] not in done]
        out[short] = judge_chunks([], None, m, batch=batch, out_name=name, items=items, rubric=rubric) if items else {}
        labs[short] = {(r["chunk_key"], r["fact_id"]): r["label"] for r in read_jsonl(_labels_dir() / name)}
    pairs = [(labs["haiku"][k], labs["sonnet"][k]) for k in ((r["chunk_key"], r["fact_id"]) for r in pick)
             if k in labs["haiku"] and k in labs["sonnet"]]
    bin_ = [("full" if x == "full" else "no", "full" if y == "full" else "no") for x, y in pairs]
    return out | {"rubric": rubric, "pairs": len(pairs),
                  "kappa_3class": round(cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]), 3),
                  "kappa_full_vs_rest": round(cohen_kappa([x for x, _ in bin_], [y for _, y in bin_]), 3),
                  "confusion": dict(Counter(f"{x}->{y}" for x, y in pairs)),
                  "haiku_labels": dict(Counter(x for x, _ in pairs)), "sonnet_labels": dict(Counter(y for _, y in pairs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chunks", "prefixes", "auto", "sample-kappa", "kappa", "recover"])
    ap.add_argument("--batches", default="", help="recover: 결과를 다시 받을 batch id 목록")
    ap.add_argument("--rubric", default="v2", choices=list(RUBRICS))
    ap.add_argument("--variants", default="")
    ap.add_argument("--personas", default="")
    ap.add_argument("--model", default="haiku", choices=list(MODELS))
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--task", default="T04", choices=["T04", "T04b"], help="원장 task (T04b 확장은 상한 $10)")
    a = ap.parse_args()
    global TASK_ID, BUDGET
    TASK_ID, BUDGET = a.task, {"T04": 20.0, "T04b": 10.0}[a.task]
    variants = [v for v in a.variants.split(",") if v]
    personas = set(a.personas.split(",")) if a.personas else None
    model = MODELS[a.model]
    if a.cmd == "chunks":
        print(json.dumps(judge_chunks(variants, personas, model, batch=a.batch, rubric=a.rubric), ensure_ascii=False))
    elif a.cmd == "recover":
        print(json.dumps(recover_chunks(variants, model, a.batches.split(","), a.rubric), ensure_ascii=False))
    elif a.cmd == "kappa":
        print(json.dumps(kappa_rerun(a.rubric, batch=a.batch), ensure_ascii=False))
    elif a.cmd == "prefixes":
        print(json.dumps(judge_prefixes(personas, model, batch=a.batch), ensure_ascii=False))
    elif a.cmd == "auto":
        print(json.dumps(auto_pass(variants), ensure_ascii=False))
    else:
        pick = kappa_sample(personas or set(), a.n)
        chunks = {c["chunk_key"]: c for c in read_jsonl(_labels_dir() / "chunks.jsonl")}
        need = defaultdict(set)
        for r in pick:
            need[(r["persona_id"], r["chunk_key"])].add(r["fact_id"])
        items = [{"cid": f"s-{pid}-{ck[:48]}", "persona_id": pid, "chunk_key": ck, "text": chunks[ck]["body"],
                  "kind": "텍스트 조각", "fact_ids": [f["fact_id"] for f in load_facts()[pid]]} for (pid, ck) in need]
        res = judge_chunks([], None, MODELS["sonnet"], batch=a.batch, out_name="containment-sonnet-pilot.jsonl",
                           items=items)
        write_jsonl(_labels_dir() / "kappa-sample.jsonl", pick)
        son = {(r["chunk_key"], r["fact_id"]): r["label"]
               for r in read_jsonl(_labels_dir() / "containment-sonnet-pilot.jsonl")}
        pairs = [(r["label"], son[(r["chunk_key"], r["fact_id"])]) for r in pick if (r["chunk_key"], r["fact_id"]) in son]
        k = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None
        print(json.dumps(res | {"pairs_compared": len(pairs), "kappa_haiku_sonnet": k,
                                "confusion": Counter(f"{x}->{y}" for x, y in pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
