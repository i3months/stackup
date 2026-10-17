"""코퍼스 마무리 (T03): 요약 보존 판정(Haiku) → facts.jsonl · manifest.json · personas.csv · 청크 수 · 검토 샘플.

    python -m corpus.finalize --personas P04,P26,P35 --preserve --review     # 파일럿
    python -m corpus.finalize --all --preserve [--batch]

- 보존 판정: 문서 1개당 Haiku 5.5 호출 1회 (분석 summary + markdown, 그 문서의 사실 목록 → 사실별 full/partial/none).
  원장 래퍼(task T03, 상한 $5)를 거친다. 구조화 출력.
- preserved_in_summary = 판정 라벨 (full = 핵심 값까지 남음).
- 청크 수: B0 변형(운영 청커 1000/200)으로 분석 마크다운을 청킹한 페르소나별 합계 (SPEC §4.1 목표 10~60).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import now_iso, results_dir, stackup_commit, write_json, write_jsonl  # noqa: E402
from corpus.generate import GEN_MODEL, load_slots  # noqa: E402

TASK_ID = "T03"
BUDGET = 5.0
JUDGE = "claude-haiku-5-5"
REVIEW_NAME = "review-sample.md"
PRES_SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact_id": {"type": "string"}, "label": {"type": "string", "enum": ["full", "partial", "none"]}},
        "required": ["fact_id", "label"], "additionalProperties": False}}},
    "required": ["judgments"], "additionalProperties": False,
}
PRES_SYSTEM = (
    "당신은 문서 요약이 원문의 사실을 보존했는지 판정합니다. 각 사실에 대해 요약(summary 와 markdown 전체)에 "
    "그 사실이 남아 있는지 고르세요.\n"
    "- full: 사실의 핵심 정보(수치·이름·도구·설정값 등)가 요약에 남아 있다. 바꿔 말한 것도 포함.\n"
    "- partial: 같은 사실을 언급하지만 핵심 값이 빠졌다.\n"
    "- none: 요약에 없다.\n"
    "모든 fact_id 에 대해 정확히 한 번씩 답하세요."
)


def _corpus() -> Path:
    return results_dir("E1-corpus")


def load_persona(pid: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    w = _corpus() / "work" / pid
    meta = json.loads((w / "persona_meta.json").read_text(encoding="utf-8"))
    facts = []
    for p in sorted(w.glob("facts-*.json")):
        facts += json.loads(p.read_text(encoding="utf-8"))
    return meta, facts


def preserve_requests(facts: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    from llm_client import cached_text  # noqa: PLC0415

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        if f["preserved_in_summary"] is None:
            by_doc[f["doc_id"]].append(f)
    reqs = []
    for doc_id, fs in sorted(by_doc.items()):
        a = json.loads((_corpus() / "analysis" / f"{doc_id}.json").read_text(encoding="utf-8"))
        md = (_corpus() / "analysis" / f"{doc_id}.md").read_text(encoding="utf-8")
        fl = "\n".join(f"- {f['fact_id']}: {f['statement']} (근거: \"{f['evidence_text']}\")" for f in fs)
        reqs.append((doc_id, {
            "model": JUDGE, "max_tokens": 2048, "output_config": {"effort": "low"},
            "system": [cached_text(PRES_SYSTEM)],
            "messages": [{"role": "user", "content": f"[요약 summary]\n{a['summary']}\n\n[요약 markdown]\n{md}\n\n[사실 목록]\n{fl}"}],
        }))
    return reqs


def apply_judgments(facts: list[dict[str, Any]], doc_id: str, parsed: dict[str, Any]) -> int:
    want = {f["fact_id"] for f in facts if f["doc_id"] == doc_id}
    got = {j["fact_id"]: j["label"] for j in parsed["judgments"] if j["fact_id"] in want}
    for f in facts:
        if f["fact_id"] in got:
            f["preserved_in_summary"] = got[f["fact_id"]]
            f["preserved_labeler"] = JUDGE
    return len(want - set(got))


def run_preserve(pids: list[str], *, batch: bool) -> dict[str, Any]:
    from llm_client import LabClient, json_of  # noqa: PLC0415

    lab = LabClient(task_id=TASK_ID, budget_usd=BUDGET, run_id="E1a-preserve")  # T04b 는 main 에서 TASK_ID·BUDGET 을 바꾼다
    all_facts: dict[str, list[dict[str, Any]]] = {}
    reqs = []
    for pid in pids:
        _, facts = load_persona(pid)
        all_facts[pid] = facts
        reqs += [(doc_id, r, pid) for doc_id, r in preserve_requests(facts)]
    missing = 0
    if batch and reqs:
        bid = lab.submit_batch([(d, r) for d, r, _ in reqs], estimate_usd=0.0005 * len(reqs), json_schema=PRES_SCHEMA)
        from common import append_jsonl  # noqa: PLC0415

        append_jsonl(_corpus() / "batches.jsonl", [{"batch_id": bid, "task": TASK_ID, "run_id": "E1a-preserve", "n": len(reqs),
                                                    "ts": now_iso()}])  # 기다리기 전에 기록 (끊겨도 회수 가능)
        print(f"batch {bid} ({len(reqs)}건) 대기")
        lab.wait_batch(bid, poll_sec=30)
        res = lab.collect_batch(bid)
        for d, _, pid in reqs:
            if res[d]["type"] == "succeeded":
                missing += apply_judgments(all_facts[pid], d, json_of(res[d]["message"]))
    else:
        for d, r, pid in reqs:
            msg = lab.call(estimate_usd=0.002, json_schema=PRES_SCHEMA, **r)
            missing += apply_judgments(all_facts[pid], d, json_of(msg))
    for pid, facts in all_facts.items():  # 사실 파일에 되써서 재실행 시 건너뛴다
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in facts:
            by_type[f["doc_id"].split("-", 1)[1]].append(f)
        for t, fs in by_type.items():
            name = {"resume": "resume", "cover": "cover_letter", "repo": "repo", "career": "career_description"}[t]
            (_corpus() / "work" / pid / f"facts-{name}.json").write_text(json.dumps(fs, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    return {"requests": len(reqs), "missing_judgments": missing, "task_spent_usd": round(lab.spent_usd(), 6)}


def chunk_counts(pids: list[str]) -> dict[str, int]:
    from ai_server.analyzer._embedding_step import _contextualize  # noqa: F401,PLC0415  (운영 경로 확인용 import)
    from ai_server.rag.chunker import MarkdownChunker  # noqa: PLC0415

    ch = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
    out = {}
    for pid in pids:
        meta, _ = load_persona(pid)
        n = 0
        for d in meta["docs"]:
            n += len(ch.split((_corpus() / d["analysis_path"]).read_text(encoding="utf-8")))
        out[pid] = n
    return out


def _by_cache_prompt(pids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for pid in pids:
        m = json.loads((_corpus() / "work" / pid / "plan.json").read_text(encoding="utf-8")).get("_meta", {})
        v = m.get("cache_prompt", None)
        out["omitted (server default true)" if v is None else str(v).lower()].append(pid)
    return dict(out)


def compile_outputs(pids: list[str]) -> dict[str, Any]:
    corpus = _corpus()
    facts, docs = [], []
    pres = Counter()
    pres_by = defaultdict(Counter)
    for pid in pids:
        meta, fs = load_persona(pid)
        facts += fs
        for d in meta["docs"]:
            docs.append({k: d.get(k) for k in ("doc_id", "doc_type", "source", "generator", "seed", "repo_url",
                                               "repo_commit", "license", "path", "analysis_path", "chars")}
                        | {"persona_id": pid, "analysis_generator": GEN_MODEL})
        for f in fs:
            pres[f["preserved_in_summary"]] += 1
            pres_by[f"doc:{f['doc_id'].split('-', 1)[1]}"][f["preserved_in_summary"]] += 1
            pres_by[f"type:{f['fact_type']}"][f["preserved_in_summary"]] += 1
    gen_commit = {pid: dict(zip(("commit", "source"), generator_commit(pid))) for pid in pids}
    for d in docs:
        d["generator_commit"] = gen_commit[d["persona_id"]]["commit"]
    write_jsonl(corpus / "facts.jsonl", facts)
    slots = load_slots()
    for s in slots:
        if s["persona_id"] in pids:
            s["generator"] = GEN_MODEL
    with (corpus / "personas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(slots[0].keys()))
        w.writeheader()
        w.writerows(slots)
    cc = chunk_counts(pids)
    stats = {
        "personas": len(pids), "docs": len(docs), "facts": len(facts),
        "evidence_exact_match": sum(1 for f in facts if _evidence_ok(f)) / max(len(facts), 1),
        "evidence_unique_rate": sum(f["evidence_unique"] for f in facts) / max(len(facts), 1),
        "facts_per_doc": dict(Counter(f["doc_id"] for f in facts)),
        "fact_types": dict(Counter(f["fact_type"] for f in facts)),
        "preserved": dict(pres),
        "preserved_rate_by": {k: {"full_rate": round(v["full"] / sum(v.values()), 3), **v} for k, v in sorted(pres_by.items())},
        "personas_by_generator_commit": {c: sorted(p for p, v in gen_commit.items() if v["commit"] == c)
                                         for c in sorted({v["commit"] for v in gen_commit.values()})},
        # T04b (P-025): P01~P40 은 요청에 cache_prompt 를 보내지 않았다 (llama-server 기본값 true), P41~ 는 false
        "personas_by_cache_prompt": _by_cache_prompt(pids),
        "chunks_per_persona": cc,
        "chunks_stats": {"min": min(cc.values()), "median": statistics.median(cc.values()), "max": max(cc.values())},
    }
    write_json(corpus / "manifest.json", {
        "created_at": now_iso(), "script": "ai/scripts/rag_eval/corpus/{generate,finalize}.py",
        "stackup_commit": stackup_commit(), "generator": GEN_MODEL, "seed_rule": "BASE_SEED 7000 + 번호×1000 + 단계 오프셋",
        "preserve_judge": JUDGE, "stats": stats, "generator_commit": gen_commit, "docs": docs,
    })
    return stats


# 생성 도중 생성기 코드가 바뀌었다 (관리자 T03 답변). 우선순위:
# 1. generation-runs.json (corpus/generation_runs.py): 실행 로그 구간별 프로세스 시작 시점 커밋 중 현재 산출물 파일을
#    만든 실행들. plan._meta.generator_commit 은 계획 저장 시점의 HEAD 라 실행 중 커밋이 섞인다 (예: P37 → 347c1ee 로
#    찍혔지만 실제 코드는 ce8a51b).
# 2. plan._meta.generator_commit (recorded) 3. plan._meta 의 timeline 유무로 추정 (inferred).
def generator_commit(pid: str) -> tuple[str, str]:
    runs = _corpus() / "generation-runs.json"
    if runs.exists():
        art = json.loads(runs.read_text(encoding="utf-8")).get("personas_artifacts", {})
        if art.get(pid):
            return "+".join(art[pid]), "run_log"
    m = json.loads((_corpus() / "work" / pid / "plan.json").read_text(encoding="utf-8")).get("_meta", {})
    if m.get("generator_commit"):
        return m["generator_commit"], "recorded"
    return ("0205187" if "timeline" in m else "aef8507"), "inferred"


def _evidence_ok(f: dict[str, Any]) -> bool:
    p = _corpus() / "docs" / f["persona_id"] / f"{f['doc_id']}.md"
    # newline="" : 레포 원문의 CRLF 를 그대로 읽는다 (P29-repo requirements.txt). read_text 는 \r\n 을 \n 으로 바꿔
    # 오프셋이 밀린다. 문서 파일은 운영 추출기 출력 그대로이고 오프셋은 그 원문 기준이다.
    with p.open(encoding="utf-8", newline="") as fh:
        t = fh.read()
    return t[f["char_start"]:f["char_end"]] == f["evidence_text"]


def review_sample(pids: list[str]) -> Path:
    corpus = _corpus()
    out = ["# E1a 검토 샘플 (T03)", "",
           "> 사람 현실성 점검용. 문서는 M-M(gemma-4-26B-A4B) 생성, 레포는 실제 공개 레포(운영 입력 구성).", ""]
    for pid in pids:
        w = corpus / "work" / pid
        plan = json.loads((w / "plan.json").read_text(encoding="utf-8"))
        meta, facts = load_persona(pid)
        out += [f"## {pid} — {plan['name']} ({plan['headline']}) · 도메인 {plan['_meta']['domain']}", "",
                "### 혼동 경험 쌍", ""]
        exps = {e["experience_id"]: e for e in plan["experiences"]}
        for cp in plan["confusable_pairs"]:
            out.append(f"- {cp['a']} {exps[cp['a']]['title']} ↔ {cp['b']} {exps[cp['b']]['title']}: "
                       f"공통 {cp['shared']} / 차이 {cp['difference']}")
        for d in meta["docs"]:
            text = (corpus / d["path"]).read_text(encoding="utf-8")
            if d["doc_type"] == "repo":
                text = text[:3000] + "\n... (이하 생략)"
            out += ["", f"### 문서 {d['doc_id']} ({d['doc_type']}, {d['chars']}자)", "", "````markdown", text.rstrip(),
                    "````", "", f"#### 사실 ({sum(f['doc_id'] == d['doc_id'] for f in facts)}개)", "",
                    "| fact_id | 경험 | 유형 | 사실 | 근거 구간 | 유일 | 요약 보존 |", "|---|---|---|---|---|---|---|"]
            for f in facts:
                if f["doc_id"] == d["doc_id"]:
                    out.append(f"| {f['fact_id'][-3:]} | {f['experience_id']} | {f['fact_type']} | {f['statement']} | "
                               f"`{f['evidence_text']}` | {'Y' if f['evidence_unique'] else 'N'} | {f['preserved_in_summary']} |")
            ana = (corpus / d["analysis_path"]).read_text(encoding="utf-8")
            out += ["", "#### 분석 요약 (운영 프롬프트 + M-M)", "", "````markdown", ana.rstrip(), "````"]
        out.append("")
    p = corpus / REVIEW_NAME
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preserve", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--review-personas", default="", help="검토 샘플 대상 (기본: 처리 대상 전부)")
    ap.add_argument("--task", default="T03", help="원장 task (T04b 확장은 T04b, 상한 $10)")
    ap.add_argument("--review-name", default="review-sample.md")
    a = ap.parse_args()
    global TASK_ID, BUDGET, REVIEW_NAME
    TASK_ID, BUDGET = a.task, {"T03": 5.0, "T04b": 10.0}[a.task]
    REVIEW_NAME = a.review_name
    done = {p.parent.name for p in (_corpus() / "work").glob("P*/persona_meta.json")}  # _보관 폴더 제외
    pids = sorted(done, key=lambda p: int(p[1:])) if a.all else [p for p in a.personas.split(",") if p in done]
    if a.preserve:
        print(json.dumps(run_preserve(pids, batch=a.batch), ensure_ascii=False))
    # facts.jsonl·manifest·personas.csv 는 통째로 다시 쓰므로 항상 완료된 페르소나 전부로 만든다
    # (T04b: --personas 로 새 페르소나만 판정해도 기존 사실이 지워지지 않게)
    print(json.dumps(compile_outputs(sorted(done, key=lambda p: int(p[1:]))), ensure_ascii=False, indent=1))
    if a.review:
        print(review_sample(a.review_personas.split(",") if a.review_personas else pids))


if __name__ == "__main__":
    main()
