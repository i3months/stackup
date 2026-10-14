"""E1b 질의 생성 (T04, SPEC §4.2·§9.2).

    python -m queries.generate --personas P03,P26,P37      # dev 파일럿
    python -m queries.generate --all

- 페르소나당 15개: followup 9 · coaching 4 · pool 2. 정답 없음은 followup·coaching 의 약 10% (전체 52).
- 질의는 사실에서 출발한다: 근거 사실 1~3개를 먼저 고르고(문서 유형·사실 유형·요약 보존 층화), 그 사실을 묻는
  직전 질문·답변(또는 자기소개)을 M-M 이 만든다. 생각 모드 끔, cache_prompt=false (P-018).
- query_text 는 운영 질의 구성 그대로다.
  followup = 직전 질문 + "\\n\\n" + 답변 (followup_consumer), coaching = 질문 + "\\n" + 답변 (feedback_consumer),
  pool = 운영 questions_consumer._build_initial_rag_query(GenerateQuestionsRequest) 를 직접 호출.
- 어휘 겹침: 사실을 담은 텍스트(답변, dontknow 는 직전 질문, pool 은 자기소개)와 근거 구간 합집합의 Kiwi 형태소
  자카드. 지정 대역과 실측이 어긋나면 seed 만 +50 바꿔 다시 만든다 (P-017, 최대 3개 seed).
- 결과: stackup-lab/results/E1-queries/queries.jsonl, 작업 파일 work/<pid>.jsonl (페르소나 단위 재실행).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import now_iso, results_dir, stackup_commit, write_json  # noqa: E402
from local_llm import LocalLLM, LocalLLMError  # noqa: E402

GEN_MODEL = "M-M"
BASE_SEED = 90000  # 페르소나 번호×1000 + 질의 번호×10, seed 라운드 +50 (코퍼스 7000 대와 겹치지 않게)
TEMP = 0.8
N_PER_PERSONA = {"followup": 9, "coaching": 4, "pool": 2}
NO_ANSWER_TOTAL = 52  # followup·coaching 520개의 10%
# 겹침 대역 경계 (Kiwi 형태소 자카드). dev 파일럿 분포로 정한다 (T04 지시: 조정은 dev 에서만).
HIGH_MIN = 0.20
LOW_MAX = 0.10
MAX_ROUNDS = 3
ROLE_KO = {
    "BACKEND": "백엔드 개발자", "FRONTEND": "프론트엔드 개발자", "DEVOPS": "인프라/DevOps 엔지니어",
    "DATA_AI": "데이터·AI 엔지니어", "MOBILE": "모바일 개발자", "PLANNING": "서비스 기획자",
    "MARKETING": "마케터", "DESIGN": "디자이너",
}
CORE_SOURCE_TYPE = {"PDF": "RESUME", "REPOSITORY": "REPOSITORY", "COVER_LETTER": "COVER_LETTER"}  # Core resolveSourceType


def _corpus() -> Path:
    return results_dir("E1-corpus")


def _out() -> Path:
    return results_dir("E1-queries")


# ---- 페르소나 자료 ---------------------------------------------------------------
def load_persona(pid: str) -> dict[str, Any]:
    c = _corpus()
    slots = {r["persona_id"]: r for r in csv.DictReader((c / "personas.csv").open(encoding="utf-8"))}
    plan = json.loads((c / "work" / pid / "plan.json").read_text(encoding="utf-8"))
    facts = [json.loads(x) for x in (c / "facts.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip() and json.loads(x)["persona_id"] == pid]
    meta = json.loads((c / "work" / pid / "persona_meta.json").read_text(encoding="utf-8"))
    docs = []
    for d in sorted(meta["docs"], key=lambda x: x["doc_id"]):
        a = json.loads((c / "analysis" / f"{d['doc_id']}.json").read_text(encoding="utf-8"))
        md = (c / "analysis" / f"{d['doc_id']}.md").read_text(encoding="utf-8")
        docs.append({"doc_id": d["doc_id"], "source_type": a["source_type"], "summary": a["summary"],
                     "tech_stack": a["tech_stack"], "markdown": md})
    return {"pid": pid, "slot": slots[pid], "plan": plan, "facts": facts, "docs": docs}


# ---- 일정: 유형·정답 없음·답변 품질·겹침 대역·근거 사실 -----------------------------------
def no_answer_personas(all_pids: list[str]) -> set[str]:
    """정답 없음 2개를 받을 페르소나 (나머지는 1개). 40명 × 1 + 12 = 52."""
    rng = random.Random(BASE_SEED + 1)
    return set(rng.sample(sorted(all_pids), NO_ANSWER_TOTAL - len(all_pids)))


def schedule(p: dict[str, Any], n_no_answer: int, idx: int) -> list[dict[str, Any]]:
    rng = random.Random(BASE_SEED + idx * 1000)
    slots: list[dict[str, Any]] = []
    # 정답 없음: 2개면 followup 1 + coaching 1, 1개면 followup (9:4 비율에 가깝게)
    na = {"followup": 1, "coaching": 1 if n_no_answer == 2 else 0}
    quality_cycle = {"followup": ["strong", "weak", "dontknow"], "coaching": ["strong", "weak"], "pool": ["strong"]}
    for qtype, n in N_PER_PERSONA.items():
        n_na = na.get(qtype, 0)
        qs = quality_cycle[qtype]
        start = rng.randrange(len(qs))
        for i in range(n):
            no_answer = i >= n - n_na
            q = qs[(start + i) % len(qs)]
            if no_answer and q == "dontknow":
                q = "weak"
            slots.append({"qtype": qtype, "no_answer": no_answer, "answer_quality": q})
    # 겹침 대역: 정답 있는 질의를 유형마다 high/low 반씩. dontknow 는 직전 질문이 사실을 직접 담으므로
    # 대역을 지정하지 않고 실측값만 남긴다.
    by_type = defaultdict(list)
    for s in slots:
        if not s["no_answer"] and s["answer_quality"] != "dontknow":
            by_type[s["qtype"]].append(s)
    flip = rng.random() < 0.5
    for qtype, ss in by_type.items():
        for i, s in enumerate(ss):
            s["overlap_band"] = ("high" if (i % 2 == 0) != flip else "low")
        flip = not flip if len(ss) % 2 else flip
    for s in slots:
        s.setdefault("overlap_band", None)
    pools = [s for s in slots if s["qtype"] == "pool"]
    for s, mode in zip(pools, ("INTEGRATED", "JOB_TAILORED")):  # 자기소개만 / 자기소개 + JD
        s["mode"] = mode
    pick_gold(p, [s for s in slots if not s["no_answer"]], rng)
    for i, s in enumerate(slots):
        s["query_id"] = f"{p['pid']}-Q{i + 1:02d}"
        s["seed"] = BASE_SEED + idx * 1000 + (i + 1) * 10
    return slots


def pick_gold(p: dict[str, Any], slots: list[dict[str, Any]], rng: random.Random) -> None:
    """근거 사실 1~3개. 주 사실은 (문서 유형, 사실 유형, 요약 보존) 조합이 고르게 돌도록 고르고,
    나머지는 같은 문서·같은 경험의 사실에서 고른다. 페르소나 안에서 주 사실은 겹치지 않게 한다."""
    facts = [f for f in p["facts"]]
    used: Counter = Counter()
    cells: Counter = Counter()
    for s in slots:
        def score(f: dict[str, Any]) -> tuple:
            cell = (f["doc_id"].split("-", 1)[1], f["fact_type"], f["preserved_in_summary"])
            return (used[f["fact_id"]], cells[(cell[0],)] , cells[(cell[1],)], cells[(cell[2],)], rng.random())
        main = min(facts, key=score)
        used[main["fact_id"]] += 1
        for key in ((main["doc_id"].split("-", 1)[1],), (main["fact_type"],), (main["preserved_in_summary"],)):
            cells[key] += 1
        k = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        if s["qtype"] == "pool":
            k = max(k, 2)  # 자기소개는 여러 사실을 말한다
        mates = [f for f in facts if f["fact_id"] != main["fact_id"] and f["doc_id"] == main["doc_id"]
                 and main["experience_id"] and f["experience_id"] == main["experience_id"]]
        rng.shuffle(mates)
        s["gold"] = [main] + mates[: k - 1]


# ---- 프롬프트 -------------------------------------------------------------------
SYSTEM = ("당신은 한국어 모의면접 평가 데이터를 만드는 사람입니다. 지시한 사실만 쓰고, 지원자 문서에 없는 사실은 "
          "지어내지 않습니다 (정답 없음 과제는 예외). JSON 만 출력합니다.")
QA_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
             "required": ["question", "answer"], "additionalProperties": False}
POOL_SCHEMA = {"type": "object", "properties": {
    "self_intro": {"type": "string"}, "company": {"type": "string"}, "job_description": {"type": "string"}},
    "required": ["self_intro", "company", "job_description"], "additionalProperties": False}

QUALITY_GUIDE = {
    "strong": "답변은 구체적이고 자신감 있게, 3~5문장.",
    "weak": "답변은 짧고 다소 모호하게, 1~2문장. 그래도 아래 사실의 일부는 분명히 언급한다.",
    "dontknow": ("직전 질문이 아래 사실을 **직접** 묻는다 (사실의 핵심 정보를 질문에 담는다). "
                 "답변은 기억이 안 나거나 잘 모른다는 짧은 말 1~2문장 (사실을 말하지 않는다)."),
}
BAND_GUIDE = {
    "high": "사실을 말할 때 근거 문장의 단어(도구 이름·수치·고유 명사·핵심 동사)를 **그대로** 쓴다.",
    "low": ("사실을 말할 때 근거 문장의 표현을 그대로 쓰지 않는다. 같은 뜻을 다른 단어로 풀어 말한다 "
            "(예: '응답 시간 73% 단축' → '기다리는 시간을 4분의 1 수준으로 줄였다'). 도구 이름은 꼭 필요할 때만 한 번."),
}


def _persona_brief(p: dict[str, Any]) -> str:
    plan = p["plan"]
    return f"지원자: {plan['name']} ({plan['headline']}), 직군 {ROLE_KO.get(p['slot']['job_role'], p['slot']['job_role'])}"


def _exp_brief(p: dict[str, Any], exp_id: str | None) -> str:
    for e in p["plan"]["experiences"]:
        if e["experience_id"] == exp_id:
            return f"경험: {e['title']} @ {e['org']} ({e['period']}) — {e['context']}"
    return "경험: (문서 전체에 걸친 사실)"


def _facts_block(gold: list[dict[str, Any]]) -> str:
    return "\n".join(f"- 사실: {f['statement']}\n  근거 문장: \"{f['evidence_text']}\"" for f in gold)


def qa_prompt(p: dict[str, Any], s: dict[str, Any]) -> list[dict[str, str]]:
    if s["qtype"] == "followup":
        role = ("면접 도중의 한 순간을 만든다. question = 면접관의 직전 질문(1~2문장), answer = 그 질문에 대한 지원자의 "
                "답변. 면접관은 이 답변을 듣고 꼬리질문을 할 것이다.")
    else:
        role = ("끝난 면접의 문답 한 쌍을 만든다. question = 면접관 질문(1~2문장), answer = 지원자 답변. 이 문답은 "
                "나중에 답변 코칭을 받는다.")
    if s["no_answer"]:
        exps = "\n".join(f"- {e['title']} @ {e['org']}: {e['context']}" for e in p["plan"]["experiences"])
        user = (f"{_persona_brief(p)}\n\n{role}\n\n**정답 없음 과제**: 지원자가 아래 문서 속 경험과 **다른**, 문서에 없는 "
                f"경험(다른 회사·다른 프로젝트·다른 도구·다른 수치)을 구체적으로 말한다. 아래 경험이나 그 도구·수치를 "
                f"언급하지 않는다.\n문서 속 경험 (언급 금지):\n{exps}\n\n{QUALITY_GUIDE[s['answer_quality']]}\n"
                "question 은 직군에 맞는 일반적인 질문이면 된다. JSON 만 출력.")
    else:
        user = (f"{_persona_brief(p)}\n{_exp_brief(p, s['gold'][0]['experience_id'])}\n\n{role}\n\n"
                f"이 문답이 다룰 사실:\n{_facts_block(s['gold'])}\n\n{QUALITY_GUIDE[s['answer_quality']]}\n"
                + ("" if s["answer_quality"] == "dontknow" else BAND_GUIDE[s["overlap_band"]] + "\n")
                + "question 은 사실의 수치·도구를 미리 말하지 않는다"
                + (" (dontknow 는 예외)" if s["answer_quality"] == "dontknow" else "") + ". JSON 만 출력.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def pool_prompt(p: dict[str, Any], s: dict[str, Any], mode: str) -> list[dict[str, str]]:
    jd = ("company 와 job_description 도 만든다: 이 지원자가 지원할 만한 가상의 회사 이름과, 그 회사의 채용공고 "
          "(주요 업무·자격 요건·우대 사항, 400~700자). 채용공고는 지원자 개인 경험을 말하지 않는다."
          if mode == "JOB_TAILORED" else "company 와 job_description 은 빈 문자열로 둔다.")
    user = (f"{_persona_brief(p)}\n\n면접 시작 직후 지원자의 자기소개 답변(self_intro, 300~500자)을 만든다. "
            f"자기소개는 아래 사실을 자연스럽게 포함하고, 그 밖의 경력은 한두 마디로만 말한다.\n"
            f"자기소개가 다룰 사실:\n{_facts_block(s['gold'])}\n\n{BAND_GUIDE[s['overlap_band']]}\n{jd}\nJSON 만 출력.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---- 운영 질의 구성 -------------------------------------------------------------
def pool_query_text(p: dict[str, Any], mode: str, self_intro: str, company: str, jd: str) -> str:
    """운영 questions_consumer._build_initial_rag_query 를 그대로 호출한다. 문서 컨텍스트는 Core
    SessionQuestionsRequester.buildDocumentContexts 와 같은 필드(요약·기술 스택·분석 마크다운)."""
    from ai_server.messaging.consumers.questions_consumer import _build_initial_rag_query  # noqa: PLC0415
    from ai_server.model.messages.questions import DocumentContext, GenerateQuestionsRequest  # noqa: PLC0415

    docs = [DocumentContext(document_id=i + 1, source_type=CORE_SOURCE_TYPE[d["source_type"]], summary=d["summary"],
                            tech_stack=d["tech_stack"], markdown=d["markdown"]) for i, d in enumerate(p["docs"])]
    req = GenerateQuestionsRequest(session_id=1, mode=mode, job_categories=[p["slot"]["job_role"]], documents=docs,
                                   self_intro_answer=self_intro,
                                   target_company_name=company or None, target_job_description=jd or None)
    return _build_initial_rag_query(req)


# ---- 겹침·검증 ------------------------------------------------------------------
def tokens(text: str) -> set[str]:
    from retrieve import kiwi_tokens  # noqa: PLC0415

    return set(kiwi_tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def band_of(j: float) -> str | None:
    return "high" if j >= HIGH_MIN else ("low" if j <= LOW_MAX else None)


def no_answer_check(text: str, p: dict[str, Any]) -> dict[str, Any]:
    """정답 없음 답변이 그 페르소나 사실과 겹치지 않는지: 사실별 자카드 최댓값과, 페르소나 사실에 나오는
    숫자·영문 토큰을 답변이 쓰는지."""
    from retrieve import kiwi_tokens  # noqa: PLC0415

    t = tokens(text)
    best = max(((jaccard(t, tokens(f["evidence_text"] + " " + f["statement"])), f["fact_id"]) for f in p["facts"]),
               default=(0.0, None))
    fact_toks = set()
    for f in p["facts"]:
        fact_toks |= set(kiwi_tokens(f["evidence_text"]))
    distinctive = {x for x in t if any(ch.isdigit() for ch in x) or x.isascii()}
    shared = sorted(distinctive & fact_toks)
    return {"max_fact_jaccard": round(best[0], 3), "max_fact_id": best[1], "shared_distinctive": shared,
            "ok": best[0] <= 0.15 and len(shared) <= 2}


# ---- 생성 ----------------------------------------------------------------------
class QueryGen:
    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm

    def one(self, p: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        rec: dict[str, Any] | None = None
        for rnd in range(MAX_ROUNDS):
            seed = s["seed"] + 50 * rnd
            try:
                rec = self._gen(p, s, seed)
            except (LocalLLMError, json.JSONDecodeError, KeyError) as e:
                history.append({"seed": seed, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            problems = []
            if s["no_answer"]:
                chk = no_answer_check(rec["fact_text"], p)
                rec["no_answer_check"] = chk
                if not chk["ok"]:
                    problems.append(f"정답 없음 답변이 페르소나 사실과 겹침 {chk}")
            elif s["overlap_band"] and band_of(rec["overlap_jaccard"]) != s["overlap_band"]:
                problems.append(f"겹침 {rec['overlap_jaccard']:.3f} 이 {s['overlap_band']} 대역 밖")
            history.append({"seed": seed, "problems": problems})
            if not problems:
                break
        else:
            if rec is None:
                raise RuntimeError(f"{s[query_id]} 생성 {MAX_ROUNDS}회 모두 오류: {history}")
            rec["band_ok" if not s["no_answer"] else "no_answer_ok"] = False
        rec.update(seed=seed, seed_round=rnd, attempts=history)
        return rec

    def _gen(self, p: dict[str, Any], s: dict[str, Any], seed: int) -> dict[str, Any]:
        gold = s.get("gold", []) if not s["no_answer"] else []
        ev = set().union(*[tokens(f["evidence_text"]) for f in gold]) if gold else set()
        base = {"query_id": s["query_id"], "persona_id": p["pid"], "split": p["slot"]["split"], "qtype": s["qtype"],
                "gold_fact_ids": [f["fact_id"] for f in gold], "no_answer": s["no_answer"],
                "answer_quality": s["answer_quality"], "overlap_band": s["overlap_band"], "generator": GEN_MODEL}
        if s["qtype"] == "pool":
            mode = s["mode"]
            d = self.llm.chat_json(pool_prompt(p, s, mode), schema=POOL_SCHEMA, seed=seed, temperature=TEMP,
                                   max_tokens=1500, tag=f"{s['query_id']}-pool")
            intro = d["self_intro"].strip()
            company, jd = (d["company"].strip(), d["job_description"].strip()) if mode == "JOB_TAILORED" else ("", "")
            qt = pool_query_text(p, mode, intro, company, jd)
            fact_text = intro
            base.update(prev_question=None, answer=None, query_text=qt,
                        context_fields={"mode": mode, "job_categories": [p["slot"]["job_role"]], "self_intro_answer": intro,
                                        "target_company_name": company or None, "target_job_description": jd or None,
                                        "documents": [d_["doc_id"] for d_ in p["docs"]]})
        else:
            d = self.llm.chat_json(qa_prompt(p, s), schema=QA_SCHEMA, seed=seed, temperature=TEMP, max_tokens=800,
                                   tag=f"{s['query_id']}-{s['qtype']}")
            q, a = d["question"].strip(), d["answer"].strip()
            sep = "\n\n" if s["qtype"] == "followup" else "\n"  # 운영 followup_consumer / feedback_consumer
            qt = f"{q}{sep}{a}"
            fact_text = q if s["answer_quality"] == "dontknow" else a
            base.update(prev_question=q, answer=a, query_text=qt, context_fields={})
        j = jaccard(tokens(fact_text), ev) if ev else None
        cov = len(tokens(fact_text) & ev) / len(ev) if ev else None
        base.update(overlap_jaccard=None if j is None else round(j, 4),
                    evidence_coverage=None if cov is None else round(cov, 4), fact_text=fact_text)
        return base


def generate_persona(gen: QueryGen, p: dict[str, Any], n_na: int, idx: int) -> list[dict[str, Any]]:
    w = _out() / "work" / f"{p['pid']}.jsonl"
    if w.exists():
        return [json.loads(x) for x in w.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    t0 = time.monotonic()
    for s in schedule(p, n_na, idx):
        out.append(gen.one(p, s))
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"== {p['pid']} {len(out)}개 {time.monotonic() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    from servers.llama_server import base_url  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    split = json.loads((_out() / "split.json").read_text(encoding="utf-8"))
    all_pids = sorted(split["dev"] + split["test"])
    pids = all_pids if a.all else a.personas.split(",")
    two = no_answer_personas(all_pids)
    llm = LocalLLM(base_url(GEN_MODEL), model=GEN_MODEL, log_path=_out() / "work" / "calls.jsonl")  # cache_prompt=False
    gen = QueryGen(llm)
    rows = []
    for pid in pids:
        p = load_persona(pid)
        rows += generate_persona(gen, p, 2 if pid in two else 1, int(pid[1:]))
    if a.all:
        (_out() / "queries.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
        write_json(_out() / "manifest.json", {
            "created_at": now_iso(), "script": "ai/scripts/rag_eval/queries/generate.py", "stackup_commit": stackup_commit(),
            "generator": GEN_MODEL, "cache_prompt": False, "temperature": TEMP, "seed_rule": "90000 + 번호×1000 + 질의×10, 라운드 +50",
            "bands": {"high_min": HIGH_MIN, "low_max": LOW_MAX}, "n": len(rows),
            "by_qtype": dict(Counter(r["qtype"] for r in rows)), "no_answer": sum(r["no_answer"] for r in rows)})


if __name__ == "__main__":
    main()
