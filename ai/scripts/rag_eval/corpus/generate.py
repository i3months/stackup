"""E1a 코퍼스 생성 (T03): 페르소나 설계 → 이력서·자소서(·레포) → 사실 추출 → 운영 분석 요약.

    python -m corpus.generate --personas P01,P21,P35      # 파일럿
    python -m corpus.generate --all

- 생성 모델은 M-M (llama-server, 생각 모드 끔). 서버는 미리 띄운다:
  python servers/llama_server.py start M-M --ctx 32768
- 단계마다 results/E1-corpus/work/<persona>/ 에 저장하고, 있으면 건너뛴다 (재실행 안전).
- 시드: 페르소나 시드 = BASE_SEED + 번호, 단계마다 고정 오프셋 (재시도는 +100).
- 문서는 docs/<persona>/<doc_id>.md, 분석 요약은 analysis/<doc_id>.md (+ .json).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import results_dir, stackup_commit  # noqa: E402
from local_llm import LocalLLM, LocalLLMError  # noqa: E402

BASE_SEED = 7000
GEN_MODEL = "M-M"  # gemma-4-26B-A4B-it UD-Q4_K_M (P-008)
TEMP_CREATIVE = 0.8  # 설계·문서 (다양성)
TEMP_EXTRACT = 0.1  # 사실 추출
FACT_MAX_TOKENS = 4096  # 사실 추출 출력 상한 (반복 루프 차단, 정상 최대 966)
ENGINEERING = {"backend", "frontend", "infra", "data", "mobile"}
CAREER_BANDS = {"4~7", "8+"}  # 경력기술서 추가 (관리자 T03 답변 1)

DOMAINS = ["이커머스", "핀테크·결제", "물류·배송", "헬스케어", "에듀테크", "게임", "미디어·콘텐츠", "공공·행정",
           "제조·스마트팩토리", "여행·숙박", "부동산·프롭테크", "모빌리티", "광고·마케팅 플랫폼", "SaaS·B2B 협업툴",
           "보험", "통신", "식품·외식", "패션·리테일"]
ROLE_KO = {"BACKEND": "백엔드 개발자", "FRONTEND": "프론트엔드 개발자", "INFRA": "인프라/DevOps 엔지니어",
           "DATA_AI": "데이터 엔지니어/데이터 사이언티스트", "MOBILE": "모바일 앱 개발자",
           "PLANNING": "서비스 기획자(PM)", "MARKETING": "마케터", "DESIGN": "프로덕트(UX/UI) 디자이너"}
YEARS_HINT = {"신입": "경력 0년 (인턴·부트캠프·학교·개인 프로젝트 위주)", "1~3": "경력 1~3년",
              "4~7": "경력 4~7년", "8+": "경력 8년 이상"}
FACT_TYPES = ["number", "tool", "incident", "code", "role", "outcome"]

# ---------------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "headline": {"type": "string"},
        "years_of_experience": {"type": "integer"},
        "education": {"type": "string"},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "experiences": {
            "type": "array", "minItems": 6, "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "experience_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["work", "project", "activity", "education"]},
                    "title": {"type": "string"},
                    "org": {"type": "string"},
                    "period": {"type": "string"},
                    "context": {"type": "string"},
                    "actions": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "outcomes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["experience_id", "kind", "title", "org", "period", "context", "actions", "outcomes", "tools"],
            },
        },
        "confusable_pairs": {
            "type": "array", "minItems": 2,
            "items": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                               "shared": {"type": "string"}, "difference": {"type": "string"}},
                "required": ["a", "b", "shared", "difference"],
            },
        },
        "repo_experience_id": {"type": "string"},
    },
    "required": ["name", "headline", "years_of_experience", "education", "certifications", "experiences",
                 "confusable_pairs", "repo_experience_id"],
}

PLAN_SYSTEM = (
    "당신은 채용 연구용 합성 데이터를 설계하는 사람입니다. 실존하지 않는 한국인 구직자 한 명의 경력 설계서를 "
    "JSON 으로 만듭니다. 이 데이터는 면접 질문 생성 연구의 평가용 코퍼스로 쓰입니다.\n"
    "규칙:\n"
    "- 이름·회사·학교·서비스 이름은 모두 지어낸 것이어야 합니다. 실존 기업·실존 인물을 쓰지 마세요. "
    "연락처·이메일·주소·주민번호·전화번호는 넣지 마세요.\n"
    "- 경험(experiences)은 6~7개. experience_id 는 E1, E2, ... 순서입니다.\n"
    "- 모든 경험에는 검증 가능한 구체 값이 2개 이상 있어야 합니다: 수치(%, 건, ms, 명, 원, 배), 기간, 규모, "
    "도구와 버전, 설정값, 고유 이름(시스템·기능·캠페인 이름) 등. '성능을 개선했다' 같은 모호한 문장만으로 된 경험은 안 됩니다.\n"
    "- **혼동 경험 쌍(confusable_pairs)을 2쌍 이상** 만드세요. 한 쌍의 두 경험은 같은 도구·같은 주제·같은 종류의 지표를 "
    "다루지만 목적·맥락·수치가 서로 다릅니다 (예: 두 프로젝트 모두 Redis 를 썼지만 하나는 세션 저장, 하나는 재고 분산 락이고 "
    "수치도 다름). shared 에 공통점, difference 에 차이를 적습니다.\n"
    "- 오늘은 2026년 10월입니다. 먼저 연차 N(주어진 범위 안의 정수)을 정하고, kind=work 경험들의 기간(YYYY.MM ~ YYYY.MM, "
    "진행 중이면 '~ 현재')을 더한 합계가 N년 ± 6개월이 되도록 시작·종료 연월을 거꾸로 계산해 정하세요. "
    "years_of_experience 와 headline 의 'N년차' 는 같은 N 입니다. 신입이면 회사 정규직 경력 없이 인턴·부트캠프·학교·개인 프로젝트로 구성합니다.\n"
    "- 직무가 개발이 아니면 기술 스택 대신 그 직무의 실무 도구·역량(예: GA4, Figma, Jira, SQL, A/B 테스트)을 씁니다.\n"
    "- 한국어로 쓰고, 기술 용어는 영문 원어를 그대로 둡니다."
)


def _repo_block(repo: dict[str, Any] | None, readme: str) -> str:
    if not repo:
        return "이 지원자는 개발 직군이 아니므로 공개 레포가 없습니다. repo_experience_id 는 빈 문자열로 둡니다."
    return (
        "이 지원자의 포트폴리오 GitHub 레포는 아래 공개 레포입니다 (지원자가 직접 만든 사이드 프로젝트로 설정합니다).\n"
        f"- 레포: {repo['full_name']} ({repo.get('language') or '?'})\n- 설명: {repo.get('description') or ''}\n"
        f"- README 앞부분:\n<<<\n{readme[:3000]}\n>>>\n"
        "경험 하나를 이 레포 프로젝트로 만들고 그 experience_id 를 repo_experience_id 에 적으세요. 그 경험의 내용(기능·도구·수치)은 "
        "README 와 모순되면 안 되며, README 에 없는 성과 수치를 지어낼 때는 개인 사용 맥락(예: 혼자 운영, 사용자 수)으로 한정하세요."
    )


def _ym(m: int) -> str:
    return f"{m // 12}.{m % 12 + 1:02d}"


def career_timeline(band: str, seed: int) -> dict[str, Any]:
    """연차 N 과 회사별 재직 기간을 코드가 정한다 — M-M 이 날짜 합계를 자주 틀려서 (T03 P02: 4회 연속 실패).
    회사 기간은 이어지고 마지막 회사는 '현재'(2026.10)까지. 합계 = N년 ± 3개월, 연차 구간 안."""
    rng = random.Random(seed + 7)
    if band == "신입":
        return {"years": 0, "companies": []}
    lo, hi = BAND_RANGE[band]
    n = rng.randint(lo, min(hi, 14))
    months = min(max(n * 12 + rng.randint(-3, 3), lo * 12 + 1), (hi + 1) * 12 - 1)
    k = 1 if n <= 3 else (2 if n <= 7 else 3)
    w = [rng.uniform(0.6, 1.4) for _ in range(k)]
    parts = [max(12, round(months * x / sum(w))) for x in w]
    parts[-1] = months - sum(parts[:-1])
    end = NOW[0] * 12 + NOW[1] - 1  # 이번 달까지 재직 (끝 월 포함 → 길이 계산은 +1)
    spans, cur = [], end + 1
    for m in reversed(parts):
        spans.append((cur - m, cur))
        cur -= m
    spans.reverse()
    comps = [{"start": _ym(a), "end": "현재" if i == len(spans) - 1 else _ym(b - 1)} for i, (a, b) in enumerate(spans)]
    return {"years": n, "companies": comps}


def plan_prompt(slot: dict[str, str], domain: str, repo: dict[str, Any] | None, readme: str,
                used_names: list[str] | None = None, timeline: dict[str, Any] | None = None) -> list[dict[str, str]]:
    avoid = f"이미 다른 지원자가 쓴 이름이라 쓰면 안 되는 이름: {', '.join(used_names)}\n" if used_names else ""
    tl = timeline or {"years": 0, "companies": []}
    if tl["companies"]:
        comp = "\n".join(f"  - 회사 {i + 1}: {c['start']} ~ {c['end']}" for i, c in enumerate(tl["companies"]))
        career = (f"연차: {tl['years']}년 (years_of_experience = {tl['years']}, headline 의 '{tl['years']}년차').\n"
                  f"정규직 재직 기간은 정확히 다음과 같습니다 (회사 이름은 지어냄):\n{comp}\n"
                  "kind=work 경험의 period 에는 그 경험이 속한 회사의 재직 기간을 위 표기 그대로 쓰세요. "
                  "한 회사에서 work 경험이 여러 개면 같은 기간을 씁니다. 개인·사이드 프로젝트는 kind=project 로 둡니다.\n")
    else:
        career = ("연차: 신입 (years_of_experience = 0, headline 은 '<직무명>, 신입'). 정규직(kind=work) 경험은 만들지 마세요. "
                  "인턴·부트캠프·동아리·학교 프로젝트는 kind=activity·project·education 으로 둡니다.\n")
    user = (
        avoid +
        f"직무: {ROLE_KO[slot['job_role']]} (운영 직군 코드 {slot['job_role']})\n"
        + career +
        f"주력 산업 도메인: {domain} (경험의 절반 이상을 이 도메인으로)\n\n"
        f"{_repo_block(repo, readme)}\n\n"
        "headline 은 '<직무명>, <N>년차' (신입이면 '<직무명>, 신입') 형식의 짧은 한 줄입니다. 설계서 JSON 만 출력하세요."
    )
    return [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}]


def _style_examples() -> str:
    import cases  # noqa: PLC0415  (common.py 가 llm_eval 경로를 sys.path 에 넣는다)

    return cases.RESUME_INFRA_DBA


RESUME_SYSTEM = (
    "당신은 구직자 본인이 되어 한국어 이력서를 씁니다. 이력서는 PDF 에서 텍스트를 추출한 것처럼 마크다운 헤딩과 불릿으로 된 "
    "평문입니다. 주어진 설계서의 사실만 쓰고, 설계서의 모든 경험과 모든 구체 값(수치·도구·이름·기간)을 빠짐없이 그대로 넣습니다. "
    "설계서에 없는 새 수치를 지어내지 마세요. 연락처·이메일·주소는 쓰지 마세요."
)


def resume_prompt(plan: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        "아래 형식 예시는 **형식만** 참고하세요 (내용은 쓰지 마세요):\n<<<\n" + _style_examples() + "\n>>>\n\n"
        "형식: 첫 줄 '# 이력서 — <이름> (<headline>)', 그 다음 '## 요약'(불릿 3개), '## 경력'(회사별 '### 회사 — 역할 (기간)' 아래 불릿), "
        "'## 프로젝트'(프로젝트별 '### 이름 (기간)'), '## 기술 스택' 또는 '## 역량·도구', '## 학력·자격'. 해당 없는 절은 생략합니다.\n\n"
        "설계서:\n" + json.dumps(plan, ensure_ascii=False, indent=1) + "\n\n이력서 본문만 출력하세요."
    )
    return [{"role": "system", "content": RESUME_SYSTEM}, {"role": "user", "content": user}]


COVER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array", "minItems": 3, "maxItems": 4,
            "items": {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                      "required": ["question", "answer"]},
        },
    },
    "required": ["title", "items"],
}
COVER_SYSTEM = (
    "당신은 구직자 본인이 되어 한국어 자기소개서를 씁니다. 국내 기업 자소서 문항(지원동기, 직무 역량을 보여 준 경험, "
    "협업·갈등 해결, 실패·성장 경험, 입사 후 포부 중 3~4개)에 답합니다. 문항당 답변은 500~900자입니다.\n"
    "- 설계서의 경험 중 3개 이상을 사용하고, 혼동 경험 쌍 중 최소 한 쌍의 두 경험을 서로 다른 문항에서 다룹니다.\n"
    "- 경험을 쓸 때 설계서의 구체 값(수치·도구·이름)을 그대로 쓰고, 이력서에는 없을 법한 맥락(왜 그렇게 했는지, 무엇을 배웠는지, "
    "누구와 어떻게 협업했는지)을 덧붙입니다. 설계서와 모순되는 새 수치는 쓰지 않습니다.\n"
    "- title 은 '지원 회사 이름 + 직무 + 지원' 형태입니다 (예: 한빛로지스 백엔드 개발자 지원). 회사 이름은 지어내고 "
    "꺾쇠·대괄호 없이 씁니다. 실존 기업 이름은 쓰지 않습니다.\n"
    "- 연락처·이메일·주소는 쓰지 않습니다."
)


def cover_prompt(plan: dict[str, Any]) -> list[dict[str, str]]:
    user = "설계서:\n" + json.dumps(plan, ensure_ascii=False, indent=1) + "\n\n자기소개서를 JSON 으로 출력하세요."
    return [{"role": "system", "content": COVER_SYSTEM}, {"role": "user", "content": user}]


CAREER_SYSTEM = (
    "당신은 경력직 구직자 본인이 되어 한국어 경력기술서를 씁니다. 경력기술서는 이력서보다 자세하게 회사·프로젝트별로 "
    "무엇을 왜 어떻게 했는지 적는 문서입니다. 설계서의 경력(work) 경험과 주요 프로젝트를 모두 다루고, 설계서의 구체 값을 "
    "그대로 쓰되 이력서에 없을 법한 세부(문제 상황, 기술 선택 이유, 구현 방식, 협업 방식, 측정 방법)를 덧붙입니다. "
    "설계서와 모순되는 새 수치는 쓰지 않습니다. 연락처·이메일·주소는 쓰지 않습니다."
)


def career_prompt(plan: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        "형식: 첫 줄 '# 경력기술서 — <이름> (<headline>)'. 회사마다 '## 회사 — 직무 (기간)', 그 아래 프로젝트마다 "
        "'### 프로젝트 이름 (기간)' 과 '- 역할:', '- 사용 기술:', '- 배경:', '- 주요 업무:'(불릿 여러 개), '- 성과:'(불릿) 항목. "
        "분량은 3,000~5,000자.\n\n설계서:\n" + json.dumps(plan, ensure_ascii=False, indent=1) + "\n\n경력기술서 본문만 출력하세요."
    )
    return [{"role": "system", "content": CAREER_SYSTEM}, {"role": "user", "content": user}]


def _java_trim(s: str) -> str:
    # Java String.trim(): 양 끝의 코드포인트 <= U+0020 만 제거
    i, j = 0, len(s)
    while i < j and ord(s[i]) <= 0x20:
        i += 1
    while j > i and ord(s[j - 1]) <= 0x20:
        j -= 1
    return s[i:j]


def build_cover_markdown(title: str, items: list[dict[str, str]]) -> str:
    """Core AnalysisRequestService.buildMarkdown 의 파이썬 이식 (문항 JSON → 분석·임베딩용 마크다운)."""
    sb = ["# 자기소개서"]
    if title is not None and _java_trim(title):
        sb.append(" — " + _java_trim(title))
    sb.append("\n\n")
    for it in items:
        q = _java_trim(it.get("question") or "")
        a = _java_trim(it.get("answer") or "")
        if not a:
            continue
        sb.append("## " + (q or "문항") + "\n")
        sb.append(a + "\n\n")
    return "".join(sb)


FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array", "minItems": 6, "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "experience_id": {"type": "string"},
                    "fact_type": {"type": "string", "enum": FACT_TYPES},
                    "statement": {"type": "string"},
                    "evidence_text": {"type": "string"},
                },
                "required": ["experience_id", "fact_type", "statement", "evidence_text"],
            },
        }
    },
    "required": ["facts"],
}
FACT_SYSTEM = (
    "당신은 문서에서 검증 가능한 사실을 뽑는 사람입니다. 사실은 면접 질문과 검색 평가의 정답 기준이 됩니다.\n"
    "규칙:\n"
    "- 문서당 6~10개. 서로 다른 내용을 고르게 뽑습니다 (한 경험에 몰지 않음).\n"
    "- 사실마다 **검증 가능한 값**(수치, 도구·버전, 설정값, 고유 이름, 기간, 역할)이 있어야 합니다. 모호한 사실 금지.\n"
    "- evidence_text 는 문서에서 **글자 하나 바꾸지 않고 그대로 복사한 연속 구간**입니다 (15~120자, 줄바꿈 없이 한 줄 안에서). "
    "사실의 핵심 값을 반드시 포함합니다. 요약·의역·생략 부호 금지.\n"
    "- statement 는 그 사실을 한 문장으로 쓴 것입니다 (주어 생략 가능).\n"
    "- experience_id 는 주어진 경험 목록에서 그 사실이 속한 경험의 ID 입니다. 어느 경험에도 속하지 않으면 'NONE'.\n"
    "- fact_type: number(수치), tool(도구·기술), incident(사건·장애·문제), code(코드·설정·함수명·의존성), role(역할·책임), outcome(성과)."
)


def fact_prompt(doc_text: str, plan: dict[str, Any], doc_type: str, feedback: str = "") -> list[dict[str, str]]:
    exps = "\n".join(f"- {e['experience_id']}: {e['title']} ({e['org']}, {e['period']})" for e in plan["experiences"])
    extra = ""
    if doc_type == "repo":
        extra = ("\n이 문서는 GitHub 레포(README·파일 트리·소스 일부)입니다. README·코드에서 함수명, 설정값, 의존성과 버전, "
                 f"명령어 같은 사실을 뽑고 experience_id 는 {plan['repo_experience_id']} 로 둡니다.")
    user = f"경험 목록:\n{exps}{extra}\n\n문서:\n<<<\n{doc_text}\n>>>\n"
    if feedback:
        user += f"\n이전 시도의 문제:\n{feedback}\n문제를 고쳐 다시 뽑으세요.\n"
    user += "\n사실 목록을 JSON 으로 출력하세요."
    return [{"role": "system", "content": FACT_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
class Generator:
    def __init__(self, llm: LocalLLM, *, repos: dict[str, dict[str, Any]]) -> None:
        self.llm = llm
        self.repos = repos
        self.corpus = results_dir("E1-corpus")

    def work(self, pid: str) -> Path:
        d = self.corpus / "work" / pid
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- 단계 ----------------------------------------------------------------
    def plan(self, slot: dict[str, str], seed: int) -> dict[str, Any]:
        p = self.work(slot["persona_id"]) / "plan.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        repo = self.repos.get(slot["persona_id"]) if slot["job_family"] in ENGINEERING else None
        readme = ""
        if repo:
            from corpus.repo_source import build_repo_text  # noqa: PLC0415

            text, _ = self.repo_text(slot["persona_id"], repo)
            readme = text.split("\n## README\n", 1)[-1].split("\n## File tree", 1)[0] if "## README" in text else ""
        domain = random.Random(seed).choice(DOMAINS)
        used = sorted({json.loads(q.read_text(encoding="utf-8"))["name"]
                       for q in (self.corpus / "work").glob("*/plan.json") if q.parent.name != slot["persona_id"]})
        timeline = career_timeline(slot["years_band"], seed)
        # 검증 실패가 4회 이어지면 seed 만 +50 바꿔 다시 4회 (최대 3개 seed, 관리자 T03 답변: P14).
        # 프롬프트·도메인·타임라인은 그대로다. 시도 이력은 _meta.history 에 남긴다.
        history: list[dict[str, Any]] = []
        for rnd in range(3):
            problems: list[str] = []
            for attempt in range(4):
                msgs = plan_prompt(slot, domain, repo, readme, used, timeline)
                if problems:
                    msgs[-1]["content"] += "\n\n이전 시도의 문제 (반드시 고칠 것): " + "; ".join(problems)
                s = seed + 50 * rnd + 100 * attempt
                plan = self.llm.chat_json(msgs, schema=PLAN_SCHEMA, seed=s, temperature=TEMP_CREATIVE,
                                          tag=f"{slot['persona_id']}-plan")
                problems = validate_plan(plan, need_repo=repo is not None, band=slot['years_band'], timeline=timeline)
                if plan["name"] in used:
                    problems.append(f"이름 {plan['name']} 중복")
                history.append({"seed_round": rnd, "attempt": attempt + 1, "seed": s, "problems": problems})
                if not problems:
                    break
                print(f"  plan 재시도 {rnd}-{attempt + 1}: {problems}")
            if not problems:
                break
            print(f"  plan seed {seed + 50 * rnd} 4회 실패 → seed {seed + 50 * (rnd + 1)}")
        else:
            raise RuntimeError(f"{slot['persona_id']} 설계서 검증 실패: {problems}")
        plan["_meta"] = {"domain": domain, "seed": s, "generator": GEN_MODEL, "attempts": len(history),
                         "seed_round": rnd, "history": history,
                         "timeline": timeline, "generator_commit": stackup_commit()}
        p.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        return plan

    def repo_text(self, pid: str, repo: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        p = self.work(pid) / "repo_source.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d["text"], d["meta"]
        from corpus.repo_source import build_repo_text  # noqa: PLC0415

        text, meta = build_repo_text(repo["full_name"], repo["commit"], repo["default_branch"], repo.get("description") or "")
        p.write_text(json.dumps({"text": text, "meta": meta}, ensure_ascii=False), encoding="utf-8")
        return text, meta

    def resume(self, pid: str, plan: dict[str, Any], seed: int) -> str:
        p = self.work(pid) / "resume.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
        text = self.llm.chat(resume_prompt(plan), seed=seed, temperature=TEMP_CREATIVE, tag=f"{pid}-resume").strip() + "\n"
        p.write_text(text, encoding="utf-8")
        return text

    def career(self, pid: str, plan: dict[str, Any], seed: int) -> tuple[str, int]:
        """경력기술서와 실제 쓴 seed. 서버가 출력 형식 파싱 오류(HTTP 500)를 내면 seed 를 바꿔 재시도한다
        (T03 P36: 'does not match the expected peg-gemma4 format'). 고정 seed 로는 같은 출력이 나온다."""
        import urllib.error  # noqa: PLC0415

        p, pm = self.work(pid) / "career.md", self.work(pid) / "career_meta.json"
        if p.exists():
            used = json.loads(pm.read_text(encoding="utf-8"))["seed"] if pm.exists() else seed
            return p.read_text(encoding="utf-8"), used
        for k in range(3):
            try:
                text = self.llm.chat(career_prompt(plan), seed=seed + 50 * k, temperature=TEMP_CREATIVE,
                                     tag=f"{pid}-career").strip() + "\n"
                break
            except urllib.error.HTTPError as e:
                if e.code != 500 or k == 2:
                    raise
                print(f"  career 서버 500, seed {seed + 50 * k} → {seed + 50 * (k + 1)} 로 재시도")
        pm.write_text(json.dumps({"seed": seed + 50 * k, "server_500_retries": k}), encoding="utf-8")
        p.write_text(text, encoding="utf-8")
        return text, seed + 50 * k

    def doc_seed(self, pid: str, doc: str, seed: int) -> int:
        """문서 재생성 seed. work/<pid>/doc_seed_rounds.json 의 {doc: r} 이면 seed + 50r (P-017: 프롬프트 유지,
        seed 만 변경. T03 P39 자소서: 사실 추출이 9회 모두 반복 루프)."""
        f = self.work(pid) / "doc_seed_rounds.json"
        r = json.loads(f.read_text(encoding="utf-8")).get(doc, 0) if f.exists() else 0
        return seed + 50 * r

    def cover(self, pid: str, plan: dict[str, Any], seed: int) -> tuple[str, int]:
        p = self.work(pid) / "cover.json"
        seed = self.doc_seed(pid, "cover", seed)
        if not p.exists():
            d = self.llm.chat_json(cover_prompt(plan), schema=COVER_SCHEMA, seed=seed, temperature=TEMP_CREATIVE,
                                   tag=f"{pid}-cover")
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        d = json.loads(p.read_text(encoding="utf-8"))
        return build_cover_markdown(d["title"], d["items"]), seed

    def facts(self, doc_id: str, doc_type: str, text: str, plan: dict[str, Any], seed: int) -> list[dict[str, Any]]:
        pid = doc_id.split("-")[0]
        p = self.work(pid) / f"facts-{doc_type}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        valid_ids = {e["experience_id"] for e in plan["experiences"]} | {"NONE"}
        kept: list[dict[str, Any]] = []
        feedback = ""
        capped = False
        for i in range(9):
            # 상한은 반복 루프 차단용 (T03 P39: 30분·2.5만 토큰 생성 후 타임아웃). 정상 출력은 최대 966 토큰이라
            # 상한 아래에서 끝나는 출력은 상한이 없을 때와 같다. 상한에 걸리면 다음 seed 로 넘어간다.
            # 기본 3회. 상한에 걸린 적이 있고 사실이 6개 미만이면 seed +50 라운드를 최대 2번 더 (설계서와 같은 규칙,
            # P39 자소서: 3회 모두 근거 문장을 끝없이 이어 붙임).
            rnd, attempt = divmod(i, 3)
            if rnd and not capped:
                break
            try:
                raw = self.llm.chat_json(fact_prompt(text, plan, doc_type, feedback), schema=FACT_SCHEMA,
                                         seed=seed + 50 * rnd + 100 * attempt, temperature=TEMP_EXTRACT,
                                         max_tokens=FACT_MAX_TOKENS, tag=f"{doc_id}-facts")
            except LocalLLMError as e:
                capped = True
                print(f"  {doc_id} facts 시도 {rnd}-{attempt + 1}: {e}")
                continue
            bad = []
            for f in raw["facts"]:
                ev = f["evidence_text"]
                if any(ev == k["evidence_text"] for k in kept):
                    continue
                reason = check_evidence(text, ev)
                if f["experience_id"] not in valid_ids:
                    reason = reason or f"experience_id {f['experience_id']} 없음"
                if reason:
                    bad.append(f"- '{ev[:60]}': {reason}")
                    continue
                kept.append({**f, "attempt": i + 1})
            if len(kept) >= 6:
                break
            feedback = (f"유효한 사실이 {len(kept)}개뿐입니다. 다음은 버려졌습니다:\n" + "\n".join(bad[:10])
                        + "\n이미 채택된 근거와 다른 사실을 더 뽑으세요.")
        kept = kept[:10]
        out = []
        for i, f in enumerate(kept, 1):
            start = text.find(f["evidence_text"])
            out.append({
                "fact_id": f"{doc_id}-F{i:02d}", "persona_id": pid, "doc_id": doc_id,
                "experience_id": None if f["experience_id"] == "NONE" else f["experience_id"],
                "fact_type": f["fact_type"], "statement": f["statement"], "evidence_text": f["evidence_text"],
                "char_start": start, "char_end": start + len(f["evidence_text"]),
                "evidence_unique": text.count(f["evidence_text"]) == 1,
                "preserved_in_summary": None, "generator": GEN_MODEL, "attempt": f["attempt"],
            })
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out

    def analyze(self, doc_id: str, source_type: str, text: str, seed: int) -> dict[str, Any]:
        """운영 문서 분석 체인과 같은 프롬프트·온도·파서. LLM 만 M-M (B0 = 운영 구조 + 로컬 모델)."""
        out_md = self.corpus / "analysis" / f"{doc_id}.md"
        out_js = self.corpus / "analysis" / f"{doc_id}.json"
        if out_js.exists():
            return json.loads(out_js.read_text(encoding="utf-8"))
        from langchain_core.output_parsers import PydanticOutputParser  # noqa: PLC0415
        from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415

        from ai_server.chain.document_analysis_chain import DocumentAnalysisResult  # noqa: PLC0415
        from ai_server.chain.prompts.document_analysis import HUMAN_PROMPT, SYSTEM_PROMPT  # noqa: PLC0415
        from ai_server.config.settings import Settings  # noqa: PLC0415

        s = Settings(_env_file=None)
        parser = PydanticOutputParser(pydantic_object=DocumentAnalysisResult)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]).partial(
            format_instructions=parser.get_format_instructions())
        msgs = prompt.format_messages(text=text, source_type=source_type)
        role = {"system": "system", "human": "user"}
        messages = [{"role": role[m.type], "content": m.content} for m in msgs]
        errors = []
        for attempt in range(s.document_analysis_max_attempts):  # 운영 with_retry 와 같은 3회
            # 운영 with_retry 는 모든 예외를 잡으므로 서버 오류(500)도 한 번의 실패 시도로 센다 (T03 P39).
            try:
                raw = self.llm.chat(messages, seed=seed + 100 * attempt, temperature=s.llm_pro_temperature,
                                    tag=f"{doc_id}-analysis")
                res = parser.parse(raw)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        else:
            raise RuntimeError(f"{doc_id} 분석 파싱 3회 실패: {errors}")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(res.markdown, encoding="utf-8")
        d = {"doc_id": doc_id, "source_type": source_type, "summary": res.summary, "tech_stack": res.tech_stack,
             "generator": GEN_MODEL, "seed": seed + 100 * attempt, "attempts": attempt + 1, "parse_errors": errors,
             "temperature": s.llm_pro_temperature, "prompt": "ai_server.chain.prompts.document_analysis"}
        out_js.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return d

    # ---- 페르소나 전체 ----------------------------------------------------------
    def persona(self, slot: dict[str, str], idx: int) -> dict[str, Any]:
        pid = slot["persona_id"]
        seed = BASE_SEED + idx * 1000
        t0 = time.monotonic()
        plan = self.plan(slot, seed)
        docs: list[dict[str, Any]] = []
        resume = self.resume(pid, plan, seed + 1)
        docs.append({"doc_id": f"{pid}-resume", "doc_type": "resume", "source_type": "PDF", "text": resume,
                     "source": "synthetic", "seed": seed + 1})
        cover, cover_seed = self.cover(pid, plan, seed + 2)
        docs.append({"doc_id": f"{pid}-cover", "doc_type": "cover_letter", "source_type": "COVER_LETTER", "text": cover,
                     "source": "synthetic", "seed": cover_seed})
        if slot["years_band"] in CAREER_BANDS:
            career, career_seed = self.career(pid, plan, seed + 3)
            # 경력기술서는 운영에서 이력서 PDF 슬롯으로 올라간다 (source_type PDF)
            docs.append({"doc_id": f"{pid}-career", "doc_type": "career_description", "source_type": "PDF",
                         "text": career, "source": "synthetic", "seed": career_seed})
        if slot["job_family"] in ENGINEERING:
            repo = self.repos[pid]
            text, meta = self.repo_text(pid, repo)
            docs.append({"doc_id": f"{pid}-repo", "doc_type": "repo", "source_type": "REPOSITORY", "text": text,
                         "source": "github", "seed": None, "repo_url": f"https://github.com/{repo['full_name']}",
                         "repo_commit": repo["commit"], "license": repo["license"], "repo_meta": meta})
        ddir = self.corpus / "docs" / pid
        ddir.mkdir(parents=True, exist_ok=True)
        all_facts = []
        for i, d in enumerate(docs):
            (ddir / f"{d['doc_id']}.md").write_text(d["text"], encoding="utf-8")
            all_facts += self.facts(d["doc_id"], d["doc_type"], d["text"], plan, seed + 10 + i)
            a = self.analyze(d["doc_id"], d["source_type"], d["text"], seed + 20 + i)
            d["analysis_attempts"] = a["attempts"]
        meta = {
            "persona_id": pid, "plan_meta": plan["_meta"], "n_experiences": len(plan["experiences"]),
            "n_confusable_pairs": len(plan["confusable_pairs"]), "elapsed_sec": round(time.monotonic() - t0, 1),
            "docs": [{k: v for k, v in d.items() if k != "text"} | {
                "path": f"docs/{pid}/{d['doc_id']}.md", "chars": len(d["text"]),
                "generator": GEN_MODEL if d["source"] == "synthetic" else f"github:{d['repo_url']}",
                "analysis_path": f"analysis/{d['doc_id']}.md"} for d in docs],
            "n_facts": len(all_facts),
        }
        (self.work(pid) / "persona_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return meta


def check_evidence(text: str, ev: str) -> str:
    if not ev or ev not in text:
        return "원문에 그대로 없음"
    if len(ev) < 15:
        return "15자 미만"
    if "\n" in ev:
        return "줄바꿈 포함"
    return ""


BAND_RANGE = {"신입": (0, 0), "1~3": (1, 3), "4~7": (4, 7), "8+": (8, 40)}
NOW = (2026, 10)
_YM = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})")
_PRESENT = re.compile(r"현재|재직|present|now", re.I)


def period_months(period: str) -> tuple[int, int] | None:
    """'2021.03 ~ 현재' → (시작 월 번호, 끝 월 번호). 해석 못 하면 None."""
    ym = _YM.findall(period)
    if not ym:
        return None
    start = int(ym[0][0]) * 12 + int(ym[0][1]) - 1
    if len(ym) >= 2:
        end = int(ym[1][0]) * 12 + int(ym[1][1]) - 1
    elif _PRESENT.search(period):
        end = NOW[0] * 12 + NOW[1] - 1
    else:
        return None
    return start, end + 1


def career_years(plan: dict[str, Any]) -> float | None:
    """kind=work 경험 기간의 합집합 (겹치는 기간은 한 번만). 해석 실패 시 None."""
    spans = []
    for e in plan["experiences"]:
        if e["kind"] != "work":
            continue
        pm = period_months(e["period"])
        if pm is None:
            return None
        spans.append(pm)
    months, cur = 0, None
    for a, b in sorted(spans):
        if cur is None or a > cur[1]:
            if cur:
                months += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        months += cur[1] - cur[0]
    return months / 12


def validate_plan(plan: dict[str, Any], *, need_repo: bool, band: str,
                  timeline: dict[str, Any] | None = None) -> list[str]:
    """연차 구간은 경력 합계로 판단하고, 합계와 years_of_experience(headline) 차이는 1년 이내 (관리자 T03 답변 3).
    연차 검사가 실패하면 지정 재직 기간 밖의 kind=work 경험을 짚어 재시도 피드백에 넣는다 (T03 P10: 지정 밖 work 경험을 더해 4회 실패)."""
    problems = []
    lo, hi = BAND_RANGE[band]
    cy = career_years(plan)
    n = plan["years_of_experience"]
    m = re.search(r"(\d+)\s*년차", plan["headline"])
    if cy is None:
        problems.append("경력 기간 해석 실패")
    elif band == "신입":
        if cy >= 1.0 or n != 0:
            problems.append(f"신입인데 정규 경력 {cy:.1f}년 / years {n}")
    else:
        if not lo <= cy < hi + 1:
            problems.append(f"경력 합계 {cy:.1f}년이 {band} 범위 밖")
        if abs(cy - n) > 1.0:
            problems.append(f"경력 합계 {cy:.1f}년과 연차 {n}년 차이 > 1년")
        if not m or int(m.group(1)) != n:
            problems.append(f"headline '{plan['headline']}' 의 연차가 {n} 과 다름")
    if problems and timeline and timeline["companies"]:  # 판정 기준은 그대로, 연차 실패 때 고칠 곳만 짚어 준다
        allowed = {period_months(f"{c['start']} ~ {c['end']}") for c in timeline["companies"]}
        bad = [e for e in plan["experiences"] if e["kind"] == "work" and period_months(e["period"]) not in allowed]
        if bad:
            spans = ", ".join(f"'{c['start']} ~ {c['end']}'" for c in timeline["companies"])
            problems.append("kind=work 경험 " + ", ".join(f"{e['experience_id']}('{e['period']}')" for e in bad)
                            + f" 의 기간이 지정 재직 기간({spans})이 아님 → 지정 기간으로 바꾸거나 kind=project·activity 로")
    ids = [e["experience_id"] for e in plan["experiences"]]
    if len(set(ids)) != len(ids):
        problems.append("experience_id 중복")
    if not 6 <= len(ids) <= 7:
        problems.append(f"경험 {len(ids)}개")
    pairs = [cp for cp in plan["confusable_pairs"] if cp["a"] in ids and cp["b"] in ids and cp["a"] != cp["b"]]
    if len(pairs) < 2:
        problems.append(f"유효한 혼동 쌍 {len(pairs)}개")
    if need_repo and plan["repo_experience_id"] not in ids:
        problems.append("repo_experience_id 가 경험 목록에 없음")
    return problems


MIN_WINDOWS_FREE_GB = 8.0  # 관리자 규칙 (P-015 이후)


def windows_free_gb() -> float | None:
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def load_slots() -> list[dict[str, str]]:
    with (results_dir("E1-corpus") / "personas.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    from servers.llama_server import base_url  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repos", default=str(results_dir("E1-corpus") / "repos.json"))
    a = ap.parse_args()
    slots = load_slots()
    want = [s for s in slots if a.all or s["persona_id"] in a.personas.split(",")]
    repos = json.loads(Path(a.repos).read_text(encoding="utf-8")) if Path(a.repos).exists() else {}
    llm = LocalLLM(base_url(GEN_MODEL), model=GEN_MODEL, log_path=results_dir("E1-corpus") / "work" / "calls.jsonl")
    gen = Generator(llm, repos=repos)
    for s in want:
        free = windows_free_gb()
        if free is not None and free < MIN_WINDOWS_FREE_GB:
            print(f"STOP Windows 여유 메모리 {free:.1f}GB < {MIN_WINDOWS_FREE_GB}GB — 관리자 보고 필요", flush=True)
            sys.exit(3)
        idx = int(s["persona_id"][1:])
        print(f"== {s['persona_id']} {s['job_role']} {s['years_band']}", flush=True)
        try:
            m = gen.persona(s, idx)
            print(f"   docs={len(m['docs'])} facts={m['n_facts']} pairs={m['n_confusable_pairs']} {m['elapsed_sec']}s", flush=True)
        except (LocalLLMError, RuntimeError) as exc:
            print(f"   FAIL {exc}", flush=True)


if __name__ == "__main__":
    main()
