"""개발 직군 페르소나 34명에 공개 레포 배정 → results/E1-corpus/repos.json.

    python -m corpus.assign_repos
    python -m corpus.assign_repos extend   # T04b: 레포 없는 새 페르소나만

후보는 results/E1-corpus/repo-candidates.json (작업자 서브에이전트 조사, 라이선스·커밋 GitHub API 확인).
규칙: 허용 라이선스(MIT·Apache-2.0·BSD)만, 파일 300개 초과·50KB 넘는 소스 다수인 레포 제외, 후보 순서대로 페르소나 번호 순 배정.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import results_dir  # noqa: E402

ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
EXCLUDE = {
    "gjovanovicst/golang-auth-api", "felipeall/transfermarkt-api", "simonwep/ocular", "LiuXin-Developer/rightHouse",
    "nellavio/nellavio", "ldakhoa/GistHub",  # 파일 300개 초과
    "Justin3go/SearchSearchGo",  # 50KB 넘는 소스 4개
}


def extend() -> None:
    """T04b: 레포가 없는 개발 직군 페르소나(P41~)에 배정한다. 기존 배정은 그대로 두고, 이미 쓴 레포는 다시 쓰지 않는다
    (청크가 페르소나끼리 겹치지 않게). 후보 순서: 기존 후보 중 안 쓴 것 → repo-candidates-ext.json."""
    corpus = results_dir("E1-corpus")
    cands = json.loads((corpus / "repo-candidates.json").read_text(encoding="utf-8"))
    cands += json.loads((corpus / "repo-candidates-ext.json").read_text(encoding="utf-8"))
    out = json.loads((corpus / "repos.json").read_text(encoding="utf-8"))
    used = {v["full_name"] for v in out.values()}
    pool: dict[str, list[dict]] = {}
    seen = set()
    for c in cands:
        if (c["license"] in ALLOWED and c["full_name"] not in EXCLUDE and c.get("commit") and c["full_name"] not in used
                and c["full_name"] not in seen and not c.get("archived")):
            seen.add(c["full_name"])
            pool.setdefault(c["job_family"], []).append(c)
    with (corpus / "personas.csv").open(encoding="utf-8") as fh:
        slots = list(csv.DictReader(fh))
    took: dict[str, int] = {}
    for s in slots:
        fam = s["job_family"]
        if fam == "non_eng" or s["persona_id"] in out:
            continue
        i = took.get(fam, 0)
        if i >= len(pool.get(fam, [])):
            raise SystemExit(f"{fam} 후보 부족 ({s['persona_id']})")
        c = pool[fam][i]
        took[fam] = i + 1
        out[s["persona_id"]] = {k: c[k] for k in ("full_name", "default_branch", "commit", "license", "language", "description")}
    assert len({v["full_name"] for v in out.values()}) == len(out), "레포 중복"
    (corpus / "repos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), {f: f"{took[f]}/{len(pool[f])}" for f in took})


def main() -> None:
    corpus = results_dir("E1-corpus")
    cands = json.loads((corpus / "repo-candidates.json").read_text(encoding="utf-8"))
    pool: dict[str, list[dict]] = {}
    for c in cands:
        if c["license"] in ALLOWED and c["full_name"] not in EXCLUDE and c.get("commit"):
            pool.setdefault(c["job_family"], []).append(c)
    with (corpus / "personas.csv").open(encoding="utf-8") as fh:
        slots = list(csv.DictReader(fh))
    out = {}
    used: dict[str, int] = {}
    for s in slots:
        fam = s["job_family"]
        if fam == "non_eng":
            continue
        i = used.get(fam, 0)
        c = pool[fam][i]
        used[fam] = i + 1
        out[s["persona_id"]] = {k: c[k] for k in ("full_name", "default_branch", "commit", "license", "language", "description")}
    (corpus / "repos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), {f: f"{used[f]}/{len(pool[f])}" for f in used})


if __name__ == "__main__":
    extend() if sys.argv[1:] == ["extend"] else main()
