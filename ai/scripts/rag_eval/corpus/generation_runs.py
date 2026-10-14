"""실행 로그 구간 → 실행별 코드 커밋(프로세스 시작 시점 HEAD) → 페르소나별 생성 커밋."""
import json, os, re

LOG = os.path.expanduser("~/rag-lab/run/corpus-all.log")
OUT = "/mnt/c/Users/wayso/OneDrive/Desktop/work_dir/stackup-lab/results/E1-corpus/generation-runs.json"
# (로그 시작 줄, 시작 시각, 프로세스가 읽은 코드 커밋, 비고) — 커밋 시각과 재개 시각, 로그 형식으로 확인
RUNS = [
    (1, "03:50", "aef8507", "첫 전체 실행 (P01 생성, P02 연차 실패 후 중단)"),
    (12, "04:15", "0205187", "타임라인 코드 결정 후 재시작"),
    (81, "13:17", "4179336", "사용자 요청 일시 중지 후 재개"),
    (194, "16:18", "ce8a51b", "P36 서버 500 후 재개 (로그 형식상 347c1ee 이전 코드)"),
    (326, "17:42", "40a8cde", "P39 사실 추출 타임아웃 후 재개, 기존 P01 보관"),
    (442, "18:41", "9efbc7f", "P39 사실 상한·분석 500 후 재개 (--personas P39,P40)"),
    (457, "19:49", "9c7e240", "P39 자소서 재생성 (seed 라운드 1)"),
]
lines = open(LOG, encoding="utf-8", errors="replace").read().splitlines()
starts = [r[0] for r in RUNS] + [len(lines) + 1]
personas: dict[str, list[str]] = {}
runs = []
for i, (ln0, t, c, note) in enumerate(RUNS):
    seg = lines[ln0 - 1:starts[i + 1] - 1]
    worked, cur, busy = [], None, False
    for ln in seg + ["== END"]:
        if ln.startswith("== "):
            if cur and busy:
                worked.append(cur)
            m = re.match(r"== (P\d\d) ", ln)
            cur, busy = (m.group(1) if m else None), False
        elif cur and not re.match(r"\s+docs=.* 0\.\ds$", ln):
            busy = True
    runs.append({"log_line": ln0, "start": t, "code_commit": c, "note": note, "personas_worked": worked})
    for p in worked:
        personas.setdefault(p, []).append(c)
# P39 재생성 실행은 --personas 실행이라 헤더가 다른 줄 뒤에 온다 — 위 구간 파싱이 잡는다
json.dump({"log": "~/rag-lab/run/corpus-all.log", "runs": runs, "personas": personas},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in runs:
    print(r["start"], r["code_commit"], r["personas_worked"])
print({p: v for p, v in personas.items() if len(v) > 1})

# 현재 산출물을 실제로 만든 실행만 남긴다: 파일 수정 시각이 그 실행 구간 안인 것 (persona_meta·repo_source 제외)
import datetime, glob
LAB = "/mnt/c/Users/wayso/OneDrive/Desktop/work_dir/stackup-lab/results/E1-corpus"
def hm(t):
    h, m = map(int, t.split(":")); return h * 60 + m
bounds = [hm(r[1]) for r in RUNS] + [24 * 60]
final = {}
for d in sorted(glob.glob(f"{LAB}/work/P*")):
    pid = d.rsplit("/", 1)[1]
    files = [f for f in glob.glob(f"{d}/*") if not f.endswith(("persona_meta.json", "repo_source.json"))]
    files += glob.glob(f"{LAB}/analysis/{pid}-*")
    used = set()
    for f in files:
        t = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        mins = t.hour * 60 + t.minute
        for i in range(len(RUNS)):
            if bounds[i] <= mins < bounds[i + 1]:
                used.add(i)
    final[pid] = [RUNS[i][2] for i in sorted(used)]
data = json.load(open(OUT, encoding="utf-8"))
data["personas_artifacts"] = final
data["note"] = ("personas_worked = 그 실행에서 작업한 페르소나(실패 포함), personas_artifacts = 현재 산출물 파일을 만든 실행의 "
                "코드 커밋. 코드 커밋은 프로세스 시작 시점 HEAD (실행 중 커밋은 반영되지 않는다).")
json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import collections
print(collections.Counter("+".join(v) for v in final.values()))
print({p: v for p, v in final.items() if len(v) != 1 or v[0] != "0205187"})
