"""T03 보고서용 페르소나별 재시도·seed 변경 이력 (실행 로그 + 산출물 메타)."""
import json, re, glob, os, collections

LAB = "/mnt/c/Users/wayso/OneDrive/Desktop/work_dir/stackup-lab/results/E1-corpus"
LOG = os.path.expanduser("~/rag-lab/run/corpus-all.log")

ev = collections.defaultdict(lambda: collections.Counter())
cur = None
for ln in open(LOG, encoding="utf-8", errors="replace"):
    m = re.match(r"== (P\d\d) ", ln)
    if m:
        cur = m.group(1); continue
    if ln.startswith("== "):
        cur = None; continue
    if cur is None:
        continue
    if "plan 재시도" in ln: ev[cur]["plan_retry_log"] += 1
    if re.search(r"FAIL P\d\d", ln): ev[cur]["plan_fail_runs"] += 1
    if "facts 시도" in ln: ev[cur]["facts_cap"] += 1
    if "career 서버 500" in ln: ev[cur]["career_500"] += 1

RUNS = json.load(open(f"{LAB}/generation-runs.json", encoding="utf-8"))["personas_artifacts"]
rows = []
for w in sorted(glob.glob(f"{LAB}/work/P*/plan.json")):
    pid = w.split("/")[-2]
    m = json.load(open(w, encoding="utf-8"))["_meta"]
    ana = []
    for a in glob.glob(f"{LAB}/analysis/{pid}-*.json"):
        d = json.load(open(a, encoding="utf-8"))
        if d["attempts"] > 1:
            ana.append(f"{d['doc_id'].split('-',1)[1]}×{d['attempts']}")
    cm = f"{LAB}/work/{pid}/career_meta.json"
    c500 = json.load(open(cm))["server_500_retries"] if os.path.exists(cm) else 0
    facts_att = []
    for f in glob.glob(f"{LAB}/work/{pid}/facts-*.json"):
        fs = json.load(open(f, encoding="utf-8"))
        mx = max((x["attempt"] for x in fs), default=0)
        if mx > 1:
            facts_att.append(f"{f.split('facts-')[1][:-5]}≤{mx}")
    dsr = f"{LAB}/work/{pid}/doc_seed_rounds.json"
    rows.append({
        "pid": pid, "plan_attempts": m.get("attempts"), "seed_round": m.get("seed_round", 0),
        "commit": "+".join(RUNS.get(pid, [])),  # 산출물을 만든 실행의 코드 커밋 (generation_runs.py)
        "failed_runs": ev[pid]["plan_fail_runs"], "plan_retries_logged": ev[pid]["plan_retry_log"],
        "facts_cap_hits": ev[pid]["facts_cap"], "facts_late_attempts": ",".join(sorted(facts_att)),
        "analysis_retries": ",".join(sorted(ana)), "career_500": c500 + ev[pid]["career_500"],
        "doc_regen": open(dsr).read().strip() if os.path.exists(dsr) else "",
    })
json.dump(rows, open(f"{LAB}/retry-history.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("| 페르소나 | 설계서 시도 (최종 실행) | seed 라운드 | 실패한 실행 | 사실 상한 초과 | 사실 추가 시도 | 분석 재시도 | 경력 500 | 문서 재생성 | 생성기 커밋 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['pid']} | {r['plan_attempts']} | {r['seed_round']} | {r['failed_runs']} | {r['facts_cap_hits']} | "
          f"{r['facts_late_attempts']} | {r['analysis_retries']} | {r['career_500']} | {r['doc_regen']} | {r['commit']} |")
anyretry = [r for r in rows if (r["plan_attempts"] or 1) > 1 or r["failed_runs"] or r["facts_cap_hits"] or r["doc_regen"]]
print(f"\n설계서·문서 단위로 다시 뽑은 페르소나: {len(anyretry)}/{len(rows)} = {len(anyretry)/len(rows):.0%}: {[r['pid'] for r in anyretry]}")
