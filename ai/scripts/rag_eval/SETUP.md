# rag_eval — 실험 환경 재현 (sudo 없음)

RAG 논문 실험(stackup-lab) 하네스의 실행 환경이다. **운영 서버와 무관**하며 운영 `.env`·키를 쓰지 않는다.
모든 설치는 사용자 홈 아래(`~/.local`, `~/rag-lab`)에만 한다.

## 0. 확인된 환경 (T01, 2026-10)

| 항목 | 값 |
|---|---|
| OS | Windows 11 + WSL2 Ubuntu 26.04, sudo 없음 |
| GPU | RTX 4070 12GB, 드라이버 591.86, 전력 제한 200W |
| RAM | WSL 48GB (`%UserProfile%\.wslconfig` 의 `memory=48GB`, 사람이 설정) |
| Python | 3.13.16 (uv 관리) — torch 2.14.1+cu130, transformers 5.19.0, sentence-transformers 6.1.0, anthropic 1.13.0 |
| 서버 | PostgreSQL 17.11 + pgvector 0.8.1, llama.cpp conda-forge `10751 cuda130` — 모두 micromamba |

## 1. 도구 (uv, micromamba)

```bash
mkdir -p ~/rag-lab ~/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh            # → ~/.local/bin/uv (0.13.0)
# micromamba: bzip2 가 없으면 tar -xj 가 실패한다. 그때는 파이썬 tarfile 로 푼다.
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/mm.tar.bz2
python3 -c "import tarfile; tarfile.open('/tmp/mm.tar.bz2','r:bz2').extract('bin/micromamba', '$HOME/.local')"
~/.local/bin/micromamba --version                           # 2.9.0
```

## 2. PostgreSQL + pgvector + llama.cpp (conda-forge)

```bash
export MAMBA_ROOT_PREFIX=~/rag-lab/mamba
~/.local/bin/micromamba create -y -n raglab -c conda-forge "llama.cpp=*=cuda130*" postgresql=17 pgvector
E=~/rag-lab/mamba/envs/raglab
$E/bin/llama-server --version          # CUDA0 = RTX 4070 이 잡히는지 확인
```

DB 초기화 (포트 55432, 소켓 /tmp, trust 인증 — 로컬 전용):

```bash
$E/bin/initdb -D ~/rag-lab/pg/data -U postgres -A trust -E UTF8 --locale=C.UTF-8
$E/bin/pg_ctl -D ~/rag-lab/pg/data -o "-p 55432 -k /tmp" -l ~/rag-lab/pg/log.txt start
$E/bin/createdb -h /tmp -p 55432 -U postgres raglab
# 운영 Flyway 마이그레이션을 번호 순서로 그대로 적용 (V1~V36)
M=<stackup>/backend/src/main/resources/db/migration
for f in $(ls $M | sort -t_ -k1.2 -n); do
  $E/bin/psql -h /tmp -p 55432 -U postgres -d raglab -v ON_ERROR_STOP=1 -q -f $M/$f || break
done
```

WSL 재시작 뒤에는 `pg_ctl ... start` 만 다시 하면 된다. 실험 테이블 `document_embeddings_d2560` 은
`smoke.py emb4` 가 처음 실행될 때 만든다 (HNSW 없음 — SPEC §2).

## 3. 파이썬 환경

```bash
cd ~/rag-lab && ~/.local/bin/uv venv --python 3.13 .venv && . .venv/bin/activate
cd <stackup>/ai
~/.local/bin/uv pip install -e . anthropic sentence-transformers kiwipiepy "psycopg[binary]" numpy pandas scipy pytest
python -c "import torch; print(torch.cuda.is_available())"   # True
```

`-e .` 로 운영 `ai_server` 패키지를 그대로 import 한다 (SPEC §10-2). 운영 코드는 고치지 않는다.

## 4. 모델 (P-008)

`~/rag-lab/models/` 에 고정 리비전으로 받는다. HF 모델은 `huggingface_hub.snapshot_download(repo, revision=<sha>,
local_dir=~/rag-lab/models/hf/<repo>)`, GGUF 는 파일 하나만 받는다. 리비전·sha256 은
`~/rag-lab/models/{revisions.txt, sha256.txt}` 에 적고 아래로 `stackup-lab/results/E0-env/models.json` 을 만든다.

```bash
python common.py models
```

> `hf download` 에 `--exclude` 를 여러 개 주면 잘못 해석돼 빈 디렉터리가 생긴 적이 있다 — `snapshot_download` 의
> `ignore_patterns` 를 쓴다. 하네스는 `HF_HUB_OFFLINE=1` 로 로컬 사본만 읽는다.

## 5. Anthropic 키

`ANTHROPIC_API_KEY` 는 `~/.anthropic_env` 에 있고 `~/.profile` 이 읽는다. 하네스는 환경변수로만 읽고,
모든 호출은 `llm_client.LabClient(task_id=..., budget_usd=...)` 를 거친다 (원장 `stackup-lab/results/_ledger.jsonl`,
상한 80% 에서 중단).

## 6. 실행

```bash
cd <stackup>/ai/scripts/rag_eval
python -m pytest tests -q                         # 원장·예산 단위 테스트 (네트워크 없음)

# 로컬 생성 모델 서버 (슬롯 1개 -np 1, ctx 16384)
python servers/llama_server.py start M-S          # :8081, 전부 GPU
python servers/llama_server.py start M-M          # :8082, --n-cpu-moe (기본값은 스크립트 상수, T01 실측)
python servers/llama_server.py bench-moe --ns 30,26,22,20,18   # → results/E0-env/m-m-offload.json
python servers/llama_server.py stop M-M

# E0 스모크 6종 (결과: stackup-lab/results/E0-env/smoke/)
python smoke.py chunk
python smoke.py emb1
python smoke.py emb4
python smoke.py rerank
python smoke.py generate --profile M-S
python smoke.py generate --profile M-M
python smoke.py claude                            # 유료: Haiku 일반 1 + Batch 1
```

## 7. 주의

- `pkill -f llama-server` 는 그 명령을 부른 셸까지 죽인다. `servers/llama_server.py stop` 을 쓴다.
- WSL 에서 `/mnt/c` 의 저장소를 git 으로 읽으면 소유자·줄바꿈 차이로 거부되거나 전부 수정됨으로 보인다.
  하네스는 호출마다 `-c safe.directory=… -c core.autocrlf=true -c core.filemode=false` 를 붙인다 (전역 설정 무변경).
- 100MB 넘는 파일(가중치·DB 덤프·캐시)은 커밋하지 않는다. `~/rag-lab/` 에 두고 출처·해시만 기록한다.
