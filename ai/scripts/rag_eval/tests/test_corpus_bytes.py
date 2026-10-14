"""E1a 코퍼스를 바이트 그대로 읽는지 (T03 보고서 §5-5, lab 9e4ce70 `results/** -text`).

레포 원문(운영 추출기 출력)에는 CRLF 파일이 있고, 사실 오프셋은 그 원문 기준이다. read_text 나 git 의
autocrlf 가 \\r 을 지우면 오프셋이 밀린다. lab 결과물이 없는 환경에서는 건너뛴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import results_dir  # noqa: E402

CORPUS = results_dir("E1-corpus")
P29_REPO = CORPUS / "docs" / "P29" / "P29-repo.md"
P29_REPO_CR = 27  # requirements.txt 등의 CRLF (T03 생성 시점 원문)

pytestmark = pytest.mark.skipif(not (CORPUS / "facts.jsonl").exists(), reason="E1a 코퍼스 없음")


def test_p29_repo_keeps_cr_on_disk():
    assert P29_REPO.read_bytes().count(b"\r") == P29_REPO_CR


def test_load_corpus_keeps_cr():
    from index import load_corpus  # noqa: PLC0415

    doc = next(d for d in load_corpus("E1") if d.doc_id == "P29-repo")
    assert doc.raw_text.count("\r") == P29_REPO_CR


def test_all_evidence_offsets_match_raw_text():
    from index import load_corpus  # noqa: PLC0415

    raw = {d.doc_id: d.raw_text for d in load_corpus("E1")}
    bad = []
    with (CORPUS / "facts.jsonl").open(encoding="utf-8") as fh:
        for ln in fh:
            f = json.loads(ln)
            if raw[f["doc_id"]][f["char_start"]:f["char_end"]] != f["evidence_text"]:
                bad.append(f["fact_id"])
    assert bad == []
