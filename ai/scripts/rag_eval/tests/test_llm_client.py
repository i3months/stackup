"""llm_client 단위 테스트 — 네트워크 없이 가짜 클라이언트로 원장·예산·비용 계산을 확인한다.

    cd ai/scripts/rag_eval && python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_client  # noqa: E402
from llm_client import (  # noqa: E402
    PRICE_VERSION,
    BudgetExceeded,
    LabClient,
    Usage,
    cost_usd,
)

LEDGER_FIELDS = {
    "ts", "task_id", "run_id", "model", "batch", "request_id", "batch_id", "custom_id",
    "input_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens", "cache_read_tokens",
    "output_tokens", "stop_reason", "price_version", "cost_usd",
}


def _usage(inp=1000, out=200, cw5=0, cw1=0, cr=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cw5 + cw1,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=cw5, ephemeral_1h_input_tokens=cw1
        ),
        cache_read_input_tokens=cr,
    )


class _FakeMessages:
    def __init__(self, usage):
        self.usage = usage
        self.calls: list[dict] = []
        self.batches = _FakeBatches(self)

    def create(self, **params):
        self.calls.append(params)
        msg = SimpleNamespace(
            model=params["model"],
            usage=self.usage,
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"ok": true}')],
        )
        msg._request_id = "req_fake"
        return msg


class _FakeBatches:
    def __init__(self, parent):
        self.parent = parent
        self.created: list = []

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_fake")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        reqs = self.created[-1]
        for r in reqs:
            msg = SimpleNamespace(
                model=r["params"]["model"], usage=self.parent.usage, stop_reason="end_turn",
                content=[],
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(type="succeeded", message=msg),
            )


def _lab(tmp_path, budget=1.0, usage=None):
    fake = SimpleNamespace(messages=_FakeMessages(usage or _usage()))
    return LabClient(task_id="T-test", budget_usd=budget, run_id="r1", lab_dir=tmp_path, client=fake)


def _ledger(tmp_path):
    p = tmp_path / "results" / "_ledger.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_cost_hand_calculated():
    # Haiku: 1000 in × 0.10 + 200 out × 0.50 + 400 cw5m × 0.125 + 100 cw1h × 0.20 + 300 cr × 0.01 (per MTok)
    u = Usage(input_tokens=1000, output_tokens=200, cache_write_5m=400, cache_write_1h=100, cache_read=300)
    expected = (100 + 100 + 50 + 20 + 3) / 1_000_000
    assert cost_usd("claude-haiku-5-5", u, batch=False) == pytest.approx(expected)
    assert cost_usd("claude-haiku-5-5", u, batch=True) == pytest.approx(expected / 2)


# 공식 가격표(https://platform.claude.com/docs/en/about-claude/pricing, 2026-10-14)를 그대로 옮긴 기대값.
# (입력, 5분 캐시 쓰기, 1시간 캐시 쓰기, 캐시 읽기, 출력) $/MTok — 표에 적힌 값 자체를 쓴다(배율로 다시 계산하지 않음).
OFFICIAL = {
    "claude-haiku-5-5": (0.10, 0.125, 0.20, 0.01, 0.50),
    "claude-sonnet-5-5": (2.00, 2.50, 4.00, 0.10, 10.00),
    "claude-opus-5-5": (4.00, 5.00, 8.00, 0.20, 20.00),
}
HAIKU_OVER_100K = (0.50, 0.625, 1.00, 0.05, 2.50)


@pytest.mark.parametrize("model", sorted(OFFICIAL))
@pytest.mark.parametrize("field", range(5))
@pytest.mark.parametrize("batch", [False, True])
def test_unit_price_matches_official_table(model, field, batch):
    # 토큰 종류 하나만 n 개 → 비용 = 단가 × n / 1M (Batch 는 절반).
    # 프롬프트 쪽은 Haiku 기본 단가표 범위(≤ 100K)에 들도록 100K, 출력은 프롬프트 길이에 안 들어가 1M.
    kw = ["input_tokens", "cache_write_5m", "cache_write_1h", "cache_read", "output_tokens"][field]
    n = 1_000_000 if field == 4 else 100_000
    u = Usage(**{kw: n})
    expected = OFFICIAL[model][field] * n / 1e6 * (0.5 if batch else 1.0)
    assert cost_usd(model, u, batch=batch) == pytest.approx(expected)


@pytest.mark.parametrize("field", range(5))
def test_haiku_over_100k_table(field):
    # 프롬프트 길이 = 입력 + 캐시 쓰기 + 캐시 읽기. 120K 프롬프트 중 해당 종류 100K + 입력 20K 로 구성.
    kw = ["input_tokens", "cache_write_5m", "cache_write_1h", "cache_read", "output_tokens"][field]
    u = Usage(input_tokens=120_000) if field in (0, 4) else Usage(input_tokens=20_000, **{kw: 100_000})
    if field == 4:
        u.output_tokens = 1_000_000
    p = HAIKU_OVER_100K
    expected = (u.input_tokens * p[0] + u.cache_write_5m * p[1] + u.cache_write_1h * p[2]
                + u.cache_read * p[3] + u.output_tokens * p[4]) / 1e6
    assert u.prompt_tokens > 100_000
    assert cost_usd("claude-haiku-5-5", u, batch=False) == pytest.approx(expected)


def test_haiku_threshold_counts_cache_tokens():
    # 입력만으로는 10만 이하지만 캐시 읽기까지 합치면 초과 → 높은 단가표
    under = Usage(input_tokens=100_000)
    over = Usage(input_tokens=60_000, cache_read=40_001)
    assert cost_usd("claude-haiku-5-5", under, batch=False) == pytest.approx(100_000 * 0.10 / 1e6)
    assert cost_usd("claude-haiku-5-5", over, batch=False) == pytest.approx(
        (60_000 * 0.50 + 40_001 * 0.05) / 1e6)


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", Usage(), batch=False)


def test_call_writes_all_ledger_fields(tmp_path):
    lab = _lab(tmp_path)
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[{"role": "user", "content": "x"}])
    rows = _ledger(tmp_path)
    assert len(rows) == 1
    rec = rows[0]
    assert LEDGER_FIELDS <= rec.keys()
    assert rec["task_id"] == "T-test" and rec["run_id"] == "r1"
    assert rec["batch"] is False and rec["request_id"] == "req_fake"
    assert rec["batch_id"] is None and rec["custom_id"] is None
    assert rec["price_version"] == PRICE_VERSION
    assert rec["stop_reason"] == "end_turn"


def test_json_schema_merged_with_effort(tmp_path):
    lab = _lab(tmp_path)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    msg = lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[],
                   output_config={"effort": "low"}, json_schema=schema)
    sent = lab.client.messages.calls[-1]
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert "json_schema" not in sent
    assert llm_client.json_of(msg) == {"ok": True}


def test_budget_stops_at_80_percent(tmp_path):
    # 1회 비용 = 1000×0.10 + 200×0.50 = 200 / 1e6 = $0.0002. 상한 $0.001 → 80% = $0.0008 → 4회째 이후 중단.
    lab = _lab(tmp_path, budget=0.001)
    n_ok = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
            n_ok += 1
    assert n_ok == 4
    assert lab.spent_usd() == pytest.approx(0.0008)
    assert len(_ledger(tmp_path)) == 4  # 중단된 호출은 API 에 가지 않고 원장에도 없다


def test_budget_counts_estimate(tmp_path):
    lab = _lab(tmp_path, budget=1.0)
    with pytest.raises(BudgetExceeded):
        lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[], estimate_usd=0.80)
    assert lab.client.messages.calls == []


def test_budget_is_per_task(tmp_path):
    other = _lab(tmp_path, budget=0.001)
    other.task_id = "T-other"
    for _ in range(3):
        other.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    lab = _lab(tmp_path, budget=0.001)
    assert lab.spent_usd() == 0.0
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])


def test_budget_required(tmp_path):
    with pytest.raises(ValueError):
        LabClient(task_id="x", budget_usd=0, lab_dir=tmp_path, client=object())


def test_batch_ledger_rows_and_idempotent_collect(tmp_path):
    lab = _lab(tmp_path)
    bid = lab.submit_batch(
        [("a", {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": []}),
         ("b", {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": []})]
    )
    lab.wait_batch(bid, poll_sec=0)
    res = lab.collect_batch(bid)
    assert set(res) == {"a", "b"}
    lab.collect_batch(bid)  # 두 번 회수해도 중복 기록 없음
    rows = _ledger(tmp_path)
    assert len(rows) == 2
    for rec in rows:
        assert LEDGER_FIELDS <= rec.keys()
        assert rec["batch"] is True and rec["batch_id"] == bid and rec["request_id"] is None
        assert rec["cost_usd"] == pytest.approx(0.0001)  # 50% 할인
    assert {r["custom_id"] for r in rows} == {"a", "b"}


def test_ledger_read_incrementally(tmp_path):
    lab = _lab(tmp_path)
    for _ in range(3):
        lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert lab.spent_usd() == pytest.approx(0.0006)
    off = lab._offset  # noqa: SLF001
    assert off == (tmp_path / "results" / "_ledger.jsonl").stat().st_size
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    lab.spent_usd()
    assert lab._offset > off  # 새 줄만 읽었다  # noqa: SLF001
    assert lab.spent_usd() == pytest.approx(0.0008)


def test_other_process_appends_are_seen(tmp_path):
    lab = _lab(tmp_path)
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert lab.spent_usd() == pytest.approx(0.0002)
    other = _lab(tmp_path)  # 다른 프로세스 역할
    other.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert lab.spent_usd() == pytest.approx(0.0004)


def test_partial_last_line_is_deferred(tmp_path):
    lab = _lab(tmp_path)
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    path = tmp_path / "results" / "_ledger.jsonl"
    line = json.dumps({"task_id": "T-test", "cost_usd": 1.0})
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line[:10])  # 다른 프로세스가 쓰는 중
    assert lab.spent_usd() == pytest.approx(0.0002)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line[10:] + "\n")
    assert lab.spent_usd() == pytest.approx(1.0002)


def test_ledger_replaced_is_reread(tmp_path):
    lab = _lab(tmp_path)
    for _ in range(2):
        lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    path = tmp_path / "results" / "_ledger.jsonl"
    path.write_text(json.dumps({"task_id": "T-test", "cost_usd": 0.0001}) + "\n", encoding="utf-8")
    assert lab.spent_usd() == pytest.approx(0.0001)


def test_ledger_records_prompt_tokens(tmp_path):
    lab = _lab(tmp_path, usage=_usage(inp=1000, cw5=10, cw1=20, cr=30))
    lab.call(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert _ledger(tmp_path)[0]["prompt_tokens"] == 1060


def test_usage_without_ttl_breakdown_counts_as_5m():
    u = Usage.from_sdk(SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=50,
                                       cache_creation=None, cache_read_input_tokens=0))
    assert (u.cache_write_5m, u.cache_write_1h) == (50, 0)
