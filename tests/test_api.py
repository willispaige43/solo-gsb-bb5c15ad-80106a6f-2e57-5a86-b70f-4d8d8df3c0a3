"""API 端到端测试（需要 fastapi/httpx，容器内或安装依赖后运行）。"""
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import database
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 每个测试用例使用独立的临时数据库文件
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "test.db"))
    database.init_db()
    with TestClient(app) as c:
        yield c


def _setup_port(client):
    r = client.post("/ports", json={"name": "示例港", "reference_depth": 4.5})
    assert r.status_code == 201
    port_id = r.json()["id"]
    r = client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={
            "points": [
                {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 0.5},
                {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
                {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 0.5},
            ]
        },
    )
    assert r.status_code == 200
    return port_id


def test_full_flow_and_correction_recompute(client):
    port_id = _setup_port(client)

    # 吃水 5.0 + 富余 0.5 = 要求水深 5.5；基准 4.5 → 需潮高 >= 1.0
    # 水深 5.0→7.5→5.0，达标窗口 [00:36, 05:24]，通航 60 分钟
    r = client.post(
        f"/ports/{port_id}/windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["required_depth"] == 5.5
    assert body["earliest_departure"] == "2026-09-24T00:36:00+00:00"
    assert body["windows"] == [
        {"start": "2026-09-24T00:36:00+00:00", "end": "2026-09-24T04:24:00+00:00"}
    ]

    # 更正高潮预报点 3.0 → 1.0：水深峰值仅 5.5，只剩零长窗口，不足 60 分钟
    r = client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}]},
    )
    assert r.status_code == 200

    r = client.post(
        f"/ports/{port_id}/windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    )
    body = r.json()
    assert body["windows"] == []
    assert body["earliest_departure"] is None


def test_validation_rejects_non_positive_duration(client):
    port_id = _setup_port(client)
    for bad in (0, -30):
        r = client.post(
            f"/ports/{port_id}/windows",
            json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": bad},
        )
        assert r.status_code == 422


def test_unknown_port_404(client):
    assert client.post(
        "/ports/999/windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).status_code == 404


def test_no_forecasts_returns_empty(client):
    r = client.post("/ports", json={"name": "空港", "reference_depth": 10.0})
    port_id = r.json()["id"]
    r = client.post(
        f"/ports/{port_id}/windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    )
    assert r.status_code == 200
    assert r.json()["windows"] == []
    assert r.json()["earliest_departure"] is None


# ---------------------------------------------------------------------------
# 单向航道预约
# ---------------------------------------------------------------------------

# 示例港：基准 4.5，潮高 0.5→3.0→0.5（00:00/03:00/06:00）
# 吃水 5.0 + 富余 0.5 → 达标窗口 [00:36, 05:24]，60 分钟通航
# → 可出发时段 [00:36, 04:24]
RESERVATION_BODY = {
    "ship_name": "远洋号",
    "draft": 5.0,
    "safety_margin": 0.5,
    "duration_minutes": 60,
    "allowed_departure_from": "2026-09-24T00:00:00Z",
    "allowed_departure_to": "2026-09-24T04:24:00Z",
}


def test_create_reservation_schedules_earliest(client):
    port_id = _setup_port(client)
    r = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "active"
    assert body["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert body["departure_end"] == "2026-09-24T01:36:00+00:00"
    assert body["allowed_departure_from"] == "2026-09-24T00:00:00+00:00"
    assert body["id"] > 0


def test_reservation_within_range_starts_no_earlier_than_range(client):
    port_id = _setup_port(client)
    body = {
        **RESERVATION_BODY,
        "allowed_departure_from": "2026-09-24T02:00:00Z",
    }
    r = client.post(f"/ports/{port_id}/reservations", json=body)
    assert r.status_code == 201
    assert r.json()["departure_start"] == "2026-09-24T02:00:00+00:00"


def test_reservations_do_not_overlap_but_may_touch(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r1.status_code == 201
    # 第二个预约最早只能紧随其后：01:36 出发（首尾相接）
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r2.status_code == 201
    assert r2.json()["departure_start"] == "2026-09-24T01:36:00+00:00"
    assert r2.json()["departure_end"] == "2026-09-24T02:36:00+00:00"


def test_reservation_conflict_returns_409_and_writes_nothing(client):
    port_id = _setup_port(client)
    # 只允许在 04:00 前出发：第一条 00:36–01:36，第二条无可行时刻
    body = {
        **RESERVATION_BODY,
        "allowed_departure_to": "2026-09-24T00:50:00Z",
    }
    assert client.post(f"/ports/{port_id}/reservations", json=body).status_code == 201
    r = client.post(f"/ports/{port_id}/reservations", json=body)
    assert r.status_code == 409
    # 409 不写入：有效预约仍只有一条
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 1


def test_list_reservations_only_active_and_port_scoped(client):
    port_id = _setup_port(client)
    other = client.post("/ports", json={"name": "其他港", "reference_depth": 4.5})
    other_id = other.json()["id"]
    client.put(
        f"/ports/{other_id}/tide-forecasts",
        json={"points": [
            {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 0.5},
            {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
            {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 0.5},
        ]},
    )

    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid = r1.json()["id"]
    client.post(f"/ports/{other_id}/reservations", json=RESERVATION_BODY)

    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert [r["id"] for r in listed] == [rid]

    # 取消后不再出现在有效列表，航道占用随即释放
    assert client.delete(f"/reservations/{rid}").status_code == 200
    assert client.get(f"/ports/{port_id}/reservations").json() == []

    # 已取消再取消 → 409；不存在 → 404
    assert client.delete(f"/reservations/{rid}").status_code == 409
    assert client.delete("/reservations/999").status_code == 404


def test_cancelled_reservation_frees_the_channel(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid = r1.json()["id"]
    # 第二条被挤到 01:36
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r2.json()["departure_start"] == "2026-09-24T01:36:00+00:00"
    # 取消第一条后新建预约可以回到最早的 00:36
    client.delete(f"/reservations/{rid}")
    r3 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r3.json()["departure_start"] == "2026-09-24T00:36:00+00:00"


def test_forecast_correction_does_not_touch_existing_reservation(client):
    port_id = _setup_port(client)
    r = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid = r.json()["id"]
    # 预报更正为全程水深不足：已有预约保持有效、时刻不变
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}]},
    )
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 1
    assert listed[0]["id"] == rid
    assert listed[0]["departure_start"] == "2026-09-24T00:36:00+00:00"
    # 但新预约按更正后的预报无窗口 → 409
    assert client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY).status_code == 409


def test_concurrent_reservations_never_double_book(client):
    port_id = _setup_port(client)
    results: list = []
    errors: list = []

    def worker():
        try:
            r = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
            results.append((r.status_code, r.json() if r.status_code == 201 else None))
        except Exception as exc:  # pragma: no cover - 并发不应抛出异常
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    created = [body for status, body in results if status == 201]
    conflicts = [status for status, _ in results if status == 409]
    # 可出发时段 [00:36, 04:24] 内恰好容纳 4 条 60 分钟预约（首尾相接）
    assert len(created) + len(conflicts) == 8
    assert len(created) == 4

    intervals = sorted(
        (b["departure_start"], b["departure_end"]) for b in created
    )
    # 两两不得重叠（首尾相接允许）
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        assert e1 <= s2


def test_reservation_validation_errors(client):
    port_id = _setup_port(client)
    # 通航时长非正 → 422
    bad_duration = {**RESERVATION_BODY, "duration_minutes": 0}
    assert client.post(f"/ports/{port_id}/reservations", json=bad_duration).status_code == 422
    # 允许出发范围起止颠倒 → 422
    bad_range = {
        **RESERVATION_BODY,
        "allowed_departure_from": "2026-09-24T04:24:00Z",
        "allowed_departure_to": "2026-09-24T00:00:00Z",
    }
    assert client.post(f"/ports/{port_id}/reservations", json=bad_range).status_code == 422
    # 船名为空 → 422
    bad_name = {**RESERVATION_BODY, "ship_name": ""}
    assert client.post(f"/ports/{port_id}/reservations", json=bad_name).status_code == 422
    # 未知港口 → 404
    assert client.post("/ports/999/reservations", json=RESERVATION_BODY).status_code == 404


def test_reservation_naive_datetime_treated_as_utc(client):
    port_id = _setup_port(client)
    body = {
        **RESERVATION_BODY,
        "allowed_departure_from": "2026-09-24T00:00:00",   # 无时区 → UTC
        "allowed_departure_to": "2026-09-24T04:24:00",
    }
    r = client.post(f"/ports/{port_id}/reservations", json=body)
    assert r.status_code == 201
    assert r.json()["departure_start"] == "2026-09-24T00:36:00+00:00"


# ---------------------------------------------------------------------------
# 预约改期 PATCH /reservations/{id}/schedule
# ---------------------------------------------------------------------------


def test_reschedule_success_keeps_identity_and_updates_schedule(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1, rid2 = r1.json()["id"], r2.json()["id"]
    assert r2.json()["departure_start"] == "2026-09-24T01:36:00+00:00"

    # 第二条改期：新允许范围 03:00–04:24 → 最早 03:00 出发
    r = client.patch(
        f"/reservations/{rid2}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T03:00:00Z",
            "allowed_departure_to": "2026-09-24T04:24:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    # 预约 ID、船名、创建时间与船舶参数保持不变
    assert body["id"] == rid2
    assert body["ship_name"] == "远洋号"
    assert body["created_at"] == r2.json()["created_at"]
    assert body["status"] == "active"
    assert body["canceled_at"] is None
    assert body["draft"] == 5.0
    assert body["safety_margin"] == 0.5
    assert body["duration_minutes"] == 60.0
    # 允许范围与通航起止已更新
    assert body["allowed_departure_from"] == "2026-09-24T03:00:00+00:00"
    assert body["allowed_departure_to"] == "2026-09-24T04:24:00+00:00"
    assert body["departure_start"] == "2026-09-24T03:00:00+00:00"
    assert body["departure_end"] == "2026-09-24T04:00:00+00:00"

    # 有效预约列表同样反映新排程（按出发时刻升序）
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert [r["id"] for r in listed] == [rid1, rid2]
    assert listed[1]["departure_start"] == "2026-09-24T03:00:00+00:00"


def test_reschedule_excludes_own_original_occupation(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1 = r1.json()["id"]
    assert r2.json()["departure_start"] == "2026-09-24T01:36:00+00:00"

    # r1 原占用 00:36–01:36，r2 占用 01:36–02:36；r1 改期排除自身原占用，
    # 最早仍是 00:36（与 r2 首尾相接允许），不会因撞上自己而被挤走
    r = client.patch(
        f"/reservations/{rid1}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T00:00:00Z",
            "allowed_departure_to": "2026-09-24T04:24:00Z",
        },
    )
    assert r.status_code == 200
    assert r.json()["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert r.json()["departure_end"] == "2026-09-24T01:36:00+00:00"


def test_reschedule_picks_earliest_slot_after_others(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1 = r1.json()["id"]
    # r2 占用 01:36–02:36；r1 改到 [01:00, 04:24]：01:00–02:36 出发都会与
    # r2 重叠，最早可紧随 r2 于 02:36 出发（首尾相接）
    r = client.patch(
        f"/reservations/{rid1}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T01:00:00Z",
            "allowed_departure_to": "2026-09-24T04:24:00Z",
        },
    )
    assert r.status_code == 200
    assert r.json()["departure_start"] == "2026-09-24T02:36:00+00:00"
    assert r.json()["departure_end"] == "2026-09-24T03:36:00+00:00"


def test_reschedule_409_keeps_original_schedule_and_range(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1 = r1.json()["id"]
    # r2 占用 01:36–02:36；新范围 [00:50, 01:00] 内任何出发都会撞上 r2 → 409
    r = client.patch(
        f"/reservations/{rid1}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T00:50:00Z",
            "allowed_departure_to": "2026-09-24T01:00:00Z",
        },
    )
    assert r.status_code == 409
    # 409 保留原排程与允许范围
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 2
    mine = next(r for r in listed if r["id"] == rid1)
    assert mine["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert mine["departure_end"] == "2026-09-24T01:36:00+00:00"
    assert mine["allowed_departure_from"] == "2026-09-24T00:00:00+00:00"
    assert mine["allowed_departure_to"] == "2026-09-24T04:24:00+00:00"


def test_reschedule_409_when_no_tide_window_in_new_range(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    # 新范围完全在预报覆盖时段之外 → 无可行时刻，原排程不变
    r = client.patch(
        f"/reservations/{rid}/schedule",
        json={
            "allowed_departure_from": "2026-09-25T00:00:00Z",
            "allowed_departure_to": "2026-09-25T04:00:00Z",
        },
    )
    assert r.status_code == 409
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert listed[0]["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert listed[0]["allowed_departure_from"] == "2026-09-24T00:00:00+00:00"


def test_reschedule_uses_latest_forecast(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    # 更正预报：潮高 1.0→3.0→1.0（00:00/03:00/06:00），水深 5.5→7.5→5.5，
    # 所需水深 5.5 → 全程达标，可出发时段变为 [00:00, 05:00]
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [
            {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 1.0},
            {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
            {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 1.0},
        ]},
    )
    # 按请求时最新预报重算：最早出发从 00:36 提前到 00:00
    r = client.patch(
        f"/reservations/{rid}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T00:00:00Z",
            "allowed_departure_to": "2026-09-24T05:00:00Z",
        },
    )
    assert r.status_code == 200
    assert r.json()["departure_start"] == "2026-09-24T00:00:00+00:00"
    assert r.json()["departure_end"] == "2026-09-24T01:00:00+00:00"


def test_reschedule_cancelled_409_and_unknown_404(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    body = {
        "allowed_departure_from": "2026-09-24T00:00:00Z",
        "allowed_departure_to": "2026-09-24T04:24:00Z",
    }
    assert client.delete(f"/reservations/{rid}").status_code == 200
    # 已取消预约不能改期
    assert client.patch(f"/reservations/{rid}/schedule", json=body).status_code == 409
    # 不存在的预约
    assert client.patch("/reservations/999/schedule", json=body).status_code == 404


def test_reschedule_validation_and_naive_datetime(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    # 允许出发范围起止颠倒 → 422
    bad_range = {
        "allowed_departure_from": "2026-09-24T04:24:00Z",
        "allowed_departure_to": "2026-09-24T00:00:00Z",
    }
    assert client.patch(f"/reservations/{rid}/schedule", json=bad_range).status_code == 422
    # 缺少必填字段 → 422
    assert client.patch(f"/reservations/{rid}/schedule", json={}).status_code == 422
    # 无时区时刻按 UTC 处理
    r = client.patch(
        f"/reservations/{rid}/schedule",
        json={
            "allowed_departure_from": "2026-09-24T02:00:00",
            "allowed_departure_to": "2026-09-24T04:24:00",
        },
    )
    assert r.status_code == 200
    assert r.json()["allowed_departure_from"] == "2026-09-24T02:00:00+00:00"
    assert r.json()["departure_start"] == "2026-09-24T02:00:00+00:00"


def test_concurrent_reschedule_never_double_book(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1, rid2 = r1.json()["id"], r2.json()["id"]

    # 新范围 [03:00, 03:30] 只容得下一条 60 分钟预约：
    # 两条同时改期，恰好一条成功，另一条 409 并保留原排程
    body = {
        "allowed_departure_from": "2026-09-24T03:00:00Z",
        "allowed_departure_to": "2026-09-24T03:30:00Z",
    }
    results: list = []
    errors: list = []

    def worker(rid):
        try:
            r = client.patch(f"/reservations/{rid}/schedule", json=body)
            results.append(r.status_code)
        except Exception as exc:  # pragma: no cover - 并发不应抛出异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(rid,)) for rid in (rid1, rid2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(results) == [200, 409]
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 2
    intervals = sorted((r["departure_start"], r["departure_end"]) for r in listed)
    # 两两不得重叠（首尾相接允许）
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        assert e1 <= s2


# ---------------------------------------------------------------------------
# 预约水深复核 GET /reservations/{id}/depth-audit
# ---------------------------------------------------------------------------


def _create_reservation(client, port_id):
    r = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r.status_code == 201
    return r.json()["id"]


def test_depth_audit_sufficient_when_forecast_unchanged(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)

    r = client.get(f"/reservations/{rid}/depth-audit")
    assert r.status_code == 200
    body = r.json()
    assert body["reservation_id"] == rid
    assert body["port_id"] == port_id
    assert body["ship_name"] == "远洋号"
    assert body["status"] == "active"
    assert body["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert body["departure_end"] == "2026-09-24T01:36:00+00:00"
    assert body["required_depth"] == 5.5
    assert body["coverage_complete"] is True
    assert body["uncovered"] == []
    assert body["deficits"] == []
    assert body["sufficient"] is True


def test_depth_audit_reflects_latest_forecast_and_stays_read_only(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)

    # 更正高潮预报点 3.0 → 1.0：通航区间 [00:36, 01:36] 内水深 5.1→5.27，
    # 全程低于所需 5.5，最大欠深在 00:36（5.5 − 5.1 = 0.4）
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}]},
    )
    body = client.get(f"/reservations/{rid}/depth-audit").json()
    assert body["coverage_complete"] is True
    assert body["uncovered"] == []
    assert body["deficits"] == [
        {
            "start": "2026-09-24T00:36:00+00:00",
            "end": "2026-09-24T01:36:00+00:00",
            "max_deficit": 0.4,
        }
    ]
    assert body["sufficient"] is False

    # 复核只读：预约的排定时刻与状态不受影响
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 1
    assert listed[0]["id"] == rid
    assert listed[0]["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert listed[0]["departure_end"] == "2026-09-24T01:36:00+00:00"
    assert listed[0]["status"] == "active"


def test_depth_audit_reports_uncovered_when_forecast_range_shrinks(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)

    # 模拟预报覆盖收缩（如外部系统清理旧预报点）：删掉 03:00/06:00 点，
    # 再补一个 01:00 点 → 预报只覆盖 [00:00, 01:00]，通航到 01:36
    conn = sqlite3.connect(database.DATABASE_PATH)
    conn.execute(
        "DELETE FROM tide_forecasts WHERE forecast_time IN (?, ?)",
        ("2026-09-24T03:00:00+00:00", "2026-09-24T06:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T01:00:00Z", "tide_height": 1.0}]},
    )

    body = client.get(f"/reservations/{rid}/depth-audit").json()
    # 覆盖不足：不得判为安全
    assert body["coverage_complete"] is False
    assert body["sufficient"] is False
    assert body["uncovered"] == [
        {"start": "2026-09-24T01:00:00+00:00", "end": "2026-09-24T01:36:00+00:00"}
    ]
    # 覆盖段 [00:36, 01:00] 内水深 5.3→5.5：00:36 起欠深，01:00 恰好达标
    assert body["deficits"] == [
        {
            "start": "2026-09-24T00:36:00+00:00",
            "end": "2026-09-24T01:00:00+00:00",
            "max_deficit": 0.2,
        }
    ]


def test_depth_audit_cancelled_reservation_still_auditable(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    assert client.delete(f"/reservations/{rid}").status_code == 200

    body = client.get(f"/reservations/{rid}/depth-audit").json()
    assert body["status"] == "cancelled"
    assert body["sufficient"] is True  # 预报未变，排定区间水深仍达标


def test_depth_audit_unknown_reservation_404(client):
    assert client.get("/reservations/999/depth-audit").status_code == 404


# ---------------------------------------------------------------------------
# 最大可通航吃水 POST /ports/{id}/max-draft
# ---------------------------------------------------------------------------

# 基准水深 4.0，潮高 1.0→3.0→1.0（00:00/02:00/04:00）：水深 5.0→7.0→5.0
def _setup_max_draft_port(client):
    r = client.post("/ports", json={"name": "吃水港", "reference_depth": 4.0})
    port_id = r.json()["id"]
    r = client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={
            "points": [
                {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 1.0},
                {"forecast_time": "2026-09-24T02:00:00Z", "tide_height": 3.0},
                {"forecast_time": "2026-09-24T04:00:00Z", "tide_height": 1.0},
            ]
        },
    )
    assert r.status_code == 200
    return port_id


MAX_DRAFT_BODY = {
    "safety_margin": 0.5,
    "duration_minutes": 60,
    "allowed_departure_from": "2026-09-24T00:00:00Z",
    "allowed_departure_to": "2026-09-24T04:00:00Z",
}


def test_max_draft_centered_window(client):
    port_id = _setup_max_draft_port(client)
    r = client.post(f"/ports/{port_id}/max-draft", json=MAX_DRAFT_BODY)
    assert r.status_code == 200
    body = r.json()
    # 最优通航区间 [01:30, 02:30]，最低水深 6.5 → 吃水 6.5 − 0.5 = 6.0
    assert body["max_draft"] == 6.0
    assert body["required_depth"] == 6.5
    assert body["departure_start"] == "2026-09-24T01:30:00+00:00"
    assert body["departure_end"] == "2026-09-24T02:30:00+00:00"
    assert body["safety_margin"] == 0.5
    assert body["duration_minutes"] == 60.0
    assert body["allowed_departure_from"] == "2026-09-24T00:00:00+00:00"
    assert body["allowed_departure_to"] == "2026-09-24T04:00:00+00:00"


def test_max_draft_flat_tide_zero_draft_allowed(client):
    # 平潮全程水深恒 5.0：最大吃水 4.5，所有时刻可行 → 返回允许范围起点
    r = client.post("/ports", json={"name": "平潮港", "reference_depth": 4.0})
    port_id = r.json()["id"]
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={
            "points": [
                {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 1.0},
                {"forecast_time": "2026-09-24T04:00:00Z", "tide_height": 1.0},
            ]
        },
    )
    body = client.post(f"/ports/{port_id}/max-draft", json=MAX_DRAFT_BODY).json()
    assert body["max_draft"] == 4.5
    assert body["required_depth"] == 5.0
    assert body["departure_start"] == "2026-09-24T00:00:00+00:00"
    assert body["departure_end"] == "2026-09-24T01:00:00+00:00"


def test_max_draft_rounds_down_to_centimeter_and_returns_earliest(client):
    port_id = _setup_max_draft_port(client)
    # 通航 70 分钟：最优区间 [01:25, 02:35]，最低水深 6.41666…
    # → 吃水 5.91666… 向下取整为 5.91；取整后最早可行出发为 01:24:36
    body = {**MAX_DRAFT_BODY, "duration_minutes": 70}
    r = client.post(f"/ports/{port_id}/max-draft", json=body)
    assert r.status_code == 200
    b = r.json()
    assert b["max_draft"] == 5.91
    assert b["required_depth"] == 6.41
    assert b["departure_start"] == "2026-09-24T01:24:36+00:00"
    assert b["departure_end"] == "2026-09-24T02:34:36+00:00"


def test_max_draft_allowed_range_clips_optimum(client):
    port_id = _setup_max_draft_port(client)
    # 只允许 01:00 前出发：最优 01:00 出发，[01:00, 02:00] 最低水深 6.0
    body = {**MAX_DRAFT_BODY, "allowed_departure_to": "2026-09-24T01:00:00Z"}
    r = client.post(f"/ports/{port_id}/max-draft", json=body)
    assert r.status_code == 200
    b = r.json()
    assert b["max_draft"] == 5.5
    assert b["required_depth"] == 6.0
    assert b["departure_start"] == "2026-09-24T01:00:00+00:00"
    assert b["departure_end"] == "2026-09-24T02:00:00+00:00"


def test_max_draft_recomputes_after_forecast_correction(client):
    port_id = _setup_max_draft_port(client)
    # 更正高潮点 3.0 → 2.0：水深 5→6→5，最优区间最低水深 5.75 → 吃水 5.25
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T02:00:00Z", "tide_height": 2.0}]},
    )
    r = client.post(f"/ports/{port_id}/max-draft", json=MAX_DRAFT_BODY)
    assert r.status_code == 200
    b = r.json()
    assert b["max_draft"] == 5.25
    assert b["required_depth"] == 5.75
    assert b["departure_start"] == "2026-09-24T01:30:00+00:00"
    assert b["departure_end"] == "2026-09-24T02:30:00+00:00"


def test_max_draft_409_when_range_outside_coverage(client):
    port_id = _setup_max_draft_port(client)
    # 允许范围完全在预报覆盖时段（00:00–04:00）之外：0 米吃水也不可行
    body = {
        **MAX_DRAFT_BODY,
        "allowed_departure_from": "2026-09-24T05:00:00Z",
        "allowed_departure_to": "2026-09-24T06:00:00Z",
    }
    r = client.post(f"/ports/{port_id}/max-draft", json=body)
    assert r.status_code == 409
    # 只读查询：不影响港口数据
    assert len(client.get(f"/ports/{port_id}/tide-forecasts").json()) == 3


def test_max_draft_409_when_duration_exceeds_coverage(client):
    port_id = _setup_max_draft_port(client)
    body = {**MAX_DRAFT_BODY, "duration_minutes": 300}  # 覆盖仅 240 分钟
    assert client.post(f"/ports/{port_id}/max-draft", json=body).status_code == 409


def test_max_draft_validation_and_404(client):
    port_id = _setup_max_draft_port(client)
    # 通航时长非正 → 422
    assert client.post(
        f"/ports/{port_id}/max-draft", json={**MAX_DRAFT_BODY, "duration_minutes": 0}
    ).status_code == 422
    # 安全富余为负 → 422
    assert client.post(
        f"/ports/{port_id}/max-draft", json={**MAX_DRAFT_BODY, "safety_margin": -0.1}
    ).status_code == 422
    # 允许出发范围起止颠倒 → 422
    bad_range = {
        **MAX_DRAFT_BODY,
        "allowed_departure_from": "2026-09-24T04:00:00Z",
        "allowed_departure_to": "2026-09-24T00:00:00Z",
    }
    assert client.post(f"/ports/{port_id}/max-draft", json=bad_range).status_code == 422
    # 未知港口 → 404
    assert client.post("/ports/999/max-draft", json=MAX_DRAFT_BODY).status_code == 404


def test_max_draft_naive_datetime_treated_as_utc(client):
    port_id = _setup_max_draft_port(client)
    body = {
        **MAX_DRAFT_BODY,
        "allowed_departure_from": "2026-09-24T00:00:00",  # 无时区 → UTC
        "allowed_departure_to": "2026-09-24T04:00:00",
    }
    r = client.post(f"/ports/{port_id}/max-draft", json=body)
    assert r.status_code == 200
    assert r.json()["max_draft"] == 6.0


# ---------------------------------------------------------------------------
# 预报影响预演 POST /ports/{id}/forecast-impact-preview
# ---------------------------------------------------------------------------


def _preview(client, port_id, points):
    return client.post(
        f"/ports/{port_id}/forecast-impact-preview", json={"points": points}
    )


def test_forecast_impact_preview_worsening_sorted_and_read_only(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1, rid2 = r1.json()["id"], r2.json()["id"]
    assert r1.json()["departure_start"] == "2026-09-24T00:36:00+00:00"
    assert r2.json()["departure_start"] == "2026-09-24T01:36:00+00:00"

    # 拟把高潮点 3.0 更正为 1.0：两条预约的通航区间都会变为全程欠深
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}
    ])
    assert r.status_code == 200
    assert r.json() == [
        {"reservation_id": rid1, "current_sufficient": True, "preview_sufficient": False},
        {"reservation_id": rid2, "current_sufficient": True, "preview_sufficient": False},
    ]

    # 预演只读：库存预报、预约、水深复核结果均不受影响
    forecasts = client.get(f"/ports/{port_id}/tide-forecasts").json()
    assert [f["tide_height"] for f in forecasts] == [0.5, 3.0, 0.5]
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert [r["id"] for r in listed] == [rid1, rid2]
    assert listed[0]["departure_start"] == "2026-09-24T00:36:00+00:00"
    audit = client.get(f"/reservations/{rid1}/depth-audit").json()
    assert audit["sufficient"] is True


def test_forecast_impact_preview_improvement(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    # 实际更正高潮点 3.0 → 1.0：预约按当前预报已不达标
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}]},
    )
    # 拟恢复为 3.0：预演判定重新达标
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0}
    ])
    assert r.json() == [
        {"reservation_id": rid, "current_sufficient": False, "preview_sufficient": True}
    ]


def test_forecast_impact_preview_no_change_returns_empty(client):
    port_id = _setup_port(client)
    _create_reservation(client, port_id)
    # 与当前预报完全相同的点 → 无变化
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0}
    ])
    assert r.status_code == 200
    assert r.json() == []
    # 新增远离通航区间、且不改变达标状态的点 → 依然无变化
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 0.4}
    ])
    assert r.json() == []


def test_forecast_impact_preview_same_time_later_submission_wins(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    # 同一时刻同批重复提交，后者覆盖前者：最终 3.0（与当前一致）→ 无变化
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0},
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
    ])
    assert r.json() == []
    # 顺序反过来：最终 1.0 → 变为不达标
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0},
    ])
    assert r.json() == [
        {"reservation_id": rid, "current_sufficient": True, "preview_sufficient": False}
    ]


def test_forecast_impact_preview_uncovered_interval_not_sufficient(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)

    # 模拟预报覆盖收缩：删掉 03:00/06:00 点，再补 01:00 点 → 只覆盖到 01:00，
    # 通航区间 [00:36, 01:36] 未被完全覆盖，当前判为不达标
    conn = sqlite3.connect(database.DATABASE_PATH)
    conn.execute(
        "DELETE FROM tide_forecasts WHERE forecast_time IN (?, ?)",
        ("2026-09-24T03:00:00+00:00", "2026-09-24T06:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [{"forecast_time": "2026-09-24T01:00:00Z", "tide_height": 1.0}]},
    )

    # 拟录入点只延伸覆盖但水深仍不足（00:36 处水深 5.3 < 5.5）→ 无变化
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
        {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 0.5},
    ])
    assert r.json() == []

    # 拟录入点同时修复覆盖与水深 → 不达标变为达标
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T01:00:00Z", "tide_height": 1.5},
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
        {"forecast_time": "2026-09-24T06:00:00Z", "tide_height": 0.5},
    ])
    assert r.json() == [
        {"reservation_id": rid, "current_sufficient": False, "preview_sufficient": True}
    ]


def test_forecast_impact_preview_ignores_cancelled_reservations(client):
    port_id = _setup_port(client)
    r1 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    r2 = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    rid1, rid2 = r1.json()["id"], r2.json()["id"]
    assert client.delete(f"/reservations/{rid1}").status_code == 200

    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}
    ])
    # 已取消预约不参与预演，只有仍有效的第二条状态变化
    assert r.json() == [
        {"reservation_id": rid2, "current_sufficient": True, "preview_sufficient": False}
    ]


def test_forecast_impact_preview_naive_datetime_treated_as_utc(client):
    port_id = _setup_port(client)
    rid = _create_reservation(client, port_id)
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00", "tide_height": 1.0}  # 无时区 → UTC
    ])
    assert r.json() == [
        {"reservation_id": rid, "current_sufficient": True, "preview_sufficient": False}
    ]


def test_forecast_impact_preview_no_reservations_returns_empty(client):
    port_id = _setup_port(client)
    r = _preview(client, port_id, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}
    ])
    assert r.status_code == 200
    assert r.json() == []


def test_forecast_impact_preview_validation_and_404(client):
    port_id = _setup_port(client)
    # 预报点列表为空 → 422
    assert _preview(client, port_id, []).status_code == 422
    # 未知港口 → 404
    assert _preview(client, 999, [
        {"forecast_time": "2026-09-24T03:00:00Z", "tide_height": 1.0}
    ]).status_code == 404


# ---------------------------------------------------------------------------
# 实测潮位校准窗口
# PUT/GET /ports/{id}/tide-observations、POST /ports/{id}/calibrated-windows
# ---------------------------------------------------------------------------


def _put_observations(client, port_id, points):
    return client.put(f"/ports/{port_id}/tide-observations", json={"points": points})


def test_upsert_observations_and_list(client):
    port_id = _setup_port(client)
    r = _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
        {"obs_time": "2026-09-24T00:00:00Z", "tide_height": 0.6},
    ])
    assert r.status_code == 200
    # 返回按时刻升序的全部实测点
    assert r.json() == [
        {"obs_time": "2026-09-24T00:00:00+00:00", "tide_height": 0.6},
        {"obs_time": "2026-09-24T03:00:00+00:00", "tide_height": 2.0},
    ]
    assert client.get(f"/ports/{port_id}/tide-observations").json() == [
        {"obs_time": "2026-09-24T00:00:00+00:00", "tide_height": 0.6},
        {"obs_time": "2026-09-24T03:00:00+00:00", "tide_height": 2.0},
    ]


def test_upsert_observations_same_utc_time_overwrites(client):
    port_id = _setup_port(client)
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
    ])
    # 同港同一 UTC 时刻重复写入覆盖旧值（不同时区写法表示同一 UTC 时刻）
    r = _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T05:00:00+02:00", "tide_height": 2.4},
    ])
    assert r.status_code == 200
    assert r.json() == [
        {"obs_time": "2026-09-24T03:00:00+00:00", "tide_height": 2.4},
    ]


def test_calibrated_windows_uses_observation_residual(client):
    port_id = _setup_port(client)
    # 03:00 实测 2.0（预报 3.0，残差 -1.0）：校准潮高 0.5→2.0→0.5，
    # 水深 5.0→6.5→5.0，所需 5.5 → 达标窗口 [01:00, 05:00]，
    # 60 分钟通航 → 可出发 [01:00, 04:00]
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
    ])
    r = client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["required_depth"] == 5.5
    assert body["duration_minutes"] == 60.0
    assert body["earliest_departure"] == "2026-09-24T01:00:00+00:00"
    assert body["windows"] == [
        {"start": "2026-09-24T01:00:00+00:00", "end": "2026-09-24T04:00:00+00:00"}
    ]

    # 覆盖写回与预报一致的值 3.0：校准窗口等同原窗口 [00:36, 04:24]
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 3.0},
    ])
    body = client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).json()
    assert body["earliest_departure"] == "2026-09-24T00:36:00+00:00"
    assert body["windows"] == [
        {"start": "2026-09-24T00:36:00+00:00", "end": "2026-09-24T04:24:00+00:00"}
    ]


def test_calibrated_windows_without_observations_equals_original(client):
    port_id = _setup_port(client)
    body = {
        "draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60,
    }
    calibrated = client.post(f"/ports/{port_id}/calibrated-windows", json=body).json()
    original = client.post(f"/ports/{port_id}/windows", json=body).json()
    assert calibrated == original


def test_calibrated_windows_ignores_observations_outside_coverage(client):
    port_id = _setup_port(client)
    # 早于首预报点 / 晚于末预报点的观测一律忽略 → 等同原窗口
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-23T23:00:00Z", "tide_height": 9.9},
        {"obs_time": "2026-09-24T07:00:00Z", "tide_height": 9.9},
    ])
    body = {"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60}
    calibrated = client.post(f"/ports/{port_id}/calibrated-windows", json=body).json()
    original = client.post(f"/ports/{port_id}/windows", json=body).json()
    assert calibrated == original


def test_calibrated_windows_interior_observation_between_knots(client):
    # 平潮预报恒 1.0（00:00/04:00），基准水深 4.0：原水深恒 5.0，
    # 需要 5.25 时无窗口；02:00 实测 1.5（残差 +0.5）后校准潮高
    # 1.0→1.5→1.0，水深 5.0→5.5→5.0，达标窗口 [01:00, 03:00]
    r = client.post("/ports", json={"name": "校准港", "reference_depth": 4.0})
    port_id = r.json()["id"]
    client.put(
        f"/ports/{port_id}/tide-forecasts",
        json={"points": [
            {"forecast_time": "2026-09-24T00:00:00Z", "tide_height": 1.0},
            {"forecast_time": "2026-09-24T04:00:00Z", "tide_height": 1.0},
        ]},
    )
    query = {"draft": 4.75, "safety_margin": 0.5, "duration_minutes": 60}
    assert client.post(f"/ports/{port_id}/windows", json=query).json()["windows"] == []

    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T02:00:00Z", "tide_height": 1.5},
    ])
    body = client.post(f"/ports/{port_id}/calibrated-windows", json=query).json()
    assert body["earliest_departure"] == "2026-09-24T01:00:00+00:00"
    assert body["windows"] == [
        {"start": "2026-09-24T01:00:00+00:00", "end": "2026-09-24T02:00:00+00:00"}
    ]


def test_calibrated_windows_boundary_observation_overrides_zero_anchor(client):
    port_id = _setup_port(client)
    # 首预报时刻 00:00 同刻观测 1.2：边界以观测为准，00:00 处水深 5.7 >= 5.5，
    # 校准窗口起点直接落在 00:00（原窗口起点为 00:36）
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T00:00:00Z", "tide_height": 1.2},
    ])
    body = client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).json()
    assert body["windows"][0]["start"] == "2026-09-24T00:00:00+00:00"


def test_calibrated_windows_query_is_read_only(client):
    port_id = _setup_port(client)
    rid = client.post(
        f"/ports/{port_id}/reservations", json=RESERVATION_BODY
    ).json()["id"]
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
    ])
    client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    )
    # 原预报不变
    forecasts = client.get(f"/ports/{port_id}/tide-forecasts").json()
    assert [f["tide_height"] for f in forecasts] == [0.5, 3.0, 0.5]
    # 预约排程不受影响；原 /windows 仍按未校准预报计算
    listed = client.get(f"/ports/{port_id}/reservations").json()
    assert len(listed) == 1
    assert listed[0]["id"] == rid
    assert listed[0]["departure_start"] == "2026-09-24T00:36:00+00:00"
    audit = client.get(f"/reservations/{rid}/depth-audit").json()
    assert audit["sufficient"] is True
    original = client.post(
        f"/ports/{port_id}/windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).json()
    assert original["earliest_departure"] == "2026-09-24T00:36:00+00:00"


def test_observations_do_not_affect_reservation_scheduling(client):
    # 实测点只用于校准窗口查询：预约创建仍按原预报排程
    port_id = _setup_port(client)
    _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
    ])
    r = client.post(f"/ports/{port_id}/reservations", json=RESERVATION_BODY)
    assert r.status_code == 201
    assert r.json()["departure_start"] == "2026-09-24T00:36:00+00:00"


def test_calibration_validation_and_404(client):
    port_id = _setup_port(client)
    # 实测点列表为空 → 422
    assert _put_observations(client, port_id, []).status_code == 422
    # 校准窗口同样校验通航时长为正
    assert client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 0},
    ).status_code == 422
    # 未知港口：写入 / 查询实测点 / 校准窗口均 404
    assert _put_observations(client, 999, [
        {"obs_time": "2026-09-24T03:00:00Z", "tide_height": 2.0},
    ]).status_code == 404
    assert client.get("/ports/999/tide-observations").status_code == 404
    assert client.post(
        "/ports/999/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).status_code == 404


def test_observations_naive_datetime_treated_as_utc(client):
    port_id = _setup_port(client)
    r = _put_observations(client, port_id, [
        {"obs_time": "2026-09-24T03:00:00", "tide_height": 2.0},  # 无时区 → UTC
    ])
    assert r.status_code == 200
    assert r.json() == [
        {"obs_time": "2026-09-24T03:00:00+00:00", "tide_height": 2.0},
    ]
    body = client.post(
        f"/ports/{port_id}/calibrated-windows",
        json={"draft": 5.0, "safety_margin": 0.5, "duration_minutes": 60},
    ).json()
    assert body["earliest_departure"] == "2026-09-24T01:00:00+00:00"
