"""核心窗口计算算法的单元测试（仅标准库）。"""
from datetime import datetime, timedelta, timezone

from app.tides import (
    calibrated_forecast_points,
    departure_windows,
    depth_deficit_audit,
    earliest_feasible_departure,
    earliest_scheduled_departure,
    max_supported_required_depth,
    merge_forecast_points,
    observation_anomalies,
    sufficient_depth_windows,
)

BASE = datetime(2026, 9, 24, tzinfo=timezone.utc)


def pts(*heights: float, step_minutes: int = 60):
    """从 BASE 起按等间隔生成 (时刻, 潮高) 序列。"""
    return [
        (BASE + timedelta(minutes=step_minutes * i), h) for i, h in enumerate(heights)
    ]


def test_single_tide_cycle_window():
    # 基准水深 3，潮高 1→3→1：水深 4→6→4，要求 5
    # 上升段 00:30 达标，下降段 01:30 退出
    windows = sufficient_depth_windows(pts(1, 3, 1), reference_depth=3.0, required_depth=5.0)
    assert len(windows) == 1
    start, end = windows[0]
    assert start == (BASE + timedelta(minutes=30)).timestamp()
    assert end == (BASE + timedelta(hours=1, minutes=30)).timestamp()


def test_exact_threshold_endpoints_included():
    # 水深恰好全程等于要求值，端点恰好达标也纳入
    windows = sufficient_depth_windows(pts(2, 2), reference_depth=3.0, required_depth=5.0)
    assert windows == [(BASE.timestamp(), (BASE + timedelta(hours=1)).timestamp())]


def test_departure_window_subtracts_duration():
    # 达标窗口 [00:30, 01:30]，通航 30 分钟 → 可出发 [00:30, 01:00]
    deps, earliest = departure_windows(
        pts(1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=1800
    )
    assert len(deps) == 1
    assert deps[0][0] == BASE + timedelta(minutes=30)
    assert deps[0][1] == BASE + timedelta(hours=1)
    assert earliest == BASE + timedelta(minutes=30)


def test_window_shorter_than_duration_discarded():
    # 达标窗口 60 分钟 < 所需 90 分钟 → 舍弃
    deps, earliest = departure_windows(
        pts(1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=5400
    )
    assert deps == []
    assert earliest is None


def test_duration_exactly_equals_window_kept():
    # 窗口 60 分钟 == 所需 60 分钟 → 保留，可出发时段退化为同一时刻
    deps, earliest = departure_windows(
        pts(1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=3600
    )
    assert len(deps) == 1
    assert deps[0][0] == deps[0][1] == BASE + timedelta(minutes=30)
    assert earliest == BASE + timedelta(minutes=30)


def test_multiple_windows():
    # 潮高 1,3,1,3,1（基准 3，要求 5）：两个独立高潮窗口 [00:30,01:30]、[02:30,03:30]
    deps, earliest = departure_windows(
        pts(1, 3, 1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=60
    )
    assert len(deps) == 2
    # 第一个窗口 [00:30, 01:30]，可出发 [00:30, 01:29]
    assert deps[0][0] == BASE + timedelta(minutes=30)
    assert deps[0][1] == BASE + timedelta(hours=1, minutes=29)
    # 第二个窗口 [02:30, 03:30]
    assert deps[1][0] == BASE + timedelta(hours=2, minutes=30)
    assert deps[1][1] == BASE + timedelta(hours=3, minutes=29)
    assert earliest == deps[0][0]


def test_all_below_required():
    deps, earliest = departure_windows(
        pts(0.5, 1.0, 0.5), reference_depth=3.0, required_depth=5.0, duration_seconds=60
    )
    assert deps == []
    assert earliest is None


def test_fewer_than_two_points_no_coverage():
    deps, earliest = departure_windows(
        pts(5.0), reference_depth=3.0, required_depth=5.0, duration_seconds=60
    )
    assert deps == []
    assert earliest is None


def test_window_clipped_at_forecast_coverage():
    # 全程达标时窗口就是预报覆盖时段本身，不外推
    windows = sufficient_depth_windows(pts(3, 4, 3), reference_depth=3.0, required_depth=5.0)
    assert windows == [(BASE.timestamp(), (BASE + timedelta(hours=2)).timestamp())]


# ---------------------------------------------------------------------------
# earliest_scheduled_departure：潮汐窗口 + 请求范围 + 已有预约排程
# ---------------------------------------------------------------------------

# 一个 [00:00, 04:00] 的可出发时段（出发时刻落在此区间内即可）
DEP_WINDOWS = [(BASE, BASE + timedelta(hours=4))]
DURATION = 3600  # 60 分钟


def test_schedule_earliest_within_range():
    # 无占用、范围覆盖整个窗口 → 取范围起点（最早）
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=3), [],
    )
    assert dep == BASE


def test_schedule_intersects_range_start():
    # 允许范围从 01:00 开始 → 01:00 出发
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE + timedelta(hours=1), BASE + timedelta(hours=3), [],
    )
    assert dep == BASE + timedelta(hours=1)


def test_schedule_blocker_pushes_after_end():
    # 已有预约占用 [00:30, 01:30] → d 必须 >= 01:30（首尾相接允许）
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=4),
        [(BASE + timedelta(minutes=30), BASE + timedelta(hours=1, minutes=30))],
    )
    assert dep == BASE + timedelta(hours=1, minutes=30)


def test_schedule_departure_exactly_touching_blocker_start_allowed():
    # 预约占用 [01:00, 02:00]：00:00 出发、01:00 结束，首尾相接 → 允许
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=4),
        [(BASE + timedelta(hours=1), BASE + timedelta(hours=2))],
    )
    assert dep == BASE


def test_schedule_blocker_covering_range_start_takes_right_edge():
    # 占用 [00:00, 01:00]：00:00 出发会在 [00:00,01:00] 内重叠 → 最早 01:00
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=4),
        [(BASE, BASE + timedelta(hours=1))],
    )
    assert dep == BASE + timedelta(hours=1)


def test_schedule_no_room_between_blockers():
    # 两个相邻占用 [00:00,01:00]、[01:30,02:30]：
    # 间隙 01:00 出发需通航到 02:00，与第二个预约重叠 → 只能 02:30
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=4),
        [
            (BASE, BASE + timedelta(hours=1)),
            (BASE + timedelta(hours=1, minutes=30), BASE + timedelta(hours=2, minutes=30)),
        ],
    )
    assert dep == BASE + timedelta(hours=2, minutes=30)


def test_schedule_exact_fit_between_blockers():
    # 占用 [00:00,01:00]、[02:00,03:00]：间隙恰好 1 小时，01:00 出发 02:00 结束
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=4),
        [
            (BASE, BASE + timedelta(hours=1)),
            (BASE + timedelta(hours=2), BASE + timedelta(hours=3)),
        ],
    )
    assert dep == BASE + timedelta(hours=1)


def test_schedule_range_fully_blocked_returns_none():
    # 占用 [00:00, 03:30]，允许范围 00:00–02:30 内无可行出发
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE, BASE + timedelta(hours=2, minutes=30),
        [(BASE, BASE + timedelta(hours=3, minutes=30))],
    )
    assert dep is None


def test_schedule_range_outside_tide_windows_returns_none():
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE + timedelta(hours=5), BASE + timedelta(hours=6), [],
    )
    assert dep is None


def test_schedule_range_touching_tide_window_endpoint_allowed():
    # 可出发时段终点 04:00，允许范围末端恰好 04:00 → 04:00 出发（端点纳入）
    dep = earliest_scheduled_departure(
        DEP_WINDOWS, DURATION,
        BASE + timedelta(hours=4), BASE + timedelta(hours=4),
        [],
    )
    assert dep == BASE + timedelta(hours=4)


# ---------------------------------------------------------------------------
# depth_deficit_audit：预约通航区间水深复核
# ---------------------------------------------------------------------------


def test_audit_single_deficit_segment_with_max_deficit():
    # 基准 3，潮高 3→1→3：水深 6→4→6，要求 5；00:30 跌破，01:30 回升
    deficits, uncovered = depth_deficit_audit(
        pts(3, 1, 3), reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=2),
    )
    assert uncovered == []
    assert len(deficits) == 1
    start, end, max_deficit = deficits[0]
    assert start == BASE + timedelta(minutes=30)
    assert end == BASE + timedelta(hours=1, minutes=30)
    assert max_deficit == 1.0  # 最低水深 4.0（01:00），欠深 5 − 4 = 1


def test_audit_merges_deficit_across_forecast_points():
    # 水深 6→4.5→4→6：欠深段在欠深的预报点处相连，合并为一段，最大欠深取全段最低
    deficits, uncovered = depth_deficit_audit(
        pts(3, 1.5, 1, 3), reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=3),
    )
    assert uncovered == []
    assert len(deficits) == 1
    start, end, max_deficit = deficits[0]
    assert start == BASE + timedelta(minutes=40)   # 6→4.5 段在 00:40 跌破 5
    assert end == BASE + timedelta(hours=2, minutes=30)  # 4→6 段在 02:30 回升到 5
    assert max_deficit == 1.0


def test_audit_two_separate_deficit_segments():
    # 潮高 3,1,3,1,3（基准 3，要求 5）：两个独立低谷，各自成段
    deficits, _ = depth_deficit_audit(
        pts(3, 1, 3, 1, 3), reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=4),
    )
    assert len(deficits) == 2
    assert deficits[0][0] == BASE + timedelta(minutes=30)
    assert deficits[0][1] == BASE + timedelta(hours=1, minutes=30)
    assert deficits[1][0] == BASE + timedelta(hours=2, minutes=30)
    assert deficits[1][1] == BASE + timedelta(hours=3, minutes=30)
    assert [m for _, _, m in deficits] == [1.0, 1.0]


def test_audit_exact_threshold_is_not_deficit():
    # 水深在 01:00 恰好触及所需水深 5.0，不算欠深
    deficits, uncovered = depth_deficit_audit(
        pts(3, 2, 3), reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=2),
    )
    assert deficits == []
    assert uncovered == []


def test_audit_clips_to_transit_interval():
    # 只在通航区间内复核：通航 00:15–01:00，水深 4.5→6，00:30 起达标
    deficits, uncovered = depth_deficit_audit(
        pts(1, 3), reference_depth=3.0, required_depth=5.0,
        start=BASE + timedelta(minutes=15), end=BASE + timedelta(hours=1),
    )
    assert uncovered == []
    assert len(deficits) == 1
    start, end, max_deficit = deficits[0]
    assert start == BASE + timedelta(minutes=15)
    assert end == BASE + timedelta(minutes=30)
    assert max_deficit == 0.5


def test_audit_forecast_not_covering_transit_head_and_tail():
    # 预报只覆盖 [01:00, 02:00]，通航 [00:00, 03:00]：首尾各一段未覆盖
    points = [
        (BASE + timedelta(hours=1), 3.0),
        (BASE + timedelta(hours=2), 3.0),
    ]
    deficits, uncovered = depth_deficit_audit(
        points, reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=3),
    )
    assert deficits == []  # 覆盖段内水深 6.0 达标
    assert uncovered == [
        (BASE, BASE + timedelta(hours=1)),
        (BASE + timedelta(hours=2), BASE + timedelta(hours=3)),
    ]


def test_audit_without_forecast_points_all_uncovered():
    start, end = BASE, BASE + timedelta(hours=1)
    for points in ([], [(BASE, 5.0)]):  # 无预报点 / 单点无插值区间
        deficits, uncovered = depth_deficit_audit(
            points, reference_depth=3.0, required_depth=5.0, start=start, end=end,
        )
        assert deficits == []
        assert uncovered == [(start, end)]


def test_audit_deficit_stops_at_forecast_coverage_edge():
    # 通航 [00:00, 02:00]，预报只覆盖 [00:00, 01:00] 且全程水深不足：
    # 欠深段到预报末点为止，其余时段按未覆盖返回（不得视为安全）
    points = [(BASE, 1.0), (BASE + timedelta(hours=1), 1.0)]
    deficits, uncovered = depth_deficit_audit(
        points, reference_depth=3.0, required_depth=5.0,
        start=BASE, end=BASE + timedelta(hours=2),
    )
    assert deficits == [(BASE, BASE + timedelta(hours=1), 1.0)]
    assert uncovered == [(BASE + timedelta(hours=1), BASE + timedelta(hours=2))]


# ---------------------------------------------------------------------------
# earliest_feasible_departure / max_supported_required_depth：最大可通航吃水
# ---------------------------------------------------------------------------


def test_earliest_feasible_departure_intersects_allowed_range():
    # 与 /windows 相同的可出发时段 [00:30, 01:00]，允许范围从 00:45 起 → 00:45
    dep = earliest_feasible_departure(
        pts(1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=1800,
        allowed_start=BASE + timedelta(minutes=45),
        allowed_end=BASE + timedelta(hours=2),
    )
    assert dep == BASE + timedelta(minutes=45)


def test_earliest_feasible_departure_none_outside_coverage():
    # 允许范围完全在预报覆盖时段之外 → 无可行时刻
    dep = earliest_feasible_departure(
        pts(1, 3, 1), reference_depth=3.0, required_depth=5.0, duration_seconds=1800,
        allowed_start=BASE + timedelta(hours=3),
        allowed_end=BASE + timedelta(hours=4),
    )
    assert dep is None


def test_max_supported_required_depth_centered_on_high_tide():
    # 基准 4.0，潮高 1→3→1（00:00/02:00/04:00）：水深 5→7→5
    # 60 分钟通航的最优区间为以高潮为中心的 [01:30, 02:30]，两端水深 6.5
    supported, departure = max_supported_required_depth(
        pts(1, 3, 1, step_minutes=120), reference_depth=4.0, min_required=0.5,
        duration_seconds=3600,
        allowed_start=BASE, allowed_end=BASE + timedelta(hours=4),
    )
    assert abs(supported - 6.5) < 1e-6
    assert departure == BASE + timedelta(hours=1, minutes=30)


def test_max_supported_required_depth_flat_tide_exact():
    # 全程水深恒为 5.0：最大所需水深恰为 5.0，不得因浮点噪声退档
    supported, departure = max_supported_required_depth(
        pts(2, 2), reference_depth=3.0, min_required=0.0,
        duration_seconds=1800,
        allowed_start=BASE, allowed_end=BASE + timedelta(hours=1),
    )
    assert abs(supported - 5.0) < 1e-6
    assert departure == BASE


def test_max_supported_required_depth_respects_allowed_range():
    # 只允许 00:00–01:00 出发：最优 01:00 出发，[01:00, 02:00] 最低水深 6.0
    supported, departure = max_supported_required_depth(
        pts(1, 3, 1, step_minutes=120), reference_depth=4.0, min_required=0.5,
        duration_seconds=3600,
        allowed_start=BASE, allowed_end=BASE + timedelta(hours=1),
    )
    assert abs(supported - 6.0) < 1e-6
    assert departure == BASE + timedelta(hours=1)


def test_max_supported_required_depth_infeasible_returns_none():
    # 允许范围在预报覆盖之外，连 min_required（0 米吃水）也不可行
    supported, departure = max_supported_required_depth(
        pts(1, 3, 1), reference_depth=4.0, min_required=0.5,
        duration_seconds=3600,
        allowed_start=BASE + timedelta(hours=5),
        allowed_end=BASE + timedelta(hours=6),
    )
    assert supported == 0.5
    assert departure is None


def test_max_supported_required_depth_duration_exceeds_coverage():
    # 通航时长超过预报覆盖时长 → 任何吃水都不可行
    _, departure = max_supported_required_depth(
        pts(3, 3), reference_depth=4.0, min_required=0.0,
        duration_seconds=7200,  # 覆盖仅 60 分钟
        allowed_start=BASE, allowed_end=BASE + timedelta(hours=1),
    )
    assert departure is None


def test_merge_forecast_points_later_submission_overrides():
    # 当前预报 3 个点；拟录入点更正 01:00、新增 03:00，
    # 且同批 02:00 重复提交 → 以最后提交的值为准
    current = pts(1.0, 2.0, 3.0)
    submitted = [
        (BASE + timedelta(hours=1), 2.5),   # 更正现有点
        (BASE + timedelta(hours=3), 4.0),   # 新增点
        (BASE + timedelta(hours=2), 9.9),   # 同批先提交
        (BASE + timedelta(hours=2), 3.5),   # 同批后提交，覆盖前者
    ]
    merged = merge_forecast_points(current, submitted)
    assert merged == [
        (BASE, 1.0),
        (BASE + timedelta(hours=1), 2.5),
        (BASE + timedelta(hours=2), 3.5),
        (BASE + timedelta(hours=3), 4.0),
    ]
    # 纯内存合并：入参不被修改
    assert current == pts(1.0, 2.0, 3.0)
    assert len(submitted) == 4


# ---------------------------------------------------------------------------
# calibrated_forecast_points：实测潮位校准
# ---------------------------------------------------------------------------


def test_calibration_without_observations_equals_forecast():
    forecast = pts(1.0, 2.0, 3.0)
    assert calibrated_forecast_points(forecast, []) == forecast
    # 纯内存计算：入参不被修改
    assert forecast == pts(1.0, 2.0, 3.0)


def test_calibration_fewer_than_two_forecast_points():
    # 无插值区间时观测无法采用，原样返回
    assert calibrated_forecast_points([], [(BASE, 1.0)]) == []
    one = [(BASE, 1.0)]
    assert calibrated_forecast_points(one, [(BASE, 2.0)]) == one


def test_calibration_single_observation_residual_interpolation():
    # 预报 0.0→2.0→0.0（00/01/02 时），01:00 实测 1.5 → 残差 -0.5
    # 锚点残差：00:00=0、01:00=-0.5、02:00=0
    forecast = pts(0.0, 2.0, 0.0)
    calibrated = calibrated_forecast_points(
        forecast, [(BASE + timedelta(hours=1), 1.5)]
    )
    assert calibrated == [
        (BASE, 0.0),
        (BASE + timedelta(hours=1), 1.5),
        (BASE + timedelta(hours=2), 0.0),
    ]


def test_calibration_interior_observation_at_forecast_knot_time():
    # 观测时刻与预报点重合（非边界）：校准值即实测值
    forecast = pts(0.0, 2.0, 0.0)
    calibrated = calibrated_forecast_points(
        forecast, [(BASE + timedelta(hours=1), 2.3)]
    )
    by_time = dict(calibrated)
    assert by_time[BASE + timedelta(hours=1)] == 2.3
    # 边界仍为零残差：校准值 = 原预报值
    assert by_time[BASE] == 0.0
    assert by_time[BASE + timedelta(hours=2)] == 0.0


def test_calibration_observation_between_knots_creates_vertex():
    # 线性预报 0.0→2.0（00:00–02:00），01:00 实测 1.8（预报插值 1.0，残差 0.8）
    forecast = [
        (BASE, 0.0),
        (BASE + timedelta(hours=2), 2.0),
    ]
    obs_t = BASE + timedelta(hours=1)
    calibrated = calibrated_forecast_points(forecast, [(obs_t, 1.8)])
    assert calibrated == [(BASE, 0.0), (obs_t, 1.8), (BASE + timedelta(hours=2), 2.0)]
    # 折点间线性：01:30 处校准潮高 = 1.9
    deps, _ = departure_windows(
        calibrated, reference_depth=0.0, required_depth=1.9, duration_seconds=1800
    )
    # 水深 0→1.8→2 单调上升，达标从 01:30 到 02:00，恰好 30 分钟可出发
    assert deps == [(BASE + timedelta(hours=1, minutes=30), BASE + timedelta(hours=1, minutes=30))]


def test_calibration_boundary_observation_overrides_zero_residual():
    # 边界时刻同刻有观测：以观测为准（残差不再强制为零）
    forecast = pts(1.0, 2.0, 1.0)
    calibrated = calibrated_forecast_points(forecast, [(BASE, 1.4)])
    by_time = dict(calibrated)
    assert by_time[BASE] == 1.4
    # 另一边界无观测，仍是零残差
    assert by_time[BASE + timedelta(hours=2)] == 1.0


def test_calibration_observations_outside_coverage_ignored():
    forecast = pts(1.0, 2.0, 1.0)
    observations = [
        (BASE - timedelta(hours=1), 9.9),   # 早于首预报点
        (BASE + timedelta(hours=3), 9.9),   # 晚于末预报点
    ]
    assert calibrated_forecast_points(forecast, observations) == forecast


def test_calibration_duplicate_observation_last_wins():
    # 调用方按写入顺序传入时，同刻以最后一个观测为准
    forecast = pts(0.0, 2.0, 0.0)
    calibrated = calibrated_forecast_points(
        forecast,
        [
            (BASE + timedelta(hours=1), 1.0),
            (BASE + timedelta(hours=1), 1.5),
        ],
    )
    assert dict(calibrated)[BASE + timedelta(hours=1)] == 1.5


def test_calibration_two_observations_residual_piecewise_linear():
    # 预报恒为 1.0（平潮）：残差锚点 00:00=0、01:00=+0.4、02:00=0
    forecast = pts(1.0, 1.0, 1.0)
    calibrated = calibrated_forecast_points(
        forecast,
        [
            (BASE + timedelta(hours=1), 1.4),
        ],
    )
    by_time = dict(calibrated)
    assert by_time[BASE] == 1.0
    assert by_time[BASE + timedelta(hours=1)] == 1.4
    assert by_time[BASE + timedelta(hours=2)] == 1.0
    # 复用窗口计算：基准水深 4.0、所需水深 5.0（潮高需 >= 1.0），
    # 除两端点外全程高于 1.0 → 达标窗口覆盖 (00:00, 02:00]
    windows = sufficient_depth_windows(calibrated, 4.0, 5.0)
    assert len(windows) == 1
    assert windows[0][0] == BASE.timestamp()
    assert windows[0][1] == (BASE + timedelta(hours=2)).timestamp()


# ---------------------------------------------------------------------------
# observation_anomalies：实测潮高异常点质检
# ---------------------------------------------------------------------------


def flat_forecast(end_hours: int = 8, height: float = 0.0):
    """平潮预报：00:00 与 end_hours 点同高，任意时刻插值潮高即 height。"""
    return [
        (BASE, height),
        (BASE + timedelta(hours=end_hours), height),
    ]


def obs_at(hours_from_base, height):
    return (BASE + timedelta(hours=hours_from_base), height)


def test_observation_anomaly_spike_with_baseline_deviation_fields():
    # 平潮预报 0.0；00:00–06:00 每小时一条实测，残差 ≈ 0，03:00 尖峰 +1.0
    observations = [
        obs_at(h, 1.0 if h == 3 else 0.0) for h in range(0, 7)
    ]
    anomalies = observation_anomalies(flat_forecast(), observations)
    assert len(anomalies) == 1
    t, observed, residual, baseline, deviation = anomalies[0]
    assert t == BASE + timedelta(hours=3)
    assert observed == 1.0
    assert residual == 1.0
    assert baseline == 0.0          # 02:00/04:00 等邻点残差均为 0
    assert deviation == 1.0         # 严格超过 0.3 阈值
    # 纯内存计算：入参不被修改
    assert len(observations) == 7


def test_observation_anomaly_fewer_than_three_neighbors_skipped():
    # 只有 3 条有效实测：每条自身之外至多 2 个邻点 → 全部不评估
    observations = [
        obs_at(0, 0.0), obs_at(2, 1.0), obs_at(4, 0.0),
    ]
    assert observation_anomalies(flat_forecast(), observations) == []


def test_observation_anomaly_neighbor_window_is_120_minutes_inclusive():
    # 03:00 尖峰；01:00 与 05:00 恰在 ±120 分钟边界上（含），计入邻点，
    # 与 02:00 一起凑满 3 个邻点 → 可评估并判异
    observations = [
        obs_at(1, 0.0), obs_at(2, 0.0), obs_at(3, 1.0), obs_at(5, 0.0),
    ]
    anomalies = observation_anomalies(flat_forecast(6), observations)
    assert len(anomalies) == 1
    assert anomalies[0][0] == BASE + timedelta(hours=3)

    # 01:00/05:00 改为恰 121 分钟外（00:00/06:00）后，03:00 自身之外无邻点
    observations = [
        obs_at(0, 0.0), obs_at(3, 1.0), obs_at(6, 0.0),
    ]
    assert observation_anomalies(flat_forecast(6), observations) == []


def test_observation_anomaly_floor_threshold_is_strict_0_3():
    # 邻点残差均为 0 → MAD=0，阈值 = 0.3；偏离恰为 0.3 不判异
    base_obs = [obs_at(1, 0.0), obs_at(2, 0.0), obs_at(4, 0.0), obs_at(5, 0.0)]
    at_threshold = base_obs + [obs_at(3, 0.3)]
    assert observation_anomalies(flat_forecast(6), at_threshold) == []

    # 偏离 0.3001 严格超过阈值 → 判异，偏离量按实际残差返回
    over_threshold = base_obs + [obs_at(3, 0.3001)]
    anomalies = observation_anomalies(flat_forecast(6), over_threshold)
    assert len(anomalies) == 1
    assert anomalies[0][0] == BASE + timedelta(hours=3)
    assert abs(anomalies[0][4] - 0.3001) < 1e-9


def test_observation_anomaly_mad_raises_threshold_in_volatile_region():
    # 平静区：残差都贴近 0，+0.5 尖峰超过 0.3 下限 → 判异
    calm = [
        obs_at(1, 0.0), obs_at(2, 0.05), obs_at(3, 0.5),
        obs_at(4, -0.05), obs_at(5, 0.0),
    ]
    anomalies = observation_anomalies(flat_forecast(6), calm)
    assert [a[0] for a in anomalies] == [BASE + timedelta(hours=3)]

    # 同一尖峰放在抖动区（邻点残差 ±0.2 交替）：03:00 邻点残差
    # -0.2/-0.2/+0.2/+0.2，基线 0、MAD=0.2，
    # 阈值 = max(0.3, 3×1.4826×0.2=0.88956) = 0.88956，0.5 偏离不再异常
    volatile = [
        obs_at(1, 0.2), obs_at(2, -0.2), obs_at(3, 0.5),
        obs_at(4, -0.2), obs_at(5, 0.2), obs_at(6, -0.2),
        obs_at(7, 0.2),
    ]
    flagged = [a[0] for a in observation_anomalies(flat_forecast(8), volatile)]
    assert BASE + timedelta(hours=3) not in flagged


def test_observation_anomaly_uses_linear_interpolation_residual():
    # 线性预报 0.0→4.0（00:00–04:00），正常实测贴合预报（残差 0），
    # 02:00 实测比插值预报（2.0）高 1.0 → 残差 +1.0 判异
    forecast = [(BASE, 0.0), (BASE + timedelta(hours=4), 4.0)]
    observations = [
        (BASE, 0.0),
        (BASE + timedelta(hours=1), 1.0),
        (BASE + timedelta(hours=2), 3.0),   # 插值预报 2.0，残差 +1.0
        (BASE + timedelta(hours=3), 3.0),
        (BASE + timedelta(hours=4), 4.0),
    ]
    anomalies = observation_anomalies(forecast, observations)
    assert len(anomalies) == 1
    t, observed, residual, baseline, deviation = anomalies[0]
    assert t == BASE + timedelta(hours=2)
    assert observed == 3.0
    assert abs(residual - 1.0) < 1e-9
    assert abs(baseline - 0.0) < 1e-9
    assert abs(deviation - 1.0) < 1e-9


def test_observation_anomaly_ignores_observations_outside_coverage():
    # 预报覆盖 [02:00, 06:00]：覆盖外观测不参与，04:00 尖峰在覆盖内但
    # 覆盖内邻点只有 03:00/05:00 两个 → 不足 3 个，不评估
    forecast = [
        (BASE + timedelta(hours=2), 0.0),
        (BASE + timedelta(hours=6), 0.0),
    ]
    observations = [
        obs_at(1, 0.0),   # 覆盖外
        obs_at(3, 0.0),
        obs_at(4, 1.0),   # 尖峰，但覆盖内邻点不足
        obs_at(5, 0.0),
        obs_at(7, 0.0),   # 覆盖外
    ]
    assert observation_anomalies(forecast, observations) == []

    # 覆盖边界时刻（含端点）的观测计入：补齐 02:00 点后邻点达 3 个 → 判异
    observations_with_edge = observations + [obs_at(2, 0.0)]
    anomalies = observation_anomalies(forecast, observations_with_edge)
    assert [a[0] for a in anomalies] == [BASE + timedelta(hours=4)]


def test_observation_anomaly_fewer_than_two_forecast_points():
    observations = [obs_at(h, 0.0) for h in range(4)]
    assert observation_anomalies([], observations) == []
    assert observation_anomalies([(BASE, 0.0)], observations) == []


def test_observation_anomaly_results_sorted_with_negative_spike():
    # 03:00 向上尖峰 +1.0、07:00 向下尖峰 -1.0，结果按时刻升序
    observations = [
        obs_at(h, (1.0 if h == 3 else -1.0 if h == 7 else 0.0))
        for h in range(0, 11)
    ]
    anomalies = observation_anomalies(flat_forecast(10), observations)
    assert [a[0] for a in anomalies] == [
        BASE + timedelta(hours=3),
        BASE + timedelta(hours=7),
    ]
    assert anomalies[0][2] == 1.0 and anomalies[0][4] == 1.0
    assert anomalies[1][2] == -1.0 and anomalies[1][4] == 1.0


def test_observation_anomaly_baseline_follows_local_residual_offset():
    # 局部残差整体偏移 +0.2（基线不为零）：与邻点一致的 +0.2 不异常，
    # 高出基线 1.0 的点才异常，偏离量相对基线计算
    observations = [
        obs_at(1, 0.2), obs_at(2, 0.2), obs_at(3, 1.2),
        obs_at(4, 0.2), obs_at(5, 0.2),
    ]
    anomalies = observation_anomalies(flat_forecast(6), observations)
    assert len(anomalies) == 1
    t, observed, residual, baseline, deviation = anomalies[0]
    assert t == BASE + timedelta(hours=3)
    assert residual == 1.2
    assert baseline == 0.2
    assert deviation == 1.0


def test_observation_anomaly_duplicate_time_last_wins():
    # 同刻重复观测以最后一个为准：先写 03:00 正常值，再覆盖为尖峰值
    observations = [obs_at(h, 0.0) for h in range(0, 7)]
    observations += [obs_at(3, 0.0), obs_at(3, 1.0)]
    anomalies = observation_anomalies(flat_forecast(), observations)
    assert len(anomalies) == 1
    assert anomalies[0][0] == BASE + timedelta(hours=3)
    assert anomalies[0][1] == 1.0
