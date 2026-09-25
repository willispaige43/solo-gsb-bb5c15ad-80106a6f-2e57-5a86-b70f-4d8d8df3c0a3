"""潮汐通航窗口核心计算（纯标准库，便于单测）。

规则：
- 实际水深(t) = 航道基准水深 + 潮高(t)；
- 相邻预报点之间潮高按线性插值；
- 只在预报覆盖时段 [首个预报点, 末个预报点] 内计算；
- 水深 >= 吃水 + 安全富余 的连续时段为达标窗口，端点恰好达标也纳入；
- 可出发时段 = [窗口起点, 窗口终点 − 通航时长]；窗口长度不足通航时长的舍弃；
- 预约水深复核：在排定通航区间内求水深低于所需水深的连续时段（相邻段合并，
  附各段最大欠深）；预报首末点未覆盖的通航时段单独返回，不得视为安全。
- 最大可通航吃水：在允许出发范围内，对“所需水深 = 吃水 + 安全富余”二分搜索，
  复用达标窗口/可出发时段规则判定可行性，求允许范围内全程水深达标所能支持的
  最大所需水深；吃水 = 最大所需水深 − 安全富余，由调用方按 0.01 米向下取整。
- 预报影响预演：拟录入的预报点与当前预报按“同一时刻后者覆盖前者”合并成
  临时预报，复用 depth_deficit_audit 分别判定全程达标与否（纯内存计算）。
- 实测潮位校准：仅采用落在预报覆盖范围内的观测（残差 = 实测 − 观测时刻的
  线性插值预报），以预报首末时刻的零残差为边界锚点（同刻有观测则以观测为准），
  锚点间线性插值残差并叠加原预报；把预报点与有效观测点时刻合并为折点后，
  即可复用现有通航窗口计算（纯内存计算，不改原预报）。
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone

_EPS = 1e-9  # 浮点容差，保证“恰好达标”的端点被纳入

ForecastPoint = tuple[datetime, float]  # (预报时刻, 潮高)


def _cross_time(t0: float, d0: float, t1: float, d1: float, required: float) -> float:
    """线性插值下水深恰好等于 required 的时刻（Unix 秒）。"""
    frac = (required - d0) / (d1 - d0)
    return t0 + frac * (t1 - t0)


def sufficient_depth_windows(
    points: list[ForecastPoint],
    reference_depth: float,
    required_depth: float,
) -> list[tuple[float, float]]:
    """返回水深达标的连续窗口列表 [(start_ts, end_ts)]（Unix 秒，按时间升序）。

    points 需按时刻升序且时刻唯一；不足 2 个预报点时无插值区间，返回空。
    """
    if len(points) < 2:
        return []

    ts = [p[0].timestamp() for p in points]
    depths = [reference_depth + p[1] for p in points]

    windows: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None

    def flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is not None:
            windows.append((cur_start, cur_end))  # type: ignore[arg-type]
            cur_start = cur_end = None

    for i in range(len(points) - 1):
        t0, t1 = ts[i], ts[i + 1]
        d0, d1 = depths[i], depths[i + 1]
        above0 = d0 >= required_depth - _EPS
        above1 = d1 >= required_depth - _EPS

        if above0 and above1:
            seg = (t0, t1)
        elif not above0 and not above1:
            # 线性段两端都不达标，整段不达标；关闭当前窗口
            flush()
            continue
        elif above0:
            seg = (t0, _cross_time(t0, d0, t1, d1, required_depth))
        else:
            seg = (_cross_time(t0, d0, t1, d1, required_depth), t1)

        if cur_start is None:
            cur_start, cur_end = seg
        elif seg[0] <= cur_end + 1e-6:  # type: ignore[operator]
            # 相邻段首尾相接，合并为同一连续窗口
            cur_end = max(cur_end, seg[1])  # type: ignore[type-var]
        else:
            flush()
            cur_start, cur_end = seg

    flush()
    return windows


def departure_windows(
    points: list[ForecastPoint],
    reference_depth: float,
    required_depth: float,
    duration_seconds: float,
) -> tuple[list[tuple[datetime, datetime]], datetime | None]:
    """计算全部可出发时段与最早出发时刻。

    返回 ([(start, end), ...], earliest)，时间均为 UTC datetime。
    """
    result: list[tuple[datetime, datetime]] = []
    for start_ts, end_ts in sufficient_depth_windows(points, reference_depth, required_depth):
        if end_ts - start_ts >= duration_seconds - _EPS:
            result.append((_to_utc(start_ts), _to_utc(end_ts - duration_seconds)))
    earliest = min((w[0] for w in result), default=None)
    return result, earliest


def earliest_feasible_departure(
    points: list[ForecastPoint],
    reference_depth: float,
    required_depth: float,
    duration_seconds: float,
    allowed_start: datetime,
    allowed_end: datetime,
) -> datetime | None:
    """某所需水深下，允许出发范围 [allowed_start, allowed_end] 内的最早可行出发时刻。

    复用 sufficient_depth_windows 的插值规则求达标窗口（已限定在预报覆盖
    时段内），减通航时长得到可出发时段后再与允许出发范围取交集。交集在
    原始（Unix 秒）时间轴上完成，避免整秒取整把可行性边界外扩；仅返回的
    展示时刻按既有约定四舍五入到整秒。无可行时刻返回 None。
    """
    lo, hi = allowed_start.timestamp(), allowed_end.timestamp()
    feasible: list[float] = []
    for start_ts, end_ts in sufficient_depth_windows(
        points, reference_depth, required_depth
    ):
        if end_ts - start_ts >= duration_seconds - _EPS:
            departure = max(start_ts, lo)
            if departure <= min(end_ts - duration_seconds, hi) + _EPS:
                feasible.append(departure)
    return _to_utc(min(feasible)) if feasible else None


def max_supported_required_depth(
    points: list[ForecastPoint],
    reference_depth: float,
    min_required: float,
    duration_seconds: float,
    allowed_start: datetime,
    allowed_end: datetime,
) -> tuple[float, datetime | None]:
    """允许出发范围内全程水深达标所能支持的最大所需水深（二分搜索）。

    可行性关于所需水深单调：所需水深越低越容易找到可行出发时刻。
    返回 (最大所需水深, 该水深下的最早可行出发时刻)；
    连 min_required（对应 0 米吃水 + 安全富余）都不可行时返回 (min_required, None)。
    """
    if (
        earliest_feasible_departure(
            points, reference_depth, min_required,
            duration_seconds, allowed_start, allowed_end,
        )
        is None
    ):
        return min_required, None

    # 上界取预报覆盖时段内的峰值水深：更高的所需水深必然不可行
    lo = min_required
    hi = max((reference_depth + h for _, h in points), default=min_required)
    for _ in range(80):  # 80 次二分，精度远超 0.01 米取整所需
        mid = (lo + hi) / 2
        if (
            earliest_feasible_departure(
                points, reference_depth, mid,
                duration_seconds, allowed_start, allowed_end,
            )
            is not None
        ):
            lo = mid
        else:
            hi = mid
    departure = earliest_feasible_departure(
        points, reference_depth, lo, duration_seconds, allowed_start, allowed_end
    )
    return lo, departure


def earliest_scheduled_departure(
    departable_windows: list[tuple[datetime, datetime]],
    duration_seconds: float,
    allowed_start: datetime,
    allowed_end: datetime,
    blockers: list[tuple[datetime, datetime]],
) -> datetime | None:
    """在潮汐可出发时段内、请求允许范围内，安排不与已有预约重叠的最早出发时刻。

    - departable_windows：departure_windows 求出的可出发时段 [s, e]
      （出发时刻 d ∈ [s, e] 即可保证整个通航过程 [d, d+时长] 水深达标）；
    - allowed_start/end：请求方允许的出发时刻范围（含端点）；
    - blockers：该港口已有有效预约的通航占用区间 [占用起点, 占用终点]；
      首尾相接允许（新区间端点恰好贴住已有区间不算重叠）。

    返回最早可出发时刻（UTC datetime）；无可用时刻返回 None。
    """
    range_lo, range_hi = allowed_start.timestamp(), allowed_end.timestamp()

    # 候选出发区间 = 潮汐可出发时段 ∩ 请求允许范围
    candidates: list[tuple[float, float]] = []
    for start, end in departable_windows:
        lo = max(start.timestamp(), range_lo)
        hi = min(end.timestamp(), range_hi)
        if lo <= hi:
            candidates.append((lo, hi))

    # 已有占用 [a, b] 会禁止 d ∈ (a-时长, b)：d == a-时长（首尾相接）
    # 与 d == b（紧随其后）均允许，因此从候选区间中挖掉开区间、保留两个端点。
    for occ_start, occ_end in sorted(blockers, key=lambda x: x[0].timestamp()):
        forbidden_lo = occ_start.timestamp() - duration_seconds
        forbidden_hi = occ_end.timestamp()
        remaining: list[tuple[float, float]] = []
        for lo, hi in candidates:
            if lo <= forbidden_lo:  # 左侧 [lo, forbidden_lo] 可出发（含相接端点）
                remaining.append((lo, min(hi, forbidden_lo)))
            if hi >= forbidden_hi:  # 右侧 [forbidden_hi, hi] 可出发（含相接端点）
                remaining.append((max(lo, forbidden_hi), hi))
        candidates = remaining
        if not candidates:
            return None

    if not candidates:
        return None
    return _to_utc(min(lo for lo, _ in candidates))


def merge_forecast_points(
    current: list[ForecastPoint],
    submitted: list[ForecastPoint],
) -> list[ForecastPoint]:
    """合并当前预报与一批拟录入的预报点，形成临时预报（纯内存，不写库）。

    同一时刻按提交顺序以后者覆盖前者：先放入当前预报，再按提交顺序
    依次覆盖，因此拟录入点可更正当前值，同批重复时刻以最后提交为准。
    返回按时刻升序的 (时刻, 潮高) 列表。
    """
    merged: dict[datetime, float] = {}
    for t, h in current:
        merged[t] = h
    for t, h in submitted:
        merged[t] = h
    return sorted(merged.items(), key=lambda kv: kv[0])


def _piecewise_linear(points: list[ForecastPoint], t: datetime) -> float:
    """按相邻点线性插值求时刻 t 的值；t 恰落在折点上时取该折点值。

    调用方需保证 t 位于 points 覆盖范围（首末时刻）之内。
    """
    times = [p[0] for p in points]
    i = bisect_right(times, t) - 1
    if i < 0:
        return points[0][1]
    if i >= len(points) - 1:
        return points[-1][1]
    t0, v0 = points[i]
    t1, v1 = points[i + 1]
    if t1 == t0:
        return v1
    frac = (t - t0).total_seconds() / (t1 - t0).total_seconds()
    return v0 + (v1 - v0) * frac


def calibrated_forecast_points(
    forecast: list[ForecastPoint],
    observations: list[ForecastPoint],
) -> list[ForecastPoint]:
    """用实测潮位校准预报，返回校准后的潮高折点序列（纯内存，不改入参）。

    - 仅采用落在预报覆盖范围 [首预报时刻, 末预报时刻] 内的观测，范围外忽略；
    - 观测时刻的预报潮高由最新预报相邻点线性插值得到，残差 = 实测 − 预报；
    - 以预报首末时刻的零残差为边界锚点；边界时刻同刻有观测时以观测为准
      （残差为实测 − 该时刻预报）；观测同刻重复时以最后一个为准；
    - 锚点（边界 + 有效观测）间残差线性插值；校准潮高 = 原预报潮高 + 残差；
    - 折点时刻 = 全部预报点时刻 ∪ 有效观测点时刻，折点间为线性，
      可直接复用 sufficient_depth_windows / departure_windows。
    - 无有效观测（或预报不足 2 点）时，校准结果与原预报完全一致。
    """
    if len(forecast) < 2:
        return list(forecast)

    first_t, last_t = forecast[0][0], forecast[-1][0]

    # 有效观测：落在预报覆盖范围内（含边界），同刻以最后写入为准
    valid: dict[datetime, float] = {}
    for t, observed in observations:
        if first_t <= t <= last_t:
            valid[t] = observed
    if not valid:
        return list(forecast)

    # 残差锚点：预报首末时刻零残差，随后由有效观测覆盖（同刻观测优先）
    anchors: dict[datetime, float] = {first_t: 0.0, last_t: 0.0}
    for t, observed in valid.items():
        predicted = _piecewise_linear(forecast, t)
        anchors[t] = observed - predicted

    anchor_times = sorted(anchors)
    anchor_points: list[ForecastPoint] = [
        (t, anchors[t]) for t in anchor_times
    ]
    # 折点时刻 = 预报点时刻 ∪ 有效观测点时刻；观测点处校准值即实测值
    observed: dict[datetime, float] = dict(valid.items())
    result: list[ForecastPoint] = []
    for t, predicted in forecast:
        if t in observed:
            result.append((t, observed.pop(t)))
        else:
            result.append((t, predicted + _piecewise_linear(anchor_points, t)))
    for t, observed_height in observed.items():
        result.append((t, observed_height))
    result.sort(key=lambda p: p[0])
    return result


def depth_deficit_audit(
    points: list[ForecastPoint],
    reference_depth: float,
    required_depth: float,
    start: datetime,
    end: datetime,
) -> tuple[list[tuple[datetime, datetime, float]], list[tuple[datetime, datetime]]]:
    """复核通航区间 [start, end] 的水深（按最新预报相邻点线性插值）。

    返回 (deficits, uncovered)：
    - deficits：通航区间内水深低于 required_depth 的连续时段
      [(起点, 终点, 最大欠深)]，跨预报点的连续欠深已合并，
      最大欠深 = required_depth − 时段内最低水深；
    - uncovered：预报首末点未覆盖的通航时段 [(起点, 终点)]，
      这些时段水深未知，调用方不得据此判为安全。
    """
    start_ts = start.timestamp()
    end_ts = end.timestamp()

    if len(points) < 2:
        # 无插值区间：整个通航区间都未被预报覆盖
        uncovered = [(start, end)] if start_ts < end_ts else []
        return [], uncovered

    ts = [p[0].timestamp() for p in points]
    depths = [reference_depth + p[1] for p in points]
    first_ts, last_ts = ts[0], ts[-1]

    uncovered_ts: list[tuple[float, float]] = []
    if start_ts < min(end_ts, first_ts):  # 通航起点早于首个预报点
        uncovered_ts.append((start_ts, min(end_ts, first_ts)))
    if max(start_ts, last_ts) < end_ts:  # 通航终点晚于末个预报点
        uncovered_ts.append((max(start_ts, last_ts), end_ts))

    lo = max(start_ts, first_ts)
    hi = min(end_ts, last_ts)

    deficits_ts: list[tuple[float, float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_max = 0.0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_max
        if cur_start is not None:
            deficits_ts.append((cur_start, cur_end, round(cur_max, 6)))  # type: ignore[arg-type]
            cur_start = cur_end = None
            cur_max = 0.0

    for i in range(len(points) - 1):
        t0, t1 = ts[i], ts[i + 1]
        a, b = max(t0, lo), min(t1, hi)  # 插值段与通航区间（覆盖范围内）的交集
        if a >= b:
            continue
        # 线性插值出交集端点处的水深
        d0 = depths[i] + (depths[i + 1] - depths[i]) * (a - t0) / (t1 - t0)
        d1 = depths[i] + (depths[i + 1] - depths[i]) * (b - t0) / (t1 - t0)
        below0 = d0 < required_depth - _EPS
        below1 = d1 < required_depth - _EPS

        if not below0 and not below1:
            # 线性段两端都不欠深，整段达标；关闭当前欠深时段
            flush()
            continue
        if below0 and below1:
            seg = (a, b)
            seg_max = required_depth - min(d0, d1)
        elif below0:
            seg = (a, _cross_time(a, d0, b, d1, required_depth))
            seg_max = required_depth - d0
        else:
            seg = (_cross_time(a, d0, b, d1, required_depth), b)
            seg_max = required_depth - d1

        if cur_start is None:
            cur_start, cur_end, cur_max = seg[0], seg[1], seg_max
        elif seg[0] <= cur_end + 1e-6:  # type: ignore[operator]
            # 相邻段首尾相接（如共享的预报点同样欠深），合并为同一连续欠深时段
            cur_end = max(cur_end, seg[1])  # type: ignore[type-var]
            cur_max = max(cur_max, seg_max)
        else:
            flush()
            cur_start, cur_end, cur_max = seg[0], seg[1], seg_max

    flush()

    deficits = [(_to_utc(s), _to_utc(e), m) for s, e, m in deficits_ts]
    uncovered = [(_to_utc(s), _to_utc(e)) for s, e in uncovered_ts]
    return deficits, uncovered


def _to_utc(ts: float) -> datetime:
    """Unix 秒转 UTC datetime，四舍五入到整秒避免浮点噪声。"""
    return datetime.fromtimestamp(round(ts), tz=timezone.utc)
