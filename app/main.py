"""港口潮汐通航窗口与单向航道预约 API。

- 录入港口（航道基准水深）与潮高预报点；
- 相邻预报点线性插值，在预报覆盖时段内求水深 >= 吃水 + 安全富余的连续窗口；
- 窗口结束时刻减去通航时长得到可出发时段，不足时长的窗口舍弃；
- 查询窗口时实时读取最新预报数据计算，预报点更正后立即生效；
- 单向航道预约：在请求的允许出发范围内，按最新预报安排最早出发时刻，
  整个通航过程不与该港口有效预约重叠（首尾相接允许）；无可用时刻返回 409；
  已落库预约记录确定时刻，预报更正不追溯调整/取消已有预约。
- 预约改期：PATCH /reservations/{id}/schedule 只提交新的允许出发起止时刻，
  沿用原预约的吃水、安全富余与通航时长，以请求时最新预报重算可出发时段；
  在同港有效预约中排除该预约自身原占用，选出新范围内不与其他占用重叠的
  最早出发时刻（首尾相接允许）；仅有效预约可改，排程检查与更新在同一写
  事务内完成；无可行时刻返回 409 并保留原排程与允许范围；成功时保持预约
  ID、船名与创建时间不变。
- 预约水深复核：按查询时最新预报复算预约排定通航区间的水深，合并返回
  欠深时段及各段最大欠深；预报未覆盖整个通航区间时明确返回未覆盖时段，
  不判为安全；复核只读，不改动预约或原排程。
- 最大可通航吃水查询：提交安全富余、正数通航分钟数与允许出发起止时刻，
  按查询时最新潮高预报与线性插值规则，求允许范围内全程水深达标的最大
  非负吃水（按 0.01 米向下取整）；同一吃水多个可行时刻时返回最早出发及
  通航结束时刻；范围内无可行的 0 米吃水时返回 409。
- 预报影响预演：提交一批拟录入的潮高预报点，与库存预报合并（同一时刻按
  提交顺序后者覆盖前者）形成临时预报；基于一次读取的当前预报与该港口
  有效预约，分别用当前与临时预报按现有线性插值、水深复核规则判定每条
  预约排定通航区间是否全程达标（未被预报覆盖的区间判为不达标），仅返回
  达标状态变化的预约；预演只读，不写入预报、不改动预约。
- 实测潮位校准窗口：批量写入港口实测潮高（同港同一 UTC 时刻重复写入覆盖
  旧值）；校准窗口查询沿用吃水、安全富余、通航时长请求与窗口响应——以
  最新预报线性插值得观测时刻预报潮高，残差 = 实测 − 预报；仅采用预报
  覆盖范围内的观测，以预报首末时刻的零残差为边界锚点（同刻观测优先），
  锚点间线性插值残差并叠加原预报，把预报点与有效观测点时刻合并为折点后
  复用现有通航窗口计算；无有效观测时等同原窗口；查询只读，不改原预报和预约。
- 实测潮高异常点质检：GET /ports/{id}/observation-anomalies 对落在最新预报
  覆盖内的每条实测记录按预报线性插值求残差，以该时刻前后各 120 分钟内其他
  有效实测残差的中位数作局部基线（至少 3 个邻点才评估），局部 MAD 为邻点
  残差与基线差的绝对值中位数，异常阈值取 max(0.3 米, 3×1.4826×MAD)；仅
  返回残差偏离基线严格超过阈值的点，按时刻升序；预报不足两预报点或无足够
  邻点返回空列表；查询只读，不影响校准窗口及预约。
"""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import math

from fastapi import Depends, FastAPI, HTTPException
from sqlite3 import Connection

from .database import get_db, init_db, locked_reservation_tx, port_lock
from .schemas import (
    DepartureWindow,
    DepthAuditDeficit,
    DepthAuditResult,
    DepthAuditUncovered,
    ForecastImpactChange,
    ForecastImpactPreview,
    MaxDraftQuery,
    MaxDraftResult,
    ObservationAnomaly,
    PortCreate,
    PortOut,
    PortUpdate,
    ReservationCreate,
    ReservationOut,
    ReservationReschedule,
    TideForecastOut,
    TideForecastUpsert,
    TideObservationOut,
    TideObservationUpsert,
    WindowQuery,
    WindowQueryResult,
)
from .tides import (
    calibrated_forecast_points,
    departure_windows,
    depth_deficit_audit,
    earliest_feasible_departure,
    earliest_scheduled_departure,
    max_supported_required_depth,
    merge_forecast_points,
    observation_anomalies,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # 启动时初始化数据库（幂等）
    yield


app = FastAPI(
    title="港口潮汐通航窗口 API",
    version="1.6.0",
    lifespan=lifespan,
)


def _normalize(dt: datetime) -> datetime:
    """无时区信息按 UTC 处理，其余统一换算为 UTC。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_port_or_404(db: Connection, port_id: int):
    row = db.execute(
        "SELECT id, name, reference_depth FROM ports WHERE id = ?", (port_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"港口 {port_id} 不存在")
    return row


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ports", response_model=PortOut, status_code=201)
def create_port(payload: PortCreate, db: Connection = Depends(get_db)):
    """录入港口：名称 + 航道基准水深。"""
    cur = db.execute(
        "INSERT INTO ports (name, reference_depth) VALUES (?, ?)",
        (payload.name, payload.reference_depth),
    )
    return PortOut(id=cur.lastrowid, name=payload.name, reference_depth=payload.reference_depth)


@app.get("/ports", response_model=list[PortOut])
def list_ports(db: Connection = Depends(get_db)):
    rows = db.execute("SELECT id, name, reference_depth FROM ports ORDER BY id").fetchall()
    return [PortOut(**dict(r)) for r in rows]


@app.get("/ports/{port_id}", response_model=PortOut)
def get_port(port_id: int, db: Connection = Depends(get_db)):
    return PortOut(**dict(_get_port_or_404(db, port_id)))


@app.patch("/ports/{port_id}", response_model=PortOut)
def update_port(port_id: int, payload: PortUpdate, db: Connection = Depends(get_db)):
    """更正港口名称或航道基准水深。"""
    row = _get_port_or_404(db, port_id)
    name = payload.name if payload.name is not None else row["name"]
    depth = payload.reference_depth if payload.reference_depth is not None else row["reference_depth"]
    db.execute(
        "UPDATE ports SET name = ?, reference_depth = ? WHERE id = ?",
        (name, depth, port_id),
    )
    return PortOut(id=port_id, name=name, reference_depth=depth)


@app.put("/ports/{port_id}/tide-forecasts", response_model=list[TideForecastOut])
def upsert_tide_forecasts(
    port_id: int, payload: TideForecastUpsert, db: Connection = Depends(get_db)
):
    """批量录入潮高预报点；同一时刻重复提交即更正（覆盖旧值）。"""
    _get_port_or_404(db, port_id)
    for p in payload.points:
        t = _normalize(p.forecast_time).isoformat()
        db.execute(
            """
            INSERT INTO tide_forecasts (port_id, forecast_time, tide_height)
            VALUES (?, ?, ?)
            ON CONFLICT (port_id, forecast_time)
            DO UPDATE SET tide_height = excluded.tide_height
            """,
            (port_id, t, p.tide_height),
        )
    return _list_forecasts(db, port_id)


@app.get("/ports/{port_id}/tide-forecasts", response_model=list[TideForecastOut])
def get_tide_forecasts(port_id: int, db: Connection = Depends(get_db)):
    _get_port_or_404(db, port_id)
    return _list_forecasts(db, port_id)


def _list_forecasts(db: Connection, port_id: int) -> list[TideForecastOut]:
    rows = db.execute(
        "SELECT forecast_time, tide_height FROM tide_forecasts "
        "WHERE port_id = ? ORDER BY forecast_time",
        (port_id,),
    ).fetchall()
    return [
        TideForecastOut(forecast_time=datetime.fromisoformat(r["forecast_time"]), tide_height=r["tide_height"])
        for r in rows
    ]


def _forecast_points(db: Connection, port_id: int) -> list[tuple[datetime, float]]:
    """读取某港口最新预报点（(UTC 时刻, 潮高)，按时刻升序）。"""
    rows = db.execute(
        "SELECT forecast_time, tide_height FROM tide_forecasts "
        "WHERE port_id = ? ORDER BY forecast_time",
        (port_id,),
    ).fetchall()
    return [
        (datetime.fromisoformat(r["forecast_time"]), r["tide_height"])
        for r in rows
    ]


def _observation_points(db: Connection, port_id: int) -> list[tuple[datetime, float]]:
    """读取某港口实测潮高点（(UTC 时刻, 实测潮高)，按时刻升序）。"""
    rows = db.execute(
        "SELECT obs_time, tide_height FROM tide_observations "
        "WHERE port_id = ? ORDER BY obs_time",
        (port_id,),
    ).fetchall()
    return [
        (datetime.fromisoformat(r["obs_time"]), r["tide_height"])
        for r in rows
    ]


@app.put("/ports/{port_id}/tide-observations", response_model=list[TideObservationOut])
def upsert_tide_observations(
    port_id: int, payload: TideObservationUpsert, db: Connection = Depends(get_db)
):
    """批量写入实测潮高点；同港同一 UTC 时刻重复写入即覆盖旧值。

    实测点只在校准窗口查询中使用，不影响原预报与已有预约。
    """
    _get_port_or_404(db, port_id)
    for p in payload.points:
        t = _normalize(p.obs_time).isoformat()
        db.execute(
            """
            INSERT INTO tide_observations (port_id, obs_time, tide_height)
            VALUES (?, ?, ?)
            ON CONFLICT (port_id, obs_time)
            DO UPDATE SET tide_height = excluded.tide_height
            """,
            (port_id, t, p.tide_height),
        )
    return _list_observations(db, port_id)


@app.get("/ports/{port_id}/tide-observations", response_model=list[TideObservationOut])
def get_tide_observations(port_id: int, db: Connection = Depends(get_db)):
    _get_port_or_404(db, port_id)
    return _list_observations(db, port_id)


def _list_observations(db: Connection, port_id: int) -> list[TideObservationOut]:
    rows = db.execute(
        "SELECT obs_time, tide_height FROM tide_observations "
        "WHERE port_id = ? ORDER BY obs_time",
        (port_id,),
    ).fetchall()
    return [
        TideObservationOut(obs_time=datetime.fromisoformat(r["obs_time"]), tide_height=r["tide_height"])
        for r in rows
    ]


@app.post("/ports/{port_id}/windows", response_model=WindowQueryResult)
def query_windows(port_id: int, payload: WindowQuery, db: Connection = Depends(get_db)):
    """船舶提交吃水、安全富余、通航时长，返回全部可出发时段与最早出发时刻。

    每次查询都基于数据库中最新的预报点实时重算，预报点更正后立即生效。
    """
    port = _get_port_or_404(db, port_id)
    points = _forecast_points(db, port_id)

    required_depth = payload.draft + payload.safety_margin
    windows, earliest = departure_windows(
        points,
        reference_depth=port["reference_depth"],
        required_depth=required_depth,
        duration_seconds=payload.duration_minutes * 60,
    )
    return WindowQueryResult(
        port_id=port_id,
        required_depth=required_depth,
        duration_minutes=payload.duration_minutes,
        earliest_departure=earliest,
        windows=[DepartureWindow(start=s, end=e) for s, e in windows],
    )


@app.post("/ports/{port_id}/calibrated-windows", response_model=WindowQueryResult)
def query_calibrated_windows(
    port_id: int, payload: WindowQuery, db: Connection = Depends(get_db)
):
    """实测潮位校准窗口查询（只读，不改原预报和预约）。

    请求与响应与 POST /ports/{id}/windows 完全一致（吃水、安全富余、
    正数通航时长；返回全部可出发时段与最早出发时刻）。查询时用最新预报
    线性插值得到各观测时刻的预报潮高，残差 = 实测 − 预报；仅采用预报
    覆盖范围内的观测，以预报首末时刻的零残差为边界锚点（同刻观测优先），
    锚点间线性插值残差并叠加原预报；把预报点与有效观测点时刻合并为折点
    后复用现有通航窗口计算。无有效观测时结果等同原窗口。
    """
    port = _get_port_or_404(db, port_id)
    points = calibrated_forecast_points(
        _forecast_points(db, port_id),
        _observation_points(db, port_id),
    )

    required_depth = payload.draft + payload.safety_margin
    windows, earliest = departure_windows(
        points,
        reference_depth=port["reference_depth"],
        required_depth=required_depth,
        duration_seconds=payload.duration_minutes * 60,
    )
    return WindowQueryResult(
        port_id=port_id,
        required_depth=required_depth,
        duration_minutes=payload.duration_minutes,
        earliest_departure=earliest,
        windows=[DepartureWindow(start=s, end=e) for s, e in windows],
    )


@app.get(
    "/ports/{port_id}/observation-anomalies",
    response_model=list[ObservationAnomaly],
)
def get_observation_anomalies(port_id: int, db: Connection = Depends(get_db)):
    """实测潮高异常点质检（只读，不影响校准窗口及预约）。

    对落在最新预报覆盖范围 [首预报时刻, 末预报时刻] 内的每条实测记录，
    按相邻预报点线性插值求预报潮高，残差 = 实测 − 预报；以该时刻前后
    各 120 分钟内其他有效实测残差的中位数作局部基线（至少 3 个邻点才
    评估）；局部 MAD 为邻点残差与基线差的绝对值中位数，异常阈值取
    max(0.3 米, 3×1.4826×MAD)。仅返回残差偏离基线严格超过阈值的点，
    含时刻、实测潮高、残差、基线与偏离量，按时刻升序。预报不足 2 个点
    或无足够邻点时返回空列表。
    """
    _get_port_or_404(db, port_id)
    rows = observation_anomalies(
        _forecast_points(db, port_id),
        _observation_points(db, port_id),
    )
    return [
        ObservationAnomaly(
            obs_time=t,
            tide_height=observed,
            residual=residual,
            baseline=baseline,
            deviation=deviation,
        )
        for t, observed, residual, baseline, deviation in rows
    ]


@app.post("/ports/{port_id}/max-draft", response_model=MaxDraftResult)
def query_max_draft(port_id: int, payload: MaxDraftQuery, db: Connection = Depends(get_db)):
    """最大可通航吃水查询（只读，不写库）。

    提交安全富余、正数通航分钟数与允许出发起止时刻，按查询时最新潮高预报
    与相邻点线性插值规则，求允许范围内全程水深（基准水深 + 潮高）达标的
    最大非负吃水，结果按 0.01 米向下取整；同一吃水有多个可行时刻时返回
    最早出发及通航结束时刻。只在预报覆盖的连续时段内求解；允许范围内连
    0 米吃水都不可行时返回 409。
    """
    port = _get_port_or_404(db, port_id)
    points = _forecast_points(db, port_id)
    allowed_from = _normalize(payload.allowed_departure_from)
    allowed_to = _normalize(payload.allowed_departure_to)
    duration_seconds = payload.duration_minutes * 60

    supported, feasible = max_supported_required_depth(
        points,
        reference_depth=port["reference_depth"],
        min_required=payload.safety_margin,  # 对应 0 米吃水
        duration_seconds=duration_seconds,
        allowed_start=allowed_from,
        allowed_end=allowed_to,
    )
    if feasible is None:
        raise HTTPException(
            status_code=409,
            detail="允许出发范围内无可行的 0 米吃水（预报未覆盖或水深不足）",
        )

    # 按 0.01 米向下取整；微小 epsilon 抵消二分与插值的浮点噪声，
    # 保证真实值恰在厘米刻度时不会被误向下取一档
    max_draft = round(
        math.floor((supported - payload.safety_margin) * 100 + 1e-9) / 100, 2
    )
    required_depth = max_draft + payload.safety_margin
    departure = earliest_feasible_departure(
        points,
        reference_depth=port["reference_depth"],
        required_depth=required_depth,
        duration_seconds=duration_seconds,
        allowed_start=allowed_from,
        allowed_end=allowed_to,
    )
    if departure is None:  # 浮点取整边界兜底：退一档后必然可行
        max_draft = round(max_draft - 0.01, 2)
        required_depth = max_draft + payload.safety_margin
        departure = earliest_feasible_departure(
            points,
            reference_depth=port["reference_depth"],
            required_depth=required_depth,
            duration_seconds=duration_seconds,
            allowed_start=allowed_from,
            allowed_end=allowed_to,
        )
    return MaxDraftResult(
        port_id=port_id,
        safety_margin=payload.safety_margin,
        duration_minutes=payload.duration_minutes,
        allowed_departure_from=allowed_from,
        allowed_departure_to=allowed_to,
        max_draft=max_draft,
        required_depth=required_depth,
        departure_start=departure,
        departure_end=departure + timedelta(seconds=duration_seconds),
    )


# ---------------------------------------------------------------------------
# 单向航道预约
# ---------------------------------------------------------------------------

# 有效预约的状态值；已取消预约不再占用航道，也不在有效列表中
_STATUS_ACTIVE = "active"
_STATUS_CANCELLED = "cancelled"


def _reservation_from_row(row) -> ReservationOut:
    return ReservationOut(
        id=row["id"],
        port_id=row["port_id"],
        ship_name=row["ship_name"],
        draft=row["draft"],
        safety_margin=row["safety_margin"],
        duration_minutes=row["duration_minutes"],
        allowed_departure_from=datetime.fromisoformat(row["allowed_departure_from"]),
        allowed_departure_to=datetime.fromisoformat(row["allowed_departure_to"]),
        departure_start=datetime.fromisoformat(row["departure_start"]),
        departure_end=datetime.fromisoformat(row["departure_end"]),
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        canceled_at=(
            datetime.fromisoformat(row["canceled_at"]) if row["canceled_at"] else None
        ),
    )


def _active_blockers(
    db: Connection, port_id: int, exclude_id: int | None = None
) -> list[tuple[datetime, datetime]]:
    """读取该港口全部有效（未取消）预约的通航占用区间。

    exclude_id 用于改期时排除该预约自身的原占用（改期不与自己冲突）。
    """
    sql = (
        "SELECT departure_start, departure_end FROM reservations "
        "WHERE port_id = ? AND status = ?"
    )
    params: list = [port_id, _STATUS_ACTIVE]
    if exclude_id is not None:
        sql += " AND id != ?"
        params.append(exclude_id)
    sql += " ORDER BY departure_start"
    rows = db.execute(sql, params).fetchall()
    return [
        (
            datetime.fromisoformat(r["departure_start"]),
            datetime.fromisoformat(r["departure_end"]),
        )
        for r in rows
    ]


@app.post("/ports/{port_id}/reservations", response_model=ReservationOut, status_code=201)
def create_reservation(port_id: int, payload: ReservationCreate):
    """创建单向航道预约。

    基于该港口**最新预报**求可出发时段，与请求的允许出发范围取交集，
    再在其中安排**最早**、整个通航过程不与现有有效预约重叠的出发时刻
    （两船首尾相接允许）。无可用时刻返回 409 且不写入。

    同港口并发创建按港口串行化，并在 SQLite 立即写事务内原子地
    “检查 + 落库”，避免双重占用。
    """
    allowed_from = _normalize(payload.allowed_departure_from)
    allowed_to = _normalize(payload.allowed_departure_to)
    duration = timedelta(minutes=payload.duration_minutes)

    # 同港先在进程内排队，避免并发写事务互相撞锁
    with port_lock(port_id):
        with locked_reservation_tx() as db:
            port = _get_port_or_404(db, port_id)
            points = _forecast_points(db, port_id)
            required_depth = payload.draft + payload.safety_margin
            tide_windows, _ = departure_windows(
                points,
                reference_depth=port["reference_depth"],
                required_depth=required_depth,
                duration_seconds=payload.duration_minutes * 60,
            )
            blockers = _active_blockers(db, port_id)
            departure = earliest_scheduled_departure(
                tide_windows,
                duration_seconds=payload.duration_minutes * 60,
                allowed_start=allowed_from,
                allowed_end=allowed_to,
                blockers=blockers,
            )
            if departure is None:
                # 无可用时刻：事务回滚，不写入任何数据
                raise HTTPException(
                    status_code=409,
                    detail="允许出发范围内无可用时刻（潮高不足或与现有预约冲突）",
                )
            arrival = departure + duration
            now = datetime.now(timezone.utc)
            cur = db.execute(
                """
                INSERT INTO reservations (
                    port_id, ship_name, draft, safety_margin, duration_minutes,
                    allowed_departure_from, allowed_departure_to,
                    departure_start, departure_end, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    port_id,
                    payload.ship_name,
                    payload.draft,
                    payload.safety_margin,
                    payload.duration_minutes,
                    allowed_from.isoformat(),
                    allowed_to.isoformat(),
                    departure.isoformat(),
                    arrival.isoformat(),
                    _STATUS_ACTIVE,
                    now.isoformat(),
                ),
            )
            row = db.execute(
                "SELECT * FROM reservations WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            return _reservation_from_row(row)


@app.get("/ports/{port_id}/reservations", response_model=list[ReservationOut])
def list_reservations(port_id: int, db: Connection = Depends(get_db)):
    """按港口列出全部**有效**（未取消）预约，按出发时刻升序。"""
    _get_port_or_404(db, port_id)
    rows = db.execute(
        "SELECT * FROM reservations WHERE port_id = ? AND status = ? "
        "ORDER BY departure_start",
        (port_id, _STATUS_ACTIVE),
    ).fetchall()
    return [_reservation_from_row(r) for r in rows]


@app.delete("/reservations/{reservation_id}", response_model=ReservationOut)
def cancel_reservation(reservation_id: int, db: Connection = Depends(get_db)):
    """取消预约；不存在返回 404，已取消返回 409。

    取消后该预约立即释放航道占用；预报点更正不会追溯影响（取消）已有预约。
    """
    row = db.execute(
        "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"预约 {reservation_id} 不存在")
    if row["status"] != _STATUS_ACTIVE:
        raise HTTPException(status_code=409, detail=f"预约 {reservation_id} 已取消")
    canceled_at = datetime.now(timezone.utc)
    db.execute(
        "UPDATE reservations SET status = ?, canceled_at = ? WHERE id = ?",
        (_STATUS_CANCELLED, canceled_at.isoformat(), reservation_id),
    )
    row = db.execute(
        "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
    ).fetchone()
    return _reservation_from_row(row)


@app.patch("/reservations/{reservation_id}/schedule", response_model=ReservationOut)
def reschedule_reservation(
    reservation_id: int,
    payload: ReservationReschedule,
    db: Connection = Depends(get_db),
):
    """预约改期：只提交新的允许出发起止时刻，其余沿用原预约。

    沿用原预约的吃水、安全富余与通航时长，以**请求时最新预报**重算潮汐
    可出发时段；在同港有效预约中**排除该预约自身的原占用**，选出新允许
    范围内不与其他占用重叠的**最早**出发时刻（首尾相接允许）。仅有效
    预约可改期（已取消返回 409）；新范围内无可行时刻返回 409 并**保留
    原排程与允许范围**。成功时预约 ID、船名与创建时间不变，仅更新允许
    范围与通航起止。

    与创建预约相同：同港先进程内串行化，再在 SQLite 立即写事务内原子地
    “检查 + 更新”，并发改期/新建不会产生双重占用。
    """
    allowed_from = _normalize(payload.allowed_departure_from)
    allowed_to = _normalize(payload.allowed_departure_to)

    # 先读出港口 id 用于进程内串行化（预约的 port_id 不会变化）
    row = db.execute(
        "SELECT port_id FROM reservations WHERE id = ?", (reservation_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"预约 {reservation_id} 不存在")
    port_id = row["port_id"]

    with port_lock(port_id):
        with locked_reservation_tx() as tx:
            row = tx.execute(
                "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            if row is None:  # 防御：事务内复核预约仍在
                raise HTTPException(
                    status_code=404, detail=f"预约 {reservation_id} 不存在"
                )
            if row["status"] != _STATUS_ACTIVE:
                raise HTTPException(
                    status_code=409,
                    detail=f"预约 {reservation_id} 已取消，不能改期",
                )
            port = _get_port_or_404(tx, port_id)
            points = _forecast_points(tx, port_id)
            required_depth = row["draft"] + row["safety_margin"]
            duration_seconds = row["duration_minutes"] * 60
            tide_windows, _ = departure_windows(
                points,
                reference_depth=port["reference_depth"],
                required_depth=required_depth,
                duration_seconds=duration_seconds,
            )
            # 排除该预约自身原占用，只避让其他有效预约
            blockers = _active_blockers(tx, port_id, exclude_id=reservation_id)
            departure = earliest_scheduled_departure(
                tide_windows,
                duration_seconds=duration_seconds,
                allowed_start=allowed_from,
                allowed_end=allowed_to,
                blockers=blockers,
            )
            if departure is None:
                # 无可行时刻：事务回滚，保留原排程与允许范围
                raise HTTPException(
                    status_code=409,
                    detail="新允许出发范围内无可用时刻（潮高不足或与现有预约冲突），"
                    "已保留原排程",
                )
            arrival = departure + timedelta(minutes=row["duration_minutes"])
            tx.execute(
                """
                UPDATE reservations
                SET allowed_departure_from = ?, allowed_departure_to = ?,
                    departure_start = ?, departure_end = ?
                WHERE id = ?
                """,
                (
                    allowed_from.isoformat(),
                    allowed_to.isoformat(),
                    departure.isoformat(),
                    arrival.isoformat(),
                    reservation_id,
                ),
            )
            row = tx.execute(
                "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            return _reservation_from_row(row)


@app.get("/reservations/{reservation_id}/depth-audit", response_model=DepthAuditResult)
def audit_reservation_depth(reservation_id: int, db: Connection = Depends(get_db)):
    """预约水深复核（只读，不改动预约或原排程）。

    读取预约排定的通航起止、吃水与安全富余，按该港口**查询时最新预报**
    相邻点线性插值复算水深：合并返回通航区间内水深低于所需水深的连续
    时段及各段最大欠深；预报首末点未覆盖整个通航区间时，明确返回未覆盖
    时段且 `sufficient` 一律为 false（不判为安全）。
    """
    row = db.execute(
        "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"预约 {reservation_id} 不存在")
    port = _get_port_or_404(db, row["port_id"])
    points = _forecast_points(db, row["port_id"])

    required_depth = row["draft"] + row["safety_margin"]
    departure_start = datetime.fromisoformat(row["departure_start"])
    departure_end = datetime.fromisoformat(row["departure_end"])
    deficits, uncovered = depth_deficit_audit(
        points,
        reference_depth=port["reference_depth"],
        required_depth=required_depth,
        start=departure_start,
        end=departure_end,
    )
    return DepthAuditResult(
        reservation_id=row["id"],
        port_id=row["port_id"],
        ship_name=row["ship_name"],
        status=row["status"],
        departure_start=departure_start,
        departure_end=departure_end,
        draft=row["draft"],
        safety_margin=row["safety_margin"],
        required_depth=required_depth,
        coverage_complete=not uncovered,
        uncovered=[DepthAuditUncovered(start=s, end=e) for s, e in uncovered],
        deficits=[
            DepthAuditDeficit(start=s, end=e, max_deficit=m) for s, e, m in deficits
        ],
        sufficient=not uncovered and not deficits,
    )


# ---------------------------------------------------------------------------
# 预报影响预演
# ---------------------------------------------------------------------------


@app.post(
    "/ports/{port_id}/forecast-impact-preview",
    response_model=list[ForecastImpactChange],
)
def preview_forecast_impact(
    port_id: int, payload: ForecastImpactPreview, db: Connection = Depends(get_db)
):
    """预报影响预演（只读：不写入预报、不改动预约）。

    提交一批拟录入的潮高预报点，按 UTC 规则与库存预报合并（同一时刻按
    提交顺序后者覆盖前者）形成临时预报；基于一次读取的当前预报与该港口
    有效预约，对每条预约排定的通航区间分别用当前与临时预报按现有线性
    插值、水深复核规则判定是否全程达标（未被预报覆盖的区间判为不达标）。
    仅返回达标状态发生变化的预约，按出发时刻升序；无变化时返回空列表。
    """
    port = _get_port_or_404(db, port_id)
    # 一次读取：当前预报点 + 该港口全部有效预约
    current_points = _forecast_points(db, port_id)
    rows = db.execute(
        "SELECT id, draft, safety_margin, departure_start, departure_end "
        "FROM reservations WHERE port_id = ? AND status = ? "
        "ORDER BY departure_start, id",
        (port_id, _STATUS_ACTIVE),
    ).fetchall()

    submitted = [(_normalize(p.forecast_time), p.tide_height) for p in payload.points]
    preview_points = merge_forecast_points(current_points, submitted)

    changes: list[ForecastImpactChange] = []
    for row in rows:
        required_depth = row["draft"] + row["safety_margin"]
        start = datetime.fromisoformat(row["departure_start"])
        end = datetime.fromisoformat(row["departure_end"])
        cur_deficits, cur_uncovered = depth_deficit_audit(
            current_points, port["reference_depth"], required_depth, start, end
        )
        new_deficits, new_uncovered = depth_deficit_audit(
            preview_points, port["reference_depth"], required_depth, start, end
        )
        current_ok = not cur_deficits and not cur_uncovered
        preview_ok = not new_deficits and not new_uncovered
        if current_ok != preview_ok:
            changes.append(
                ForecastImpactChange(
                    reservation_id=row["id"],
                    current_sufficient=current_ok,
                    preview_sufficient=preview_ok,
                )
            )
    return changes
