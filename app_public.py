from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from app import (
    build_auto_trade_history_df,
    load_candidate_snapshot_payload,
    load_mock_trade_state_payload,
    load_strategy_state_payload,
    run_auto_trading_cycle,
    safe_float,
)
from services.trading_service import build_active_positions_frame, has_mock_trading_config
from services.stock_service import fetch_daily_stock_data
from utils.config import get_settings


PUBLIC_RUNTIME_STATE_PATH = Path(__file__).resolve().parent / "data" / "public_runtime_state.json"
KST = ZoneInfo("Asia/Seoul")


def env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_text(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def load_public_runtime_state() -> dict[str, object]:
    if not PUBLIC_RUNTIME_STATE_PATH.exists():
        return {}

    try:
        payload = json.loads(PUBLIC_RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_public_runtime_state(state: dict[str, object]) -> None:
    PUBLIC_RUNTIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_RUNTIME_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_public_cycle_config() -> dict[str, object]:
    return {
        "enabled": env_bool("PUBLIC_APP_ENABLE_AUTO_CYCLE", True),
        "refresh_seconds": max(3, env_int("PUBLIC_APP_REFRESH_SECONDS", 15)),
        "min_cycle_interval_seconds": max(10, env_int("PUBLIC_APP_MIN_CYCLE_INTERVAL_SECONDS", 30)),
        "spotlight_symbol": env_text("PUBLIC_APP_SPOTLIGHT_SYMBOL", ""),
        "selected_market": str(os.getenv("PUBLIC_APP_SELECTED_MARKET", "전체")).strip() or "전체",
        "scan_pool_size": max(20, env_int("PUBLIC_APP_SCAN_POOL_SIZE", 60)),
        "article_limit": max(10, env_int("PUBLIC_APP_ARTICLE_LIMIT", 20)),
        "news_days": max(3, env_int("PUBLIC_APP_NEWS_DAYS", 7)),
        "recommendation_limit": max(3, env_int("PUBLIC_APP_RECOMMENDATION_LIMIT", 8)),
        "use_realtime": env_bool("PUBLIC_APP_USE_REALTIME", True),
        "max_positions": max(1, env_int("PUBLIC_APP_MAX_POSITIONS", 3)),
        "max_daily_buys": max(0, env_int("PUBLIC_APP_MAX_DAILY_BUYS", 0)),
        "budget_per_trade": max(10000.0, env_float("PUBLIC_APP_BUDGET_PER_TRADE", 300000.0)),
        "min_final_score": env_float("PUBLIC_APP_MIN_FINAL_SCORE", 55.0),
        "min_up_probability": env_float("PUBLIC_APP_MIN_UP_PROBABILITY", 60.0),
        "min_expected_return": env_float("PUBLIC_APP_MIN_EXPECTED_RETURN", 1.0),
        "min_sentiment": env_float("PUBLIC_APP_MIN_SENTIMENT", 0.05),
        "max_realtime_change": env_float("PUBLIC_APP_MAX_REALTIME_CHANGE", 7.0),
        "target_profit_factor": env_float("PUBLIC_APP_TARGET_PROFIT_FACTOR", 0.6),
        "min_target_profit": env_float("PUBLIC_APP_MIN_TARGET_PROFIT", 1.5),
        "max_target_profit": env_float("PUBLIC_APP_MAX_TARGET_PROFIT", 4.0),
        "stop_loss_pct": env_float("PUBLIC_APP_STOP_LOSS_PCT", 2.0),
        "max_hold_hours": env_float("PUBLIC_APP_MAX_HOLD_HOURS", 24.0),
        "scan_interval_seconds": max(30, env_int("PUBLIC_APP_SCAN_INTERVAL_SECONDS", 180)),
        "buy_interval_seconds": max(5, env_int("PUBLIC_APP_BUY_INTERVAL_SECONDS", 30)),
        "monitor_interval_seconds": max(5, env_int("PUBLIC_APP_MONITOR_INTERVAL_SECONDS", 15)),
    }


@st.cache_data(ttl=900)
def load_public_stock_series(symbol: str, kis_app_key: str, kis_app_secret: str) -> pd.DataFrame:
    return fetch_daily_stock_data(symbol=symbol, kis_app_key=kis_app_key, kis_app_secret=kis_app_secret)


def maybe_run_public_cycle(settings, cycle_config: dict[str, object]) -> tuple[bool, str]:
    if not bool(cycle_config["enabled"]):
        return False, "공개 자동 운용이 비활성화되어 있습니다."

    if not has_mock_trading_config(settings):
        return False, "모의투자 계좌 설정이 없어 공개 자동 운용을 건너뜁니다."

    runtime_state = load_public_runtime_state()
    now_ts = time.time()
    last_cycle_ts = safe_float(runtime_state.get("last_cycle_ts"), 0.0)
    min_cycle_interval = safe_float(cycle_config["min_cycle_interval_seconds"], 30.0)

    if now_ts - last_cycle_ts < min_cycle_interval:
        last_cycle_at = str(runtime_state.get("last_cycle_at", "")).strip()
        if last_cycle_at:
            return False, f"최근 자동 운용 실행 시각: {last_cycle_at}"
        return False, "최근 주기에 이미 자동 운용이 실행되었습니다."

    runtime_state["last_cycle_ts"] = now_ts
    runtime_state["last_cycle_at"] = datetime.now(KST).isoformat(timespec="seconds")
    save_public_runtime_state(runtime_state)

    run_auto_trading_cycle(
        settings,
        selected_market=str(cycle_config["selected_market"]),
        scan_pool_size=int(cycle_config["scan_pool_size"]),
        article_limit=int(cycle_config["article_limit"]),
        news_days=int(cycle_config["news_days"]),
        recommendation_limit=int(cycle_config["recommendation_limit"]),
        use_realtime=bool(cycle_config["use_realtime"]),
        max_positions=int(cycle_config["max_positions"]),
        max_daily_buys=int(cycle_config["max_daily_buys"]),
        budget_per_trade=float(cycle_config["budget_per_trade"]),
        min_final_score=float(cycle_config["min_final_score"]),
        min_up_probability=float(cycle_config["min_up_probability"]),
        min_expected_return=float(cycle_config["min_expected_return"]),
        min_sentiment=float(cycle_config["min_sentiment"]),
        max_realtime_change=float(cycle_config["max_realtime_change"]),
        target_profit_factor=float(cycle_config["target_profit_factor"]),
        min_target_profit=float(cycle_config["min_target_profit"]),
        max_target_profit=float(cycle_config["max_target_profit"]),
        stop_loss_pct=float(cycle_config["stop_loss_pct"]),
        max_hold_hours=float(cycle_config["max_hold_hours"]),
        scan_interval_seconds=int(cycle_config["scan_interval_seconds"]),
        buy_interval_seconds=int(cycle_config["buy_interval_seconds"]),
        monitor_interval_seconds=int(cycle_config["monitor_interval_seconds"]),
    )
    return True, f"자동 운용 실행 완료: {runtime_state['last_cycle_at']}"


def build_public_summary(active_positions_df: pd.DataFrame, trade_history_df: pd.DataFrame) -> dict[str, float]:
    holding_count = float(len(active_positions_df))
    holding_quantity = float(active_positions_df["quantity"].sum()) if not active_positions_df.empty else 0.0
    total_buy_amount = (
        float((active_positions_df["entry_price"] * active_positions_df["quantity"]).sum())
        if not active_positions_df.empty else 0.0
    )
    total_eval_amount = (
        float((active_positions_df["current_price"] * active_positions_df["quantity"]).sum())
        if not active_positions_df.empty else 0.0
    )
    unrealized_pnl = total_eval_amount - total_buy_amount
    realized_sell_df = trade_history_df[trade_history_df["side"] == "sell"] if not trade_history_df.empty else pd.DataFrame()
    realized_pnl = float(realized_sell_df["realized_pnl"].sum()) if not realized_sell_df.empty else 0.0
    total_profit = unrealized_pnl + realized_pnl
    cumulative_return_pct = (total_profit / total_buy_amount * 100) if total_buy_amount > 0 else 0.0
    return {
        "holding_count": holding_count,
        "holding_quantity": holding_quantity,
        "total_buy_amount": total_buy_amount,
        "total_eval_amount": total_eval_amount,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "total_profit": total_profit,
        "cumulative_return_pct": cumulative_return_pct,
    }


def render_public_summary_cards(summary: dict[str, float]) -> None:
    top_cols = st.columns(4)
    top_cols[0].metric("보유 종목 수", f"{int(summary['holding_count']):,}")
    top_cols[1].metric("보유 주식 수", f"{int(summary['holding_quantity']):,}")
    top_cols[2].metric("총 매수금액", f"₩{summary['total_buy_amount']:,.0f}")
    top_cols[3].metric("총 평가금액", f"₩{summary['total_eval_amount']:,.0f}")

    bottom_cols = st.columns(4)
    bottom_cols[0].metric("미실현 손익", f"₩{summary['unrealized_pnl']:,.0f}")
    bottom_cols[1].metric("실현 손익", f"₩{summary['realized_pnl']:,.0f}")
    bottom_cols[2].metric("누적 손익", f"₩{summary['total_profit']:,.0f}")
    bottom_cols[3].metric("누적 수익률", f"{summary['cumulative_return_pct']:.2f}%")


def render_public_positions(active_positions_df: pd.DataFrame) -> tuple[str, str]:
    if active_positions_df.empty:
        st.info("현재 공개 포트폴리오에 보유 중인 종목이 없습니다.")
        return "", ""

    display_df = active_positions_df.copy()
    display_df["총 매수금액"] = display_df["quantity"] * display_df["entry_price"]
    display_df["현재 평가금액"] = display_df["quantity"] * display_df["current_price"]
    display_df["평가손익"] = display_df["현재 평가금액"] - display_df["총 매수금액"]
    display_df = display_df.sort_values(["현재 평가금액", "current_return_pct"], ascending=[False, False]).reset_index(drop=True)

    table_df = display_df.rename(
        columns={
            "name": "종목명",
            "symbol": "종목코드",
            "market": "시장",
            "quantity": "보유수량",
            "entry_price": "매수가",
            "current_price": "현재가",
            "current_return_pct": "현재 수익률(%)",
            "expected_return_pct": "예상 상승률(%)",
            "target_profit_pct": "자동매도 목표(%)",
            "created_at": "진입 시각",
        }
    )[
        [
            "종목명",
            "종목코드",
            "시장",
            "보유수량",
            "매수가",
            "총 매수금액",
            "현재가",
            "현재 평가금액",
            "평가손익",
            "현재 수익률(%)",
            "예상 상승률(%)",
            "자동매도 목표(%)",
            "진입 시각",
        ]
    ].copy()
    st.caption("표에서 보유 종목 한 줄을 선택하면 아래에 해당 종목의 주가 변동 그래프가 표시됩니다.")

    selection_columns = [
        "종목명",
        "종목코드",
        "시장",
        "보유수량",
        "현재가",
        "평가손익",
        "현재 수익률(%)",
        "예상 상승률(%)",
    ]
    selection_df = table_df[selection_columns].copy()

    selected_symbol = ""
    selected_name = ""
    dataframe_kwargs = {
        "use_container_width": True,
        "hide_index": True,
        "column_config": {
            "보유수량": st.column_config.NumberColumn("보유수량", format="%d"),
            "현재가": st.column_config.NumberColumn("현재가", format="₩%.0f"),
            "평가손익": st.column_config.NumberColumn("평가손익", format="₩%.0f"),
            "현재 수익률(%)": st.column_config.NumberColumn("현재 수익률(%)", format="%.2f"),
            "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
        },
    }

    try:
        selection_event = st.dataframe(
            selection_df,
            on_select="rerun",
            selection_mode="single-row",
            key="public_positions_selector",
            **dataframe_kwargs,
        )
        selected_rows = []
        if isinstance(selection_event, dict):
            selected_rows = selection_event.get("selection", {}).get("rows", [])
        else:
            selected_rows = getattr(selection_event.selection, "rows", [])
        if selected_rows:
            selected_index = int(selected_rows[0])
            selected_symbol = str(selection_df.iloc[selected_index]["종목코드"])
            selected_name = str(selection_df.iloc[selected_index]["종목명"])
    except TypeError:
        st.dataframe(selection_df, **dataframe_kwargs)

    if not selected_symbol:
        selected_symbol = str(selection_df.iloc[0]["종목코드"])
        selected_name = str(selection_df.iloc[0]["종목명"])

    return selected_symbol, selected_name


def render_public_trade_history(trade_history_df: pd.DataFrame) -> None:
    if trade_history_df.empty:
        st.info("아직 공개할 자동 매매 이력이 없습니다.")
        return

    display_df = trade_history_df.copy()
    display_df["ordered_at"] = pd.to_datetime(display_df["ordered_at"], errors="coerce")
    display_df["ordered_at"] = display_df["ordered_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["side"] = display_df["side"].map({"buy": "매수", "sell": "매도"}).fillna(display_df["side"])
    display_df = display_df.rename(
        columns={
            "ordered_at": "체결 시각",
            "side": "구분",
            "symbol": "종목코드",
            "name": "종목명",
            "quantity": "수량",
            "price": "체결가",
            "status": "상태",
            "realized_pnl": "실현손익",
            "realized_pnl_pct": "실현손익률(%)",
        }
    )

    st.dataframe(
        display_df.head(30),
        use_container_width=True,
        hide_index=True,
        column_config={
            "수량": st.column_config.NumberColumn("수량", format="%d"),
            "체결가": st.column_config.NumberColumn("체결가", format="₩%.0f"),
            "실현손익": st.column_config.NumberColumn("실현손익", format="₩%.0f"),
            "실현손익률(%)": st.column_config.NumberColumn("실현손익률(%)", format="%.2f"),
        },
    )


def choose_spotlight_symbol(
    cycle_config: dict[str, object],
    active_positions_df: pd.DataFrame,
    snapshot_payload: dict[str, object],
) -> str:
    configured_symbol = str(cycle_config.get("spotlight_symbol", "")).strip()
    if configured_symbol:
        return configured_symbol

    if not active_positions_df.empty and "symbol" in active_positions_df.columns:
        return str(active_positions_df.iloc[0]["symbol"]).strip()

    candidate_rows = snapshot_payload.get("candidates", [])
    if isinstance(candidate_rows, list):
        for row in candidate_rows:
            symbol = str((row or {}).get("symbol", "")).strip()
            if symbol:
                return symbol

    return "005930"


def build_public_movement_table(
    settings,
    active_positions_df: pd.DataFrame,
    snapshot_payload: dict[str, object],
) -> pd.DataFrame:
    symbol_meta: dict[str, dict[str, object]] = {}

    if not active_positions_df.empty:
        for _, row in active_positions_df.iterrows():
            symbol = str(row.get("symbol", "")).strip()
            if not symbol:
                continue
            symbol_meta[symbol] = {
                "name": str(row.get("name", "")),
                "market": str(row.get("market", "")),
                "source": "보유 종목",
                "current_price_hint": safe_float(row.get("current_price"), 0.0),
            }

    candidate_rows = snapshot_payload.get("candidates", [])
    if isinstance(candidate_rows, list):
        for row in candidate_rows[:12]:
            symbol = str((row or {}).get("symbol", "")).strip()
            if not symbol:
                continue
            existing = symbol_meta.get(symbol, {})
            symbol_meta[symbol] = {
                "name": str((row or {}).get("name", existing.get("name", ""))),
                "market": str((row or {}).get("market", existing.get("market", ""))),
                "source": existing.get("source", "추천 후보"),
                "current_price_hint": safe_float(existing.get("current_price_hint"), 0.0),
            }

    rows: list[dict[str, object]] = []
    for symbol, meta in list(symbol_meta.items())[:12]:
        try:
            stock_df = load_public_stock_series(
                symbol,
                getattr(settings, "kis_app_key", ""),
                getattr(settings, "kis_app_secret", ""),
            )
        except Exception:
            continue

        if stock_df.empty or "close" not in stock_df.columns:
            continue

        latest_close = safe_float(stock_df["close"].iloc[-1], 0.0)
        previous_close = safe_float(stock_df["close"].iloc[-2], latest_close) if len(stock_df) > 1 else latest_close
        weekly_base = safe_float(stock_df["close"].iloc[-6], previous_close) if len(stock_df) > 5 else previous_close
        monthly_base = safe_float(stock_df["close"].iloc[-21], previous_close) if len(stock_df) > 20 else previous_close
        current_price = safe_float(meta.get("current_price_hint"), latest_close) or latest_close
        daily_change_pct = ((current_price / previous_close) - 1) * 100 if previous_close else 0.0
        weekly_change_pct = ((latest_close / weekly_base) - 1) * 100 if weekly_base else 0.0
        monthly_change_pct = ((latest_close / monthly_base) - 1) * 100 if monthly_base else 0.0

        rows.append(
            {
                "종목명": str(meta.get("name", symbol)),
                "종목코드": symbol,
                "시장": str(meta.get("market", "")),
                "구분": str(meta.get("source", "")),
                "현재가": current_price,
                "전일 대비(%)": daily_change_pct,
                "5일 변화(%)": weekly_change_pct,
                "20일 변화(%)": monthly_change_pct,
            }
        )

    movement_df = pd.DataFrame(rows)
    if movement_df.empty:
        return movement_df
    return movement_df.sort_values(["전일 대비(%)", "20일 변화(%)"], ascending=[False, False]).reset_index(drop=True)


def render_public_movement_table(movement_df: pd.DataFrame) -> None:
    if movement_df.empty:
        st.info("주가 변동 요약 표를 생성할 데이터가 아직 없습니다.")
        return

    def _movement_style(value: object) -> str:
        number = safe_float(value)
        if number > 0:
            return "color: #15803d; font-weight: 700;"
        if number < 0:
            return "color: #b91c1c; font-weight: 700;"
        return "color: #475569;"

    styler = movement_df.style.format(
        {
            "현재가": "₩{:,.0f}",
            "전일 대비(%)": "{:,.2f}",
            "5일 변화(%)": "{:,.2f}",
            "20일 변화(%)": "{:,.2f}",
        }
    )
    if hasattr(styler, "map"):
        styler = styler.map(_movement_style, subset=["전일 대비(%)", "5일 변화(%)", "20일 변화(%)"])
    else:
        styler = styler.applymap(_movement_style, subset=["전일 대비(%)", "5일 변화(%)", "20일 변화(%)"])
    st.dataframe(styler, use_container_width=True, hide_index=True)


def render_public_spotlight_chart(settings, symbol: str) -> None:
    if not symbol:
        st.info("대표 종목 그래프를 표시할 종목이 없습니다.")
        return

    try:
        stock_df = load_public_stock_series(
            symbol,
            getattr(settings, "kis_app_key", ""),
            getattr(settings, "kis_app_secret", ""),
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"대표 종목 그래프를 불러오지 못했습니다: {exc}")
        return

    if stock_df.empty:
        st.info("대표 종목 그래프를 표시할 데이터가 없습니다.")
        return

    stock_df = stock_df.copy().sort_values("date").reset_index(drop=True)
    chart_df = stock_df.tail(90).copy()
    latest_close = safe_float(chart_df["close"].iloc[-1], 0.0)
    previous_close = safe_float(chart_df["close"].iloc[-2], latest_close) if len(chart_df) > 1 else latest_close
    daily_change_pct = ((latest_close / previous_close) - 1) * 100 if previous_close else 0.0
    low_90 = safe_float(chart_df["low"].min(), latest_close)
    high_90 = safe_float(chart_df["high"].max(), latest_close)

    metric_cols = st.columns(4)
    metric_cols[0].metric("대표 종목", symbol)
    metric_cols[1].metric("현재 종가", f"₩{latest_close:,.0f}")
    metric_cols[2].metric("전일 대비", f"{daily_change_pct:.2f}%")
    metric_cols[3].metric("90일 범위", f"₩{low_90:,.0f} ~ ₩{high_90:,.0f}")

    fig = px.line(
        chart_df,
        x="date",
        y="close",
        markers=True,
        title=f"{symbol} 최근 90일 종가 추이",
    )
    fig.update_traces(line_color="#0f766e")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="종가",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_public_candidates(snapshot_payload: dict[str, object]) -> None:
    candidate_rows = snapshot_payload.get("candidates", [])
    if not isinstance(candidate_rows, list) or not candidate_rows:
        st.info("최근 시장 스캔 결과가 아직 없습니다.")
        return

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        st.info("최근 시장 스캔 결과가 아직 없습니다.")
        return

    preferred_columns = [
        "name",
        "symbol",
        "market",
        "final_score",
        "up_probability_pct",
        "expected_return_pct",
        "realtime_change_rate",
        "article_count",
        "top_tags",
    ]
    visible_columns = [column for column in preferred_columns if column in candidate_df.columns]
    candidate_df = candidate_df[visible_columns].copy()
    if "top_tags" in candidate_df.columns:
        candidate_df["top_tags"] = candidate_df["top_tags"].apply(
            lambda tags: ", ".join(tags) if isinstance(tags, list) else str(tags or "")
        )

    candidate_df = candidate_df.rename(
        columns={
            "name": "종목명",
            "symbol": "종목코드",
            "market": "시장",
            "final_score": "최종 점수",
            "up_probability_pct": "상승 확률(%)",
            "expected_return_pct": "예상 상승률(%)",
            "realtime_change_rate": "실시간 변동률(%)",
            "article_count": "기사 수",
            "top_tags": "주요 태그",
        }
    )

    st.dataframe(
        candidate_df.head(12),
        use_container_width=True,
        hide_index=True,
        column_config={
            "최종 점수": st.column_config.NumberColumn("최종 점수", format="%.2f"),
            "상승 확률(%)": st.column_config.NumberColumn("상승 확률(%)", format="%.2f"),
            "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
            "실시간 변동률(%)": st.column_config.NumberColumn("실시간 변동률(%)", format="%.2f"),
            "기사 수": st.column_config.NumberColumn("기사 수", format="%d"),
        },
    )


def render_public_realized_chart(trade_history_df: pd.DataFrame) -> None:
    if trade_history_df.empty:
        return

    sell_df = trade_history_df[trade_history_df["side"] == "sell"].copy()
    if sell_df.empty:
        return

    sell_df["ordered_at"] = pd.to_datetime(sell_df["ordered_at"], errors="coerce")
    sell_df = sell_df.dropna(subset=["ordered_at"]).sort_values("ordered_at")
    if sell_df.empty:
        return

    sell_df["누적 실현손익"] = sell_df["realized_pnl"].cumsum()
    chart_df = sell_df[["ordered_at", "누적 실현손익"]].rename(columns={"ordered_at": "체결 시각"})
    st.line_chart(chart_df.set_index("체결 시각"), height=260)


@st.fragment(run_every=f"{get_public_cycle_config()['refresh_seconds']}s")
def render_public_portfolio_fragment() -> None:
    settings = get_settings()
    cycle_config = get_public_cycle_config()
    ran_cycle, cycle_message = maybe_run_public_cycle(settings, cycle_config)

    strategy_payload = load_strategy_state_payload()
    trade_state_payload = load_mock_trade_state_payload()
    snapshot_payload = load_candidate_snapshot_payload()

    positions_error = ""
    try:
        active_positions_df = build_active_positions_frame(settings)
    except Exception as exc:  # noqa: BLE001
        active_positions_df = pd.DataFrame()
        positions_error = str(exc)

    trade_history_df = build_auto_trade_history_df(strategy_payload, trade_state_payload)
    summary = build_public_summary(active_positions_df, trade_history_df)
    overview_symbol = choose_spotlight_symbol(cycle_config, active_positions_df, snapshot_payload)
    movement_df = build_public_movement_table(settings, active_positions_df, snapshot_payload)
    runtime_state = load_public_runtime_state()
    last_cycle_at = str(runtime_state.get("last_cycle_at", "")).strip() or "-"

    status_cols = st.columns(4)
    status_cols[0].metric("공개 자동 운용", "실행 중" if bool(cycle_config["enabled"]) else "모니터 전용")
    status_cols[1].metric("최근 엔진 실행", last_cycle_at)
    status_cols[2].metric("보유 종목 수", f"{int(summary['holding_count']):,}")
    status_cols[3].metric("최근 거래 건수", f"{len(trade_history_df):,}")

    if ran_cycle:
        st.success(cycle_message)
    else:
        st.caption(cycle_message)

    if positions_error:
        st.warning(f"현재가 갱신 중 일부 데이터 조회에 실패했습니다: {positions_error}")

    render_public_summary_cards(summary)

    st.subheader("대표 종목 주가 그래프")
    st.caption("공개 화면에서는 대표 종목을 자동으로 선택해 최근 주가 흐름을 보여줍니다.")
    render_public_spotlight_chart(settings, overview_symbol)

    st.subheader("주가 변동 요약")
    st.caption("보유 종목과 최근 추천 후보를 기준으로 전일, 5일, 20일 변화를 요약합니다.")
    render_public_movement_table(movement_df)

    holdings_tab, history_tab, candidates_tab = st.tabs(["현재 포지션", "최근 매매 이력", "최근 추천 후보"])
    with holdings_tab:
        selected_symbol, selected_name = render_public_positions(active_positions_df)
        if selected_symbol:
            st.markdown("**선택한 보유 종목 주가 그래프**")
            st.caption(f"{selected_name} ({selected_symbol})의 최근 주가 흐름입니다.")
            render_public_spotlight_chart(settings, selected_symbol)
    with history_tab:
        render_public_trade_history(trade_history_df)
        st.markdown("**누적 실현손익 추이**")
        render_public_realized_chart(trade_history_df)
    with candidates_tab:
        render_public_candidates(snapshot_payload)


def main() -> None:
    st.title("한국 주식 포트폴리오 공개 모니터")
    st.caption("읽기 전용 공개 페이지입니다. 포트폴리오 현황과 자동 운용 진행 결과만 표시되며, 방문자는 주문이나 설정을 조작할 수 없습니다.")
    st.info("이 앱은 이력서/포트폴리오 공유용 공개 화면입니다. 실전투자 키 대신 모의투자 키만 사용하는 구성을 권장합니다.")

    with st.expander("배포 안내", expanded=False):
        st.markdown(
            """
            - Streamlit Community Cloud에서 메인 파일을 `app_public.py`로 지정해 배포하세요.
            - 동적으로 포트폴리오가 갱신되게 하려면 `PUBLIC_APP_ENABLE_AUTO_CYCLE=true`를 설정하세요.
            - 공개 배포에는 실전 계좌 키를 넣지 말고, 모의투자 키만 사용하세요.
            """
        )

    render_public_portfolio_fragment()


if __name__ == "__main__":
    main()
