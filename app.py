from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.event_tags import add_event_tags
from analysis.forecast import calculate_price_forecast
from analysis.features import build_daily_feature_frame
from analysis.scoring import calculate_impact_signal, format_signal_summary
from analysis.sentiment import add_sentiment_columns
from services.learning_service import apply_learning_to_row, load_learning_state, update_learning_from_history
from services.news_service import fetch_company_news
from services.stock_service import (
    fetch_daily_stock_data,
    fetch_kis_realtime_quote,
    get_krx_stock_catalog,
    search_supported_symbols,
)
from services.trading_service import (
    build_active_positions_frame,
    close_mock_position,
    evaluate_mock_auto_sell,
    get_reference_price,
    get_mock_trading_missing_fields,
    has_mock_trading_config,
    inquire_mock_balance,
    inquire_mock_orderable_cash,
    place_mock_cash_order,
    register_auto_sell_position,
)
from utils.config import get_settings
from utils.helpers import (
    build_news_query,
    compute_recent_return,
    format_percentage,
    get_price_currency_symbol,
)
from workers.auto_trader import run_once as run_auto_trader_once
from workers.market_scanner import run_once as run_market_scanner_once
from workers.position_monitor import run_once as run_position_monitor_once

st.set_page_config(
    page_title="한국 주식 뉴스 영향 대시보드",
    page_icon=":bar_chart:",
    layout="wide",
)

RECOMMENDATION_UNIVERSE = [
    {"symbol": "005930", "display_name": "삼성전자", "news_query": "삼성전자"},
    {"symbol": "000660", "display_name": "SK하이닉스", "news_query": "SK하이닉스"},
    {"symbol": "035420", "display_name": "NAVER", "news_query": "네이버"},
    {"symbol": "005380", "display_name": "현대차", "news_query": "현대차"},
    {"symbol": "012330", "display_name": "현대모비스", "news_query": "현대모비스"},
    {"symbol": "051910", "display_name": "LG화학", "news_query": "LG화학"},
    {"symbol": "006400", "display_name": "삼성SDI", "news_query": "삼성SDI"},
    {"symbol": "066570", "display_name": "LG전자", "news_query": "LG전자"},
    {"symbol": "035720", "display_name": "카카오", "news_query": "카카오"},
    {"symbol": "068270", "display_name": "셀트리온", "news_query": "셀트리온"},
    {"symbol": "207940", "display_name": "삼성바이오로직스", "news_query": "삼성바이오로직스"},
    {"symbol": "105560", "display_name": "KB금융", "news_query": "KB금융"},
    {"symbol": "055550", "display_name": "신한지주", "news_query": "신한지주"},
    {"symbol": "096770", "display_name": "SK이노베이션", "news_query": "SK이노베이션"},
    {"symbol": "003670", "display_name": "포스코홀딩스", "news_query": "포스코홀딩스"},
    {"symbol": "034020", "display_name": "두산에너빌리티", "news_query": "두산에너빌리티"},
    {"symbol": "042700", "display_name": "한미반도체", "news_query": "한미반도체"},
    {"symbol": "012450", "display_name": "한화에어로스페이스", "news_query": "한화에어로스페이스"},
    {"symbol": "323410", "display_name": "카카오뱅크", "news_query": "카카오뱅크"},
    {"symbol": "028260", "display_name": "삼성물산", "news_query": "삼성물산"},
]

FALLBACK_STOCK_CATALOG = [
    {
        "symbol": item["symbol"],
        "name": item["display_name"],
        "market": "주요 종목",
        "news_query": item["news_query"],
    }
    for item in RECOMMENDATION_UNIVERSE
]

LIVE_CHART_HISTORY_LIMIT = 30
DATA_DIR = Path(__file__).resolve().parent / "data"
CANDIDATE_SNAPSHOT_PATH = DATA_DIR / "candidate_snapshot.json"
STRATEGY_STATE_PATH = DATA_DIR / "strategy_state.json"
MOCK_TRADE_STATE_PATH = DATA_DIR / "mock_trade_state.json"
AUTO_RUNTIME_STATE_PATH = DATA_DIR / "auto_runtime_state.json"
KST = ZoneInfo("Asia/Seoul")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_kr_market_open() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False

    current_hhmm = now.hour * 100 + now.minute
    return 905 <= current_hhmm <= 1515


def compute_volatility_balance_bonus(
    recent_volatility_pct: float,
    *,
    sweet_spot: float = 4.0,
    tolerance: float = 2.8,
    max_bonus: float = 8.0,
    excess_penalty: float = 0.9,
) -> float:
    volatility = max(0.0, safe_float(recent_volatility_pct))
    distance = abs(volatility - sweet_spot)
    if distance <= tolerance:
        return max(0.0, max_bonus - (distance / max(tolerance, 0.1)) * (max_bonus * 0.75))
    return -max(0.0, (distance - tolerance) * excess_penalty)


def init_page_state() -> None:
    defaults = {
        "analysis_ready": False,
        "scan_ready": False,
        "last_selected_ticker": "",
        "live_price_history": [],
        "live_price_ticker": "",
        "auto_trading_logs": [],
        "auto_last_scan_ts": 0.0,
        "auto_last_buy_ts": 0.0,
        "auto_last_monitor_ts": 0.0,
        "auto_last_cycle_ts": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_page_state() -> None:
    st.session_state.analysis_ready = False
    st.session_state.scan_ready = False
    st.session_state.live_price_history = []
    st.session_state.live_price_ticker = ""


def sync_selected_ticker(ticker: str) -> None:
    if st.session_state.get("last_selected_ticker") != ticker:
        st.session_state.last_selected_ticker = ticker
        st.session_state.live_price_history = []
        st.session_state.live_price_ticker = ""


def get_query_param_text(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default

    if isinstance(value, list):
        value = value[0] if value else default
    return str(value or default).strip()


def build_portfolio_share_suffix(symbol: str = "") -> str:
    params = ["view=portfolio"]
    normalized_symbol = str(symbol).strip()
    if normalized_symbol:
        params.append(f"symbol={normalized_symbol}")
    return "?" + "&".join(params)


@st.cache_data(ttl=1800)
def load_stock_data(symbol: str, kis_app_key: str, kis_app_secret: str) -> pd.DataFrame:
    return fetch_daily_stock_data(symbol=symbol, kis_app_key=kis_app_key, kis_app_secret=kis_app_secret)


@st.cache_data(ttl=1800)
def load_scan_stock_data(symbol: str) -> pd.DataFrame:
    return fetch_daily_stock_data(symbol=symbol, kis_app_key="", kis_app_secret="")


@st.cache_data(ttl=1800)
def load_symbol_suggestions(keywords: str) -> pd.DataFrame:
    return search_supported_symbols(keywords=keywords)


@st.cache_data(ttl=86400)
def load_full_stock_catalog() -> pd.DataFrame:
    try:
        catalog_df = get_krx_stock_catalog()
        if catalog_df.empty:
            return pd.DataFrame(FALLBACK_STOCK_CATALOG)
        return catalog_df
    except Exception:
        return pd.DataFrame(FALLBACK_STOCK_CATALOG)


@st.cache_data(ttl=30)
def load_realtime_quote(symbol: str, kis_app_key: str, kis_app_secret: str) -> dict[str, object]:
    if not kis_app_key or not kis_app_secret:
        return {}
    try:
        return fetch_kis_realtime_quote(symbol=symbol, app_key=kis_app_key, app_secret=kis_app_secret)
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def load_news_data(
    query: str,
    client_id: str,
    client_secret: str,
    page_size: int,
) -> pd.DataFrame:
    return fetch_company_news(
        query=query,
        client_id=client_id,
        client_secret=client_secret,
        page_size=page_size,
    )


def filter_news_window(news_df: pd.DataFrame, latest_date: pd.Timestamp, news_days: int) -> pd.DataFrame:
    if news_df.empty:
        return news_df.copy()

    start_bound = latest_date - timedelta(days=news_days)
    filtered_df = news_df[news_df["published_at"] >= start_bound].copy()
    return filtered_df.reset_index(drop=True)


def enrich_news_and_signal(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    enriched_news_df = add_sentiment_columns(news_df)
    enriched_news_df = add_event_tags(enriched_news_df)
    feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=enriched_news_df)
    signal = calculate_impact_signal(stock_df=stock_df, news_df=enriched_news_df, feature_df=feature_df)
    return enriched_news_df, feature_df, signal


def build_price_only_signal(stock_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    empty_news_df = pd.DataFrame(columns=["published_at", "title", "sentiment_score", "event_tag_list"])
    feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=empty_news_df)
    signal = calculate_impact_signal(stock_df=stock_df, news_df=empty_news_df, feature_df=feature_df)
    return feature_df, signal


def build_recommendation_row(candidate: dict[str, object], settings) -> dict[str, object] | None:
    try:
        symbol = str(candidate["symbol"])
        display_name = str(candidate.get("display_name") or candidate.get("name") or symbol)

        stock_df = load_scan_stock_data(symbol)
        if stock_df.empty:
            return None

        _, signal = build_price_only_signal(stock_df)
        realtime_quote: dict[str, object] = {}

        if getattr(settings, "kis_app_key", "") and getattr(settings, "kis_app_secret", ""):
            realtime_quote = load_realtime_quote(symbol, settings.kis_app_key, settings.kis_app_secret)

        latest_close = safe_float(stock_df["close"].iloc[-1])
        current_price = safe_float(realtime_quote.get("current_price"), latest_close)

        fallback_change_rate = (
            safe_float(stock_df["daily_return"].iloc[-1]) * 100 if "daily_return" in stock_df.columns else 0.0
        )
        realtime_change_rate = safe_float(realtime_quote.get("change_rate"), fallback_change_rate)

        forecast = calculate_price_forecast(
            stock_df=stock_df,
            signal=signal,
            current_price=current_price,
        )

        recent_volatility_pct = (
            safe_float(stock_df["daily_return"].tail(10).std()) * 100 if "daily_return" in stock_df.columns else 0.0
        )
        expected_return_pct = safe_float(forecast.get("expected_return_pct")) * 100
        up_probability_pct = safe_float(forecast.get("up_probability")) * 100
        recent_return_5d_pct = compute_recent_return(stock_df["close"], periods=5) * 100 if "close" in stock_df.columns else 0.0
        volatility_balance_bonus = compute_volatility_balance_bonus(recent_volatility_pct)

        market_cap_score = min(15.0, safe_float(candidate.get("market_cap")) / 15000.0)
        roe_score = max(0.0, min(12.0, safe_float(candidate.get("roe")) / 2.0))

        opportunity_score = (
            expected_return_pct * 3.55
            + up_probability_pct * 0.50
            + recent_volatility_pct * 0.65
            + volatility_balance_bonus
            + safe_float(signal.get("impact_score")) * 0.18
            + max(0.0, min(realtime_change_rate, 4.0)) * 0.55
            + max(0.0, recent_return_5d_pct) * 0.9
            + market_cap_score * 0.2
            + roe_score * 0.2
            - max(0.0, recent_volatility_pct - 9.0) * 0.85
        )

        row = {
            "symbol": symbol,
            "name": display_name,
            "market": str(candidate.get("market", "")),
            "current_price": current_price,
            "realtime_change_rate": realtime_change_rate,
            "opportunity_score": round(opportunity_score, 1),
            "impact_score": int(safe_float(signal.get("impact_score"))),
            "article_count": 0,
            "direction": str(forecast.get("direction", "중립")),
            "expected_return_pct": expected_return_pct,
            "up_probability": up_probability_pct,
            "recent_volatility_pct": recent_volatility_pct,
            "recent_return_5d_pct": recent_return_5d_pct,
            "volatility_balance_bonus": round(volatility_balance_bonus, 2),
        }
        return apply_learning_to_row(row, score_key="opportunity_score")
    except Exception:
        return None


def scan_recommendation_universe(
    candidates: list[dict[str, object]],
    settings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        row = build_recommendation_row(candidate=candidate, settings=settings)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "market",
                "current_price",
                "realtime_change_rate",
                "opportunity_score",
                "impact_score",
                "article_count",
                "direction",
                "expected_return_pct",
                "up_probability",
                "recent_volatility_pct",
                "recent_return_5d_pct",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["opportunity_score", "expected_return_pct", "up_probability", "volatility_balance_bonus", "recent_volatility_pct"],
        ascending=[False, False, False, False, True],
    )


def build_market_scan_candidates(
    stock_catalog_df: pd.DataFrame,
    selected_market: str,
    scan_pool_size: int,
) -> list[dict[str, object]]:
    if stock_catalog_df.empty:
        return RECOMMENDATION_UNIVERSE

    candidate_df = stock_catalog_df.copy()

    if selected_market != "전체" and "market" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["market"] == selected_market].copy()

    if candidate_df.empty:
        return RECOMMENDATION_UNIVERSE

    for numeric_column in ["prev_volume", "market_cap", "reference_price", "roe"]:
        if numeric_column not in candidate_df.columns:
            candidate_df[numeric_column] = 0.0
        candidate_df[numeric_column] = pd.to_numeric(candidate_df[numeric_column], errors="coerce").fillna(0.0)

    candidate_df["liquidity_rank"] = candidate_df["prev_volume"].rank(method="average", pct=True)
    candidate_df["market_cap_rank"] = candidate_df["market_cap"].rank(method="average", pct=True)
    candidate_df["roe_rank"] = candidate_df["roe"].clip(lower=0).rank(method="average", pct=True)
    candidate_df["balanced_liquidity_rank"] = (1.0 - (candidate_df["liquidity_rank"] - 0.62).abs() / 0.62).clip(lower=0.0)
    candidate_df["balanced_market_cap_rank"] = (1.0 - (candidate_df["market_cap_rank"] - 0.55).abs() / 0.55).clip(lower=0.0)
    candidate_df["scan_priority"] = (
        candidate_df["liquidity_rank"] * 0.38
        + candidate_df["balanced_liquidity_rank"] * 0.22
        + candidate_df["market_cap_rank"] * 0.12
        + candidate_df["balanced_market_cap_rank"] * 0.18
        + candidate_df["roe_rank"] * 0.10
    )
    candidate_df = candidate_df.sort_values(
        ["scan_priority", "prev_volume", "market_cap"],
        ascending=[False, False, False],
    )

    top_bucket_count = max(1, int(scan_pool_size * 0.45))
    balanced_bucket_count = max(1, int(scan_pool_size * 0.35))
    broad_bucket_count = max(1, scan_pool_size - top_bucket_count - balanced_bucket_count)

    top_bucket = candidate_df.head(top_bucket_count)
    balanced_bucket = candidate_df.sort_values(
        ["balanced_liquidity_rank", "balanced_market_cap_rank", "roe_rank"],
        ascending=[False, False, False],
    ).head(balanced_bucket_count)
    broad_bucket = candidate_df.sort_values(
        ["roe_rank", "balanced_market_cap_rank", "liquidity_rank"],
        ascending=[False, False, False],
    ).head(broad_bucket_count)

    candidate_df = pd.concat([top_bucket, balanced_bucket, broad_bucket], ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset=["symbol"]).head(scan_pool_size)

    return [
        {
            "symbol": row["symbol"],
            "display_name": row["name"],
            "name": row["name"],
            "market": row.get("market", ""),
            "news_query": row.get("news_query", row["name"]),
            "market_cap": row.get("market_cap", 0.0),
            "roe": row.get("roe", 0.0),
        }
        for _, row in candidate_df.iterrows()
    ]


def build_mover_table(recommendation_df: pd.DataFrame) -> pd.DataFrame:
    if recommendation_df.empty:
        return recommendation_df.copy()

    mover_df = recommendation_df.copy()
    balance_bonus = mover_df["recent_volatility_pct"].apply(compute_volatility_balance_bonus)
    mover_df["volatility_priority"] = (
        mover_df["expected_return_pct"] * 0.45
        + mover_df["up_probability"] * 0.25
        + mover_df["impact_score"] * 0.12
        + balance_bonus * 0.18
        - (mover_df["recent_volatility_pct"] - 9.0).clip(lower=0.0) * 0.35
    )
    mover_df = mover_df.sort_values(
        ["volatility_priority", "expected_return_pct", "up_probability", "impact_score"],
        ascending=[False, False, False, False],
    )
    return mover_df.drop(columns=["volatility_priority"])


def update_live_price_history(ticker: str, price: float) -> pd.DataFrame:
    history_key = "live_price_history"
    ticker_key = "live_price_ticker"
    current_time = pd.Timestamp.now()

    if st.session_state.get(ticker_key) != ticker:
        st.session_state[ticker_key] = ticker
        st.session_state[history_key] = []

    history = st.session_state.get(history_key, [])
    last_price = history[-1]["price"] if history else None
    last_time = history[-1]["timestamp"] if history else None

    should_append = (
        last_price is None
        or abs(safe_float(last_price) - safe_float(price)) > 0
        or (last_time is not None and (current_time - pd.Timestamp(last_time)).total_seconds() >= 10)
    )

    if should_append:
        history.append({"timestamp": current_time, "price": safe_float(price)})
        history = history[-LIVE_CHART_HISTORY_LIMIT:]
        st.session_state[history_key] = history

    return pd.DataFrame(history)


def render_symbol_suggestions(suggestions_df: pd.DataFrame) -> None:
    if suggestions_df.empty:
        st.info("일치하는 국내 종목 심볼을 찾지 못했습니다.")
        return

    st.write("국내 종목 검색 결과")
    st.dataframe(
        suggestions_df.rename(
            columns={
                "symbol": "심볼",
                "name": "종목명",
                "market": "시장",
            }
        )[["심볼", "종목명", "시장"]],
        use_container_width=True,
        hide_index=True,
    )


def render_overview_metrics(
    stock_df: pd.DataFrame,
    signal: dict[str, object],
    currency_symbol: str,
    current_quote: dict[str, object],
    forecast: dict[str, object],
) -> None:
    latest_close = safe_float(stock_df["close"].iloc[-1])
    realtime_price = safe_float(current_quote.get("current_price"), latest_close)
    change_rate = safe_float(current_quote.get("change_rate"), 0.0)
    recent_return = compute_recent_return(stock_df["close"], periods=5) if "close" in stock_df.columns else 0.0
    avg_sentiment = safe_float(signal.get("average_sentiment"), 0.0)
    impact_score = int(safe_float(signal.get("impact_score"), 0.0))
    direction = str(forecast.get("direction", "중립"))

    metric_columns = st.columns(4)
    metric_columns[0].metric("실시간 현재가", f"{currency_symbol}{realtime_price:,.2f}", delta=f"{change_rate:+.2f}%")
    metric_columns[1].metric("최근 5일 수익률", format_percentage(recent_return, show_sign=True))
    metric_columns[2].metric("평균 감성 점수", f"{avg_sentiment:.2f}")
    metric_columns[3].metric("영향 점수", f"{impact_score}/100", delta=direction)


def render_forecast_metrics(
    forecast: dict[str, object],
    currency_symbol: str,
) -> None:
    metric_columns = st.columns(3)
    metric_columns[0].metric("예상 등락률", format_percentage(safe_float(forecast.get("expected_return_pct")), show_sign=True))
    metric_columns[1].metric("상승 확률", f"{safe_float(forecast.get('up_probability')) * 100:.1f}%")
    metric_columns[2].metric("예상 주가", f"{currency_symbol}{safe_float(forecast.get('predicted_price')):,.2f}")


def render_quote_status(current_quote: dict[str, object]) -> None:
    if current_quote:
        st.caption("실시간 현재가는 KIS Open API 기준입니다.")
    else:
        st.caption("KIS 실시간 현재가를 불러오지 못해 최근 종가 기준으로 예측을 계산했습니다.")


def render_recommendation_table(recommendation_df: pd.DataFrame) -> None:
    if recommendation_df.empty:
        st.info("시장 기회 후보를 계산하지 못했습니다.")
        return

    display_df = recommendation_df[
        [
            "symbol",
            "name",
            "market",
            "current_price",
            "recent_volatility_pct",
            "realtime_change_rate",
            "opportunity_score",
            "learning_adjustment",
            "expected_return_pct",
            "up_probability",
            "impact_score",
            "direction",
        ]
    ].rename(
        columns={
            "symbol": "종목",
            "name": "종목명",
            "market": "시장",
            "current_price": "현재 기준가",
            "recent_volatility_pct": "최근 변동성(%)",
            "realtime_change_rate": "실시간 등락률(%)",
            "opportunity_score": "기회 점수",
            "learning_adjustment": "학습 보정",
            "expected_return_pct": "예상 상승률(%)",
            "up_probability": "상승 확률(%)",
            "impact_score": "영향 점수",
            "direction": "예상 방향",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재 기준가": st.column_config.NumberColumn("현재 기준가", format="₩%.2f"),
            "최근 변동성(%)": st.column_config.NumberColumn("최근 변동성(%)", format="%.2f"),
            "실시간 등락률(%)": st.column_config.NumberColumn("실시간 등락률(%)", format="%.2f"),
            "기회 점수": st.column_config.NumberColumn("기회 점수", format="%.1f"),
            "학습 보정": st.column_config.NumberColumn("학습 보정", format="%.2f"),
            "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
            "상승 확률(%)": st.column_config.NumberColumn("상승 확률(%)", format="%.1f"),
            "영향 점수": st.column_config.NumberColumn("영향 점수", format="%d"),
        },
    )


def render_top_movers_table(mover_df: pd.DataFrame, limit: int) -> None:
    if mover_df.empty:
        st.info("고변동성 후보를 계산하지 못했습니다.")
        return

    display_df = mover_df.head(limit)[
        [
            "symbol",
            "name",
            "market",
            "current_price",
            "recent_volatility_pct",
            "realtime_change_rate",
            "direction",
            "expected_return_pct",
            "up_probability",
            "impact_score",
        ]
    ].rename(
        columns={
            "symbol": "종목",
            "name": "종목명",
            "market": "시장",
            "current_price": "현재 기준가",
            "recent_volatility_pct": "최근 변동성(%)",
            "realtime_change_rate": "실시간 등락률(%)",
            "direction": "예상 방향",
            "expected_return_pct": "예상 상승률(%)",
            "up_probability": "상승 확률(%)",
            "impact_score": "영향 점수",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재 기준가": st.column_config.NumberColumn("현재 기준가", format="₩%.2f"),
            "최근 변동성(%)": st.column_config.NumberColumn("최근 변동성(%)", format="%.2f"),
            "실시간 등락률(%)": st.column_config.NumberColumn("실시간 등락률(%)", format="%.2f"),
            "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
            "상승 확률(%)": st.column_config.NumberColumn("상승 확률(%)", format="%.1f"),
            "영향 점수": st.column_config.NumberColumn("영향 점수", format="%d"),
        },
    )


@st.fragment(run_every="3s")
def render_live_price_chart_fragment(
    ticker: str,
    current_quote: dict[str, object],
    fallback_price: float,
    currency_symbol: str,
) -> None:
    live_price = safe_float(current_quote.get("current_price"), fallback_price)
    live_history_df = update_live_price_history(ticker=ticker, price=live_price)

    if live_history_df.empty:
        st.info("실시간 가격 기록이 아직 없습니다.")
        return

    fig = px.line(
        live_history_df,
        x="timestamp",
        y="price",
        markers=True,
        title="실시간 가격 추이",
        labels={"timestamp": "시각", "price": f"가격 ({currency_symbol})"},
    )
    fig.update_traces(line={"width": 3})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_stock_chart(stock_df: pd.DataFrame) -> None:
    chart_df = stock_df.copy()

    if "ma_5" not in chart_df.columns and "close" in chart_df.columns:
        chart_df["ma_5"] = chart_df["close"].rolling(5).mean()

    if "ma_20" not in chart_df.columns and "close" in chart_df.columns:
        chart_df["ma_20"] = chart_df["close"].rolling(20).mean()

    chart_df = chart_df.rename(
        columns={
            "close": "종가",
            "ma_5": "5일 이동평균",
            "ma_20": "20일 이동평균",
        }
    )

    fig = px.line(
        chart_df,
        x="date",
        y=["종가", "5일 이동평균", "20일 이동평균"],
        labels={"value": "가격", "date": "날짜", "variable": "지표"},
        title="주가 추이",
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def render_news_volume_chart(feature_df: pd.DataFrame) -> None:
    fig = px.bar(
        feature_df,
        x="date",
        y="article_count",
        title="일별 뉴스 기사 수",
        labels={"date": "날짜", "article_count": "기사 수"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_table(feature_df: pd.DataFrame, currency_symbol: str) -> None:
    display_df = (
        feature_df[
            [
                "date",
                "close",
                "daily_return",
                "volatility_5d",
                "article_count",
                "avg_sentiment",
                "positive_event_count",
                "negative_event_count",
            ]
        ]
        .sort_values("date", ascending=False)
        .rename(
            columns={
                "date": "날짜",
                "close": "종가",
                "daily_return": "일간 수익률",
                "volatility_5d": "5일 변동성",
                "article_count": "기사 수",
                "avg_sentiment": "평균 감성",
                "positive_event_count": "긍정 이벤트 수",
                "negative_event_count": "부정 이벤트 수",
            }
        )
    )

    display_df["일간 수익률"] = display_df["일간 수익률"] * 100
    display_df["5일 변동성"] = display_df["5일 변동성"] * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
            "종가": st.column_config.NumberColumn("종가", format=f"{currency_symbol}%.2f"),
            "일간 수익률": st.column_config.NumberColumn("일간 수익률", format="%.2f%%"),
            "5일 변동성": st.column_config.NumberColumn("5일 변동성", format="%.2f%%"),
            "기사 수": st.column_config.NumberColumn("기사 수", format="%d"),
            "평균 감성": st.column_config.NumberColumn("평균 감성", format="%.2f"),
            "긍정 이벤트 수": st.column_config.NumberColumn("긍정 이벤트 수", format="%d"),
            "부정 이벤트 수": st.column_config.NumberColumn("부정 이벤트 수", format="%d"),
        },
    )


def render_news_table(news_df: pd.DataFrame) -> None:
    if news_df.empty:
        st.info("선택한 기간에는 조회된 뉴스가 없습니다.")
        return

    display_df = news_df[
        [
            "published_at",
            "source",
            "title",
            "sentiment_label",
            "sentiment_score",
            "event_tags",
            "url",
        ]
    ].copy()

    display_df["published_at"] = display_df["published_at"].dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "published_at": "발행 시각",
            "source": "언론사",
            "title": "기사 제목",
            "sentiment_label": "감성",
            "sentiment_score": st.column_config.NumberColumn("감성 점수", format="%.2f"),
            "event_tags": "이벤트 태그",
            "url": st.column_config.LinkColumn("원문 링크"),
        },
    )


def render_mock_positions_table(positions_df: pd.DataFrame) -> None:
    if positions_df.empty:
        st.info("현재 저장된 모의투자 자동매도 포지션이 없습니다.")
        return

    display_df = positions_df.rename(
        columns={
            "symbol": "종목",
            "name": "종목명",
            "market": "시장",
            "quantity": "보유수량",
            "entry_price": "진입가",
            "current_price": "현재가",
            "expected_return_pct": "예상상승률(%)",
            "target_profit_pct": "자동매도 목표(%)",
            "current_return_pct": "현재수익률(%)",
            "auto_sell_enabled": "자동매도",
            "created_at": "등록시각",
        }
    )

    st.dataframe(
        display_df[
            [
                "종목",
                "종목명",
                "시장",
                "보유수량",
                "진입가",
                "현재가",
                "예상상승률(%)",
                "자동매도 목표(%)",
                "현재수익률(%)",
                "자동매도",
                "등록시각",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "보유수량": st.column_config.NumberColumn("보유수량", format="%d"),
            "진입가": st.column_config.NumberColumn("진입가", format="₩%.2f"),
            "현재가": st.column_config.NumberColumn("현재가", format="₩%.2f"),
            "예상상승률(%)": st.column_config.NumberColumn("예상상승률(%)", format="%.2f"),
            "자동매도 목표(%)": st.column_config.NumberColumn("자동매도 목표(%)", format="%.2f"),
            "현재수익률(%)": st.column_config.NumberColumn("현재수익률(%)", format="%.2f"),
        },
    )


def render_mock_positions_dashboard(
    positions_df: pd.DataFrame,
    *,
    key_prefix: str = "positions_dashboard",
) -> None:
    if positions_df.empty:
        st.info("현재 등록된 모의투자 자동매도 포지션이 없습니다.")
        return

    display_df = positions_df.copy()
    numeric_columns = [
        "quantity",
        "entry_price",
        "current_price",
        "expected_return_pct",
        "target_profit_pct",
        "current_return_pct",
    ]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").fillna(0.0)

    display_df["position_buy_amount"] = display_df["quantity"] * display_df["entry_price"]
    display_df["position_eval_amount"] = display_df["quantity"] * display_df["current_price"]
    display_df["position_pnl_amount"] = display_df["position_eval_amount"] - display_df["position_buy_amount"]

    total_buy_amount = float(display_df["position_buy_amount"].sum())
    total_eval_amount = float(display_df["position_eval_amount"].sum())
    total_pnl_amount = float(display_df["position_pnl_amount"].sum())

    metric_cols = st.columns(3)
    metric_cols[0].metric("보유 종목 총 매수금액", f"₩{total_buy_amount:,.0f}")
    metric_cols[1].metric("보유 종목 현재 평가금액", f"₩{total_eval_amount:,.0f}")
    metric_cols[2].metric("보유 종목 평가손익", f"₩{total_pnl_amount:,.0f}")

    control_cols = st.columns([1.6, 1.0])
    sort_mode = control_cols[0].radio(
        "포지션 정렬",
        options=["평가금액 상위", "수익률 상위", "손실률 상위"],
        horizontal=True,
        key=f"{key_prefix}_sort_mode",
    )
    highlight_enabled = control_cols[1].toggle(
        "손익 색상 강조",
        value=True,
        key=f"{key_prefix}_highlight_toggle",
    )

    if sort_mode == "수익률 상위":
        display_df = display_df.sort_values(
            by=["current_return_pct", "position_pnl_amount"],
            ascending=[False, False],
        )
    elif sort_mode == "손실률 상위":
        display_df = display_df.sort_values(
            by=["current_return_pct", "position_pnl_amount"],
            ascending=[True, True],
        )
    else:
        display_df = display_df.sort_values(
            by=["position_eval_amount", "current_return_pct"],
            ascending=[False, False],
        )
    display_df = display_df.reset_index(drop=True)

    display_df = display_df.rename(
        columns={
            "name": "종목명",
            "symbol": "종목코드",
            "market": "시장",
            "quantity": "보유수량",
            "entry_price": "매수단가",
            "position_buy_amount": "총 매수금액",
            "current_price": "현재가",
            "position_eval_amount": "현재 평가금액",
            "position_pnl_amount": "평가손익",
            "expected_return_pct": "예상 상승률(%)",
            "target_profit_pct": "자동매도 목표(%)",
            "current_return_pct": "현재 수익률(%)",
            "auto_sell_enabled": "자동매도",
            "created_at": "등록시각",
        }
    )

    table_df = display_df[
        [
            "종목명",
            "종목코드",
            "시장",
            "보유수량",
            "매수단가",
            "총 매수금액",
            "현재가",
            "현재 평가금액",
            "평가손익",
            "현재 수익률(%)",
            "예상 상승률(%)",
            "자동매도 목표(%)",
            "자동매도",
            "등록시각",
        ]
    ].copy()
    table_df["자동매도"] = table_df["자동매도"].map(lambda value: "ON" if bool(value) else "OFF")

    st.caption("정렬 기준을 바꾸면서 종목별 투자금, 평가금액, 평가손익을 한눈에 비교할 수 있습니다.")

    if highlight_enabled:
        def _profit_style(value: object) -> str:
            number = safe_float(value)
            if number > 0:
                return "color: #15803d; font-weight: 700;"
            if number < 0:
                return "color: #b91c1c; font-weight: 700;"
            return "color: #475569;"

        def _auto_sell_style(value: object) -> str:
            if str(value).upper() == "ON":
                return "background-color: rgba(21, 128, 61, 0.10); color: #166534; font-weight: 700;"
            return "color: #64748b;"

        styled_df = (
            table_df.style
            .format(
                {
                    "보유수량": "{:,.0f}",
                    "매수단가": "₩{:,.0f}",
                    "총 매수금액": "₩{:,.0f}",
                    "현재가": "₩{:,.0f}",
                    "현재 평가금액": "₩{:,.0f}",
                    "평가손익": "₩{:,.0f}",
                    "현재 수익률(%)": "{:,.2f}",
                    "예상 상승률(%)": "{:,.2f}",
                    "자동매도 목표(%)": "{:,.2f}",
                }
            )
            .applymap(_profit_style, subset=["평가손익", "현재 수익률(%)", "예상 상승률(%)"])
            .applymap(_auto_sell_style, subset=["자동매도"])
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "보유수량": st.column_config.NumberColumn("보유수량", format="%d"),
                "매수단가": st.column_config.NumberColumn("매수단가", format="₩%.0f"),
                "총 매수금액": st.column_config.NumberColumn("총 매수금액", format="₩%.0f"),
                "현재가": st.column_config.NumberColumn("현재가", format="₩%.0f"),
                "현재 평가금액": st.column_config.NumberColumn("현재 평가금액", format="₩%.0f"),
                "평가손익": st.column_config.NumberColumn("평가손익", format="₩%.0f"),
                "현재 수익률(%)": st.column_config.NumberColumn("현재 수익률(%)", format="%.2f"),
                "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
                "자동매도 목표(%)": st.column_config.NumberColumn("자동매도 목표(%)", format="%.2f"),
            },
        )


def render_auto_sell_actions(actions: list[dict[str, object]]) -> None:
    if not actions:
        st.info("이번 점검에서는 처리할 자동매도 이벤트가 없었습니다.")
        return

    action_df = pd.DataFrame(actions).rename(
        columns={
            "symbol": "종목",
            "name": "종목명",
            "status": "상태",
            "current_return_pct": "현재수익률(%)",
            "target_profit_pct": "목표수익률(%)",
            "sell_price": "매도가",
        }
    )

    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재수익률(%)": st.column_config.NumberColumn("현재수익률(%)", format="%.2f"),
            "목표수익률(%)": st.column_config.NumberColumn("목표수익률(%)", format="%.2f"),
            "매도가": st.column_config.NumberColumn("매도가", format="₩%.2f"),
        },
    )


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_payload(path: Path) -> dict[str, object]:
    _ensure_data_dir()
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json_payload(path: Path, payload: dict[str, object]) -> None:
    _ensure_data_dir()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_auto_runtime_state() -> dict[str, object]:
    state = _load_json_payload(AUTO_RUNTIME_STATE_PATH)
    return {
        "enabled": bool(state.get("enabled", False)),
        "last_market_open": state.get("last_market_open"),
        "last_resumed_at": str(state.get("last_resumed_at", "")),
    }


def save_auto_runtime_state(state: dict[str, object]) -> None:
    _ensure_data_dir()
    AUTO_RUNTIME_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_auto_resume_state(auto_trading_enabled: bool) -> None:
    runtime_state = load_auto_runtime_state()
    market_open = is_kr_market_open()
    previous_market_open = runtime_state.get("last_market_open")

    runtime_state["enabled"] = bool(auto_trading_enabled)
    runtime_state["last_market_open"] = market_open

    if auto_trading_enabled and market_open and previous_market_open is False:
        st.session_state.auto_last_scan_ts = 0.0
        st.session_state.auto_last_buy_ts = 0.0
        st.session_state.auto_last_monitor_ts = 0.0
        st.session_state.auto_last_cycle_ts = 0.0
        runtime_state["last_resumed_at"] = datetime.now(KST).isoformat(timespec="seconds")
        append_auto_log("장 시작이 감지되어 자동 운용을 다시 시작합니다.")

    save_auto_runtime_state(runtime_state)


def load_candidate_snapshot_payload() -> dict[str, object]:
    return _load_json_payload(CANDIDATE_SNAPSHOT_PATH)


def load_strategy_state_payload() -> dict[str, object]:
    return _load_json_payload(STRATEGY_STATE_PATH)


def load_mock_trade_state_payload() -> dict[str, object]:
    return _load_json_payload(MOCK_TRADE_STATE_PATH)


def append_auto_log(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    logs = st.session_state.get("auto_trading_logs", [])
    logs.append(f"[{timestamp}] {message}")
    st.session_state.auto_trading_logs = logs[-80:]


def record_manual_sell_history(
    position_row: pd.Series,
    *,
    sell_price: float,
    sold_at: str,
    reason: str = "manual_sell",
) -> tuple[float, float]:
    strategy_payload = load_strategy_state_payload()
    strategy_payload.setdefault("orders", [])

    quantity = int(safe_float(position_row.get("quantity"), 0))
    entry_price = safe_float(position_row.get("entry_price"), 0.0)
    pnl_amount = (sell_price - entry_price) * quantity
    pnl_pct = ((sell_price / entry_price) - 1) * 100 if entry_price else 0.0

    strategy_payload["orders"].append(
        {
            "type": "sell",
            "symbol": str(position_row.get("symbol", "")),
            "name": str(position_row.get("name", "")),
            "reason": reason,
            "quantity": quantity,
            "price": float(sell_price),
            "ordered_at": sold_at,
            "entry_price": entry_price,
            "realized_pnl": pnl_amount,
            "realized_pnl_pct": pnl_pct,
        }
    )
    _save_json_payload(STRATEGY_STATE_PATH, strategy_payload)
    update_learning_from_history(
        strategy_payload=strategy_payload,
        trade_state_payload=load_mock_trade_state_payload(),
    )
    return pnl_amount, pnl_pct


def execute_manual_position_sell(settings, position_row: pd.Series, *, reason: str = "manual_sell") -> tuple[float, float]:
    position_id = str(position_row.get("id", ""))
    symbol = str(position_row.get("symbol", ""))
    quantity = int(safe_float(position_row.get("quantity"), 0))
    current_price = safe_float(position_row.get("current_price"), 0.0)

    if not position_id or not symbol or quantity < 1:
        raise ValueError("매도할 포지션 정보를 확인할 수 없습니다.")

    sell_payload = place_mock_cash_order(
        settings=settings,
        side="sell",
        symbol=symbol,
        quantity=quantity,
        order_type="market",
    )
    sold_at = datetime.now(KST).isoformat(timespec="seconds")
    close_mock_position(
        position_id=position_id,
        sell_payload=sell_payload,
        sell_price=current_price,
        sold_at=sold_at,
    )
    return record_manual_sell_history(
        position_row,
        sell_price=current_price,
        sold_at=sold_at,
        reason=reason,
    )


def render_position_sell_controls(settings, positions_df: pd.DataFrame, *, key_prefix: str = "position_sell") -> None:
    if positions_df.empty:
        return

    st.markdown("**포지션 빠른 매도**")
    st.caption("자동 운용 포지션을 바로 정리하고 싶을 때 종목별 즉시 매도 버튼을 사용할 수 있습니다.")

    for _, row in positions_df.iterrows():
        quantity = int(safe_float(row.get("quantity"), 0))
        current_return_pct = safe_float(row.get("current_return_pct"), 0.0)
        current_price = safe_float(row.get("current_price"), 0.0)
        label_col, stat_col, button_col = st.columns([3.4, 1.4, 1.2])
        label_col.markdown(
            f"**{row.get('name', '')} ({row.get('symbol', '')})**  \n"
            f"보유 {quantity:,}주 · 현재가 ₩{current_price:,.0f}"
        )
        stat_col.metric("현재 수익률", f"{current_return_pct:.2f}%")
        if button_col.button("즉시 매도", key=f"{key_prefix}_{row.get('id', '')}"):
            try:
                pnl_amount, pnl_pct = execute_manual_position_sell(settings, row, reason="manual_sell")
                append_auto_log(
                    f"수동 매도 완료: {row.get('name', '')}({row.get('symbol', '')}) "
                    f"{quantity}주 / 손익 ₩{pnl_amount:,.0f} ({pnl_pct:.2f}%)"
                )
                st.success(
                    f"{row.get('name', '')}({row.get('symbol', '')}) {quantity}주를 즉시 매도했습니다. "
                    f"실현손익은 ₩{pnl_amount:,.0f} ({pnl_pct:.2f}%) 입니다."
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"즉시 매도에 실패했습니다: {exc}")


def render_auto_snapshot_table(snapshot_payload: dict[str, object]) -> None:
    candidate_rows = snapshot_payload.get("candidates", [])
    if not isinstance(candidate_rows, list) or not candidate_rows:
        st.info("아직 자동 탐색으로 저장된 상승 기대주 스냅샷이 없습니다.")
        return

    snapshot_df = pd.DataFrame(candidate_rows)
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
    visible_columns = [column for column in preferred_columns if column in snapshot_df.columns]
    snapshot_df = snapshot_df[visible_columns].copy()
    if "top_tags" in snapshot_df.columns:
        snapshot_df["top_tags"] = snapshot_df["top_tags"].apply(
            lambda tags: ", ".join(tags) if isinstance(tags, list) else str(tags or "")
        )

    st.dataframe(
        snapshot_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "종목명",
            "symbol": "종목코드",
            "market": "시장",
            "final_score": st.column_config.NumberColumn("최종점수", format="%.2f"),
            "up_probability_pct": st.column_config.NumberColumn("상승확률(%)", format="%.2f"),
            "expected_return_pct": st.column_config.NumberColumn("예상수익률(%)", format="%.2f"),
            "realtime_change_rate": st.column_config.NumberColumn("실시간등락률(%)", format="%.2f"),
            "article_count": st.column_config.NumberColumn("뉴스수", format="%d"),
            "top_tags": "주요태그",
        },
    )


def build_auto_trade_history_df(
    strategy_payload: dict[str, object],
    trade_state_payload: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_positions = trade_state_payload.get("positions", [])
    closed_positions = trade_state_payload.get("history", [])

    if isinstance(active_positions, list):
        for row in active_positions:
            quantity = int(safe_float(row.get("quantity"), 0))
            entry_price = safe_float(row.get("entry_price"), 0.0)
            rows.append(
                {
                    "ordered_at": str(row.get("created_at", "")),
                    "side": "buy",
                    "symbol": str(row.get("symbol", "")),
                    "name": str(row.get("name", "")),
                    "quantity": quantity,
                    "price": entry_price,
                    "status": "holding",
                    "realized_pnl": 0.0,
                    "realized_pnl_pct": 0.0,
                }
            )

    if isinstance(closed_positions, list):
        for row in closed_positions:
            quantity = int(safe_float(row.get("quantity"), 0))
            entry_price = safe_float(row.get("entry_price"), 0.0)
            sell_price = safe_float(row.get("sell_price"), 0.0)
            pnl_amount = (sell_price - entry_price) * quantity
            pnl_pct = (((sell_price / entry_price) - 1) * 100) if entry_price else 0.0
            rows.append(
                {
                    "ordered_at": str(row.get("created_at", "")),
                    "side": "buy",
                    "symbol": str(row.get("symbol", "")),
                    "name": str(row.get("name", "")),
                    "quantity": quantity,
                    "price": entry_price,
                    "status": "closed",
                    "realized_pnl": 0.0,
                    "realized_pnl_pct": 0.0,
                }
            )
            rows.append(
                {
                    "ordered_at": str(row.get("closed_at", "")),
                    "side": "sell",
                    "symbol": str(row.get("symbol", "")),
                    "name": str(row.get("name", "")),
                    "quantity": quantity,
                    "price": sell_price,
                    "status": "closed",
                    "realized_pnl": pnl_amount,
                    "realized_pnl_pct": pnl_pct,
                }
            )

    history_df = pd.DataFrame(rows)
    if history_df.empty:
        order_rows = strategy_payload.get("orders", [])
        if isinstance(order_rows, list) and order_rows:
            history_df = pd.DataFrame(order_rows)
            if "ordered_at" not in history_df.columns:
                history_df["ordered_at"] = ""
            if "type" in history_df.columns:
                history_df = history_df.rename(columns={"type": "side"})
            history_df["status"] = history_df.get("status", "logged")
            history_df["realized_pnl"] = 0.0
            history_df["realized_pnl_pct"] = 0.0

    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "ordered_at",
                "side",
                "symbol",
                "name",
                "quantity",
                "price",
                "status",
                "realized_pnl",
                "realized_pnl_pct",
            ]
        )

    history_df["ordered_at"] = pd.to_datetime(history_df["ordered_at"], errors="coerce")
    history_df = history_df.sort_values("ordered_at", ascending=False).reset_index(drop=True)
    return history_df


def render_auto_trade_summary_cards(
    active_positions_df: pd.DataFrame,
    trade_history_df: pd.DataFrame,
    account_summary: dict[str, object] | None = None,
) -> None:
    holding_count = int(len(active_positions_df))
    holding_quantity = int(active_positions_df["quantity"].sum()) if not active_positions_df.empty else 0
    active_principal = (
        float((active_positions_df["entry_price"] * active_positions_df["quantity"]).sum())
        if not active_positions_df.empty else 0.0
    )
    active_eval_amount = (
        float((active_positions_df["current_price"] * active_positions_df["quantity"]).sum())
        if not active_positions_df.empty else 0.0
    )
    unrealized_pnl = (
        float(((active_positions_df["current_price"] - active_positions_df["entry_price"]) * active_positions_df["quantity"]).sum())
        if not active_positions_df.empty else 0.0
    )
    realized_sell_df = trade_history_df[trade_history_df["side"] == "sell"] if not trade_history_df.empty else pd.DataFrame()
    realized_pnl = float(realized_sell_df["realized_pnl"].sum()) if not realized_sell_df.empty else 0.0
    total_invested_principal = active_principal
    stock_eval_amount = safe_float((account_summary or {}).get("stock_eval_amount"), active_eval_amount)
    total_eval_amount = safe_float((account_summary or {}).get("total_eval_amount"), stock_eval_amount)
    total_profit_amount = unrealized_pnl + realized_pnl
    cumulative_return_pct = (
        (total_profit_amount / total_invested_principal) * 100
        if total_invested_principal > 0 else 0.0
    )
    available_cash_amount = safe_float((account_summary or {}).get("cash_balance_amount"), 0.0)
    orderable_cash_amount = safe_float((account_summary or {}).get("orderable_cash_amount"), 0.0)

    latest_buy = (
        trade_history_df[trade_history_df["side"] == "buy"].head(1).to_dict("records")[0]
        if not trade_history_df.empty and not trade_history_df[trade_history_df["side"] == "buy"].empty
        else None
    )
    latest_sell = (
        trade_history_df[trade_history_df["side"] == "sell"].head(1).to_dict("records")[0]
        if not trade_history_df.empty and not trade_history_df[trade_history_df["side"] == "sell"].empty
        else None
    )

    top_metric_cols = st.columns(6)
    top_metric_cols[0].metric("보유 종목 수", f"{holding_count:,}")
    top_metric_cols[1].metric("보유 주식 수", f"{holding_quantity:,}")
    top_metric_cols[2].metric("총 투자원금", f"₩{total_invested_principal:,.0f}")
    top_metric_cols[3].metric("총 평가금액", f"₩{total_eval_amount:,.0f}")

    top_metric_cols[4].metric("예수금", f"₩{available_cash_amount:,.0f}")
    top_metric_cols[5].metric("주문 가능 금액", f"₩{orderable_cash_amount:,.0f}")

    bottom_metric_cols = st.columns(4)
    bottom_metric_cols[0].metric("미실현 손익", f"₩{unrealized_pnl:,.0f}")
    bottom_metric_cols[1].metric("실현 손익", f"₩{realized_pnl:,.0f}")
    bottom_metric_cols[2].metric("누적 손익", f"₩{total_profit_amount:,.0f}")
    bottom_metric_cols[3].metric("누적 수익률", f"{cumulative_return_pct:.2f}%")

    recent_cols = st.columns(2)
    if latest_buy:
        recent_cols[0].info(
            f"최근 자동 매수: {latest_buy['name']}({latest_buy['symbol']}) "
            f"{int(safe_float(latest_buy['quantity']))}주 · ₩{safe_float(latest_buy['price']):,.0f}"
        )
    else:
        recent_cols[0].info("최근 자동 매수 내역이 없습니다.")

    if latest_sell:
        recent_cols[1].info(
            f"최근 자동 매도: {latest_sell['name']}({latest_sell['symbol']}) "
            f"{int(safe_float(latest_sell['quantity']))}주 · ₩{safe_float(latest_sell['price']):,.0f} "
            f"({safe_float(latest_sell['realized_pnl_pct']):.2f}%)"
        )
    else:
        recent_cols[1].info("최근 자동 매도 내역이 없습니다.")


def render_auto_trade_history_table(trade_history_df: pd.DataFrame) -> None:
    if trade_history_df.empty:
        st.info("아직 자동 매매 이력이 없습니다.")
        return

    display_df = trade_history_df.copy()
    display_df["ordered_at"] = pd.to_datetime(display_df["ordered_at"], errors="coerce")
    display_df["ordered_at"] = display_df["ordered_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["side"] = display_df["side"].map({"buy": "매수", "sell": "매도"}).fillna(display_df["side"])

    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ordered_at": "체결 시각",
            "side": "구분",
            "symbol": "종목코드",
            "name": "종목명",
            "quantity": st.column_config.NumberColumn("수량", format="%d"),
            "price": st.column_config.NumberColumn("체결가", format="₩%.0f"),
            "status": "상태",
            "realized_pnl": st.column_config.NumberColumn("실현손익", format="₩%.0f"),
            "realized_pnl_pct": st.column_config.NumberColumn("실현손익률(%)", format="%.2f"),
        },
    )


def run_auto_trading_cycle(
    settings,
    *,
    selected_market: str,
    scan_pool_size: int,
    article_limit: int,
    news_days: int,
    recommendation_limit: int,
    use_realtime: bool,
    max_positions: int,
    max_daily_buys: int,
    budget_per_trade: float,
    min_final_score: float,
    min_up_probability: float,
    min_expected_return: float,
    min_sentiment: float,
    max_realtime_change: float,
    target_profit_factor: float,
    min_target_profit: float,
    max_target_profit: float,
    stop_loss_pct: float,
    max_hold_hours: float,
    scan_interval_seconds: int,
    buy_interval_seconds: int,
    monitor_interval_seconds: int,
) -> None:
    now_ts = time.time()
    scan_due = (now_ts - float(st.session_state.get("auto_last_scan_ts", 0.0))) >= scan_interval_seconds
    buy_due = (now_ts - float(st.session_state.get("auto_last_buy_ts", 0.0))) >= buy_interval_seconds
    monitor_due = (now_ts - float(st.session_state.get("auto_last_monitor_ts", 0.0))) >= monitor_interval_seconds

    if monitor_due:
        try:
            run_position_monitor_once(
                stop_loss_pct=stop_loss_pct,
                max_hold_hours=max_hold_hours,
                ignore_market_hours=False,
            )
            append_auto_log("포지션 감시를 실행했습니다.")
        except Exception as exc:  # noqa: BLE001
            append_auto_log(f"포지션 감시 실패: {exc}")
        finally:
            st.session_state.auto_last_monitor_ts = now_ts

    if scan_due:
        try:
            run_market_scanner_once(
                market=selected_market,
                scan_pool_size=scan_pool_size,
                top_n_for_news=max(recommendation_limit, 6),
                article_limit=article_limit,
                news_days=news_days,
                max_results=max(recommendation_limit, 5),
                use_realtime=use_realtime,
            )
            append_auto_log("상승 기대주 자동 탐색을 갱신했습니다.")
        except Exception as exc:  # noqa: BLE001
            append_auto_log(f"상승 기대주 탐색 실패: {exc}")
        finally:
            st.session_state.auto_last_scan_ts = now_ts

    if buy_due:
        try:
            run_auto_trader_once(
                max_positions=max_positions,
                max_daily_buys=max_daily_buys,
                budget_per_trade=budget_per_trade,
                min_final_score=min_final_score,
                min_up_probability=min_up_probability,
                min_expected_return=min_expected_return,
                min_sentiment=min_sentiment,
                max_realtime_change=max_realtime_change,
                target_profit_factor=target_profit_factor,
                min_target_profit=min_target_profit,
                max_target_profit=max_target_profit,
                ignore_market_hours=False,
            )
            append_auto_log("자동 매수 판단을 실행했습니다.")
        except Exception as exc:  # noqa: BLE001
            append_auto_log(f"자동 매수 실행 실패: {exc}")
        finally:
            st.session_state.auto_last_buy_ts = now_ts

    st.session_state.auto_last_cycle_ts = now_ts


@st.fragment(run_every="3s")
def render_auto_trading_fragment(
    settings,
    *,
    portfolio_view_only: bool,
    auto_trading_enabled: bool,
    selected_symbol: str,
    selected_market: str,
    scan_pool_size: int,
    article_limit: int,
    news_days: int,
    recommendation_limit: int,
    use_realtime: bool,
    max_positions: int,
    max_daily_buys: int,
    budget_per_trade: float,
    min_final_score: float,
    min_up_probability: float,
    min_expected_return: float,
    min_sentiment: float,
    max_realtime_change: float,
    target_profit_factor: float,
    min_target_profit: float,
    max_target_profit: float,
    stop_loss_pct: float,
    max_hold_hours: float,
    scan_interval_seconds: int,
    buy_interval_seconds: int,
    monitor_interval_seconds: int,
) -> None:
    st.subheader("자동 운용")

    if not auto_trading_enabled:
        st.info("자동 운용이 꺼져 있습니다. 사이드바에서 자동 운용을 켜면 상승 기대주 탐색과 모의 자동매매가 시작됩니다.")
        return

    if not has_mock_trading_config(settings):
        st.warning("자동 운용을 사용하려면 모의투자 계좌 설정이 먼저 필요합니다.")
        return

    run_auto_trading_cycle(
        settings,
        selected_market=selected_market,
        scan_pool_size=scan_pool_size,
        article_limit=article_limit,
        news_days=news_days,
        recommendation_limit=recommendation_limit,
        use_realtime=use_realtime,
        max_positions=max_positions,
        max_daily_buys=max_daily_buys,
        budget_per_trade=budget_per_trade,
        min_final_score=min_final_score,
        min_up_probability=min_up_probability,
        min_expected_return=min_expected_return,
        min_sentiment=min_sentiment,
        max_realtime_change=max_realtime_change,
        target_profit_factor=target_profit_factor,
        min_target_profit=min_target_profit,
        max_target_profit=max_target_profit,
        stop_loss_pct=stop_loss_pct,
        max_hold_hours=max_hold_hours,
        scan_interval_seconds=scan_interval_seconds,
        buy_interval_seconds=buy_interval_seconds,
        monitor_interval_seconds=monitor_interval_seconds,
    )

    snapshot_payload = load_candidate_snapshot_payload()
    strategy_payload = load_strategy_state_payload()
    trade_state_payload = load_mock_trade_state_payload()
    learning_state = update_learning_from_history(strategy_payload, trade_state_payload)
    account_summary: dict[str, object] = {}
    try:
        account_summary = inquire_mock_balance(settings).get("summary", {})
        if selected_symbol:
            reference_price = get_reference_price(settings, selected_symbol)
            account_summary["orderable_cash_amount"] = inquire_mock_orderable_cash(
                settings,
                selected_symbol,
                reference_price,
            )
    except Exception as exc:  # noqa: BLE001
        append_auto_log(f"모의 잔고 조회 실패: {exc}")
    active_positions_df = build_active_positions_frame(settings)
    trade_history_df = build_auto_trade_history_df(strategy_payload, trade_state_payload)

    status_cols = st.columns(5)
    status_cols[0].metric("자동 상태", "실행 중")
    status_cols[1].metric("활성 포지션", f"{len(active_positions_df):,}")
    status_cols[2].metric("탐색 후보 수", f"{len(snapshot_payload.get('candidates', [])):,}")
    status_cols[3].metric("누적 주문 기록", f"{len(strategy_payload.get('orders', [])):,}")
    status_cols[4].metric("학습 샘플 수", f"{int(learning_state.get('sample_count', 0)):,}")

    st.caption(
        f"화면은 3초마다 갱신되고, 탐색은 {scan_interval_seconds}초마다, 자동 매수는 {buy_interval_seconds}초마다, "
        f"포지션 감시는 {monitor_interval_seconds}초마다 실행됩니다."
    )

    render_auto_trade_summary_cards(active_positions_df, trade_history_df, account_summary=account_summary)

    with st.expander("자동 탐색 스냅샷", expanded=True):
        render_auto_snapshot_table(snapshot_payload)

    with st.expander("자동 운용 포지션", expanded=True):
        render_mock_positions_dashboard_safe(active_positions_df, key_prefix="auto_positions")
        render_position_sell_controls(settings, active_positions_df, key_prefix="auto_position_sell")

    with st.expander("최근 자동 매매 이력", expanded=True):
        render_auto_trade_history_table(trade_history_df)

    with st.expander("자동 운용 로그", expanded=True):
        logs = list(reversed(st.session_state.get("auto_trading_logs", [])))
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("아직 자동 운용 로그가 없습니다.")


@st.fragment(run_every="3s")
def render_portfolio_monitor_fragment(
    settings,
    *,
    auto_trading_enabled: bool,
    selected_symbol: str,
    selected_market: str,
    scan_pool_size: int,
    article_limit: int,
    news_days: int,
    recommendation_limit: int,
    use_realtime: bool,
    max_positions: int,
    max_daily_buys: int,
    budget_per_trade: float,
    min_final_score: float,
    min_up_probability: float,
    min_expected_return: float,
    min_sentiment: float,
    max_realtime_change: float,
    target_profit_factor: float,
    min_target_profit: float,
    max_target_profit: float,
    stop_loss_pct: float,
    max_hold_hours: float,
    scan_interval_seconds: int,
    buy_interval_seconds: int,
    monitor_interval_seconds: int,
) -> None:
    st.subheader("포트폴리오 모니터")

    if not has_mock_trading_config(settings):
        st.warning("포트폴리오 모니터를 사용하려면 모의투자 계좌 설정이 먼저 필요합니다.")
        return

    if auto_trading_enabled:
        run_auto_trading_cycle(
            settings,
            selected_market=selected_market,
            scan_pool_size=scan_pool_size,
            article_limit=article_limit,
            news_days=news_days,
            recommendation_limit=recommendation_limit,
            use_realtime=use_realtime,
            max_positions=max_positions,
            max_daily_buys=max_daily_buys,
            budget_per_trade=budget_per_trade,
            min_final_score=min_final_score,
            min_up_probability=min_up_probability,
            min_expected_return=min_expected_return,
            min_sentiment=min_sentiment,
            max_realtime_change=max_realtime_change,
            target_profit_factor=target_profit_factor,
            min_target_profit=min_target_profit,
            max_target_profit=max_target_profit,
            stop_loss_pct=stop_loss_pct,
            max_hold_hours=max_hold_hours,
            scan_interval_seconds=scan_interval_seconds,
            buy_interval_seconds=buy_interval_seconds,
            monitor_interval_seconds=monitor_interval_seconds,
        )

    strategy_payload = load_strategy_state_payload()
    trade_state_payload = load_mock_trade_state_payload()
    account_summary: dict[str, object] = {}
    try:
        account_summary = inquire_mock_balance(settings).get("summary", {})
        if selected_symbol:
            reference_price = get_reference_price(settings, selected_symbol)
            account_summary["orderable_cash_amount"] = inquire_mock_orderable_cash(
                settings,
                selected_symbol,
                reference_price,
            )
    except Exception as exc:  # noqa: BLE001
        append_auto_log(f"모의 잔고 조회 실패: {exc}")

    active_positions_df = build_active_positions_frame(settings)
    trade_history_df = build_auto_trade_history_df(strategy_payload, trade_state_payload)

    status_cols = st.columns(4)
    status_cols[0].metric("자동 상태", "실행 중" if auto_trading_enabled else "모니터링")
    status_cols[1].metric("보유 종목 수", f"{len(active_positions_df):,}")
    status_cols[2].metric("누적 주문 기록", f"{len(strategy_payload.get('orders', [])):,}")
    status_cols[3].metric("표시 기준 종목", str(selected_symbol or "-"))

    st.caption("이 화면은 포트폴리오 변화만 보이는 읽기 전용 모니터 화면입니다. 3초마다 자동 갱신됩니다.")

    render_auto_trade_summary_cards(active_positions_df, trade_history_df, account_summary=account_summary)

    with st.expander("현재 보유 포지션", expanded=True):
        render_mock_positions_dashboard_safe(active_positions_df, key_prefix="portfolio_monitor_positions")

    with st.expander("최근 자동 매매 이력", expanded=True):
        render_auto_trade_history_table(trade_history_df)

    with st.expander("자동 운용 로그", expanded=False):
        logs = list(reversed(st.session_state.get("auto_trading_logs", [])))
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("아직 자동 운용 로그가 없습니다.")


def render_missing_api_guide(settings, analysis_ready: bool) -> bool:
    missing_naver = []
    if not getattr(settings, "naver_client_id", ""):
        missing_naver.append("NAVER_CLIENT_ID")
    if not getattr(settings, "naver_client_secret", ""):
        missing_naver.append("NAVER_CLIENT_SECRET")

    missing_kis = []
    if not getattr(settings, "kis_app_key", ""):
        missing_kis.append("KIS_APP_KEY")
    if not getattr(settings, "kis_app_secret", ""):
        missing_kis.append("KIS_APP_SECRET")

    if analysis_ready and missing_naver:
        st.warning(
            "개별 종목 뉴스 분석을 하려면 `.env`에 네이버 뉴스 API 키가 필요합니다.\n\n"
            f"누락 항목: {', '.join(missing_naver)}"
        )
        st.code(
            "NAVER_CLIENT_ID=your_naver_client_id\n"
            "NAVER_CLIENT_SECRET=your_naver_client_secret"
        )
        return False

    if missing_kis:
        st.info(
            "KIS 앱키가 없어도 일별 시세는 보조 소스로 조회될 수 있지만, "
            "실시간 현재가와 시장 스캔 정확도는 떨어질 수 있습니다."
        )
        st.code(
            "KIS_APP_KEY=your_kis_app_key\n"
            "KIS_APP_SECRET=your_kis_app_secret"
        )

    return True


def main() -> None:
    init_page_state()
    settings = get_settings()
    update_learning_from_history(
        strategy_payload=load_strategy_state_payload(),
        trade_state_payload=load_mock_trade_state_payload(),
    )

    st.title("한국 주식 뉴스 영향 대시보드")
    st.caption(
        "네이버 뉴스, KIS Open API, pykrx를 조합해 한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 포트폴리오용 대시보드입니다."
    )

    auto_runtime_state = load_auto_runtime_state()
    query_view = get_query_param_text("view").lower()
    is_portfolio_share_view = query_view in {"portfolio", "monitor", "portfolio-only"}
    query_symbol = get_query_param_text("symbol")

    full_stock_catalog_df = load_full_stock_catalog()
    using_fallback_catalog = len(full_stock_catalog_df) <= len(FALLBACK_STOCK_CATALOG)

    market_options = ["전체", "KOSPI", "KOSDAQ", "KONEX"]
    selected_market = "전체"

    selectable_catalog_df = full_stock_catalog_df.copy()
    if selectable_catalog_df.empty:
        selectable_catalog_df = pd.DataFrame(FALLBACK_STOCK_CATALOG)

    selectable_catalog_df = selectable_catalog_df.sort_values(["name", "symbol"]).reset_index(drop=True)
    stock_options_by_symbol = selectable_catalog_df.set_index("symbol").to_dict("index")
    stock_symbol_options = selectable_catalog_df["symbol"].tolist()

    last_selected_symbol = str(st.session_state.get("last_selected_ticker") or "").strip()
    selected_symbol = ""
    if query_symbol in stock_options_by_symbol:
        selected_symbol = query_symbol
    elif last_selected_symbol in stock_options_by_symbol:
        selected_symbol = last_selected_symbol
    elif "005930" in stock_options_by_symbol:
        selected_symbol = "005930"
    elif stock_symbol_options:
        selected_symbol = stock_symbol_options[0]

    if not selected_symbol:
        st.error("표시할 종목 목록을 불러오지 못했습니다.")
        return

    selected_stock = stock_options_by_symbol[selected_symbol]
    ticker = selected_symbol
    company_name = str(selected_stock.get("news_query") or selected_stock["name"])
    sync_selected_ticker(ticker)

    app_mode = "포트폴리오 모니터" if is_portfolio_share_view else "전체 기능"
    news_days = 7
    article_limit = 20
    recommendation_limit = 8
    scan_pool_size = 60
    auto_trading_enabled = bool(auto_runtime_state.get("enabled", False))
    auto_use_realtime = True
    auto_budget_per_trade = 300000
    auto_max_positions = 3
    auto_max_daily_buys = 0
    auto_stop_loss_pct = 2.0
    auto_max_hold_hours = 24.0
    auto_min_final_score = 55.0
    auto_min_up_probability = 60.0
    auto_min_expected_return = 1.0
    auto_min_sentiment = 0.05
    auto_max_realtime_change = 7.0
    auto_target_profit_factor = 0.6
    auto_min_target_profit = 1.5
    auto_max_target_profit = 4.0
    auto_scan_interval_seconds = 180
    auto_buy_interval_seconds = 30
    auto_monitor_interval_seconds = 15

    with st.sidebar:
        st.header("분석 설정")
        if is_portfolio_share_view:
            st.info("공유 링크 전용 화면입니다. 포트폴리오 모니터만 표시됩니다.")
        else:
            app_mode = st.radio(
                "화면 모드",
                options=["포트폴리오 모니터", "전체 기능"],
                horizontal=True,
            )

        if using_fallback_catalog:
            st.warning("전체 상장사 목록을 불러오지 못해 현재는 주요 종목만 표시되고 있습니다.")

        market_options = ["전체", "KOSPI", "KOSDAQ", "KONEX"]
        selected_market = st.selectbox("시장", options=market_options, index=0)

        if selected_market == "전체":
            selectable_catalog_df = full_stock_catalog_df.copy()
        else:
            selectable_catalog_df = full_stock_catalog_df[full_stock_catalog_df["market"] == selected_market].copy()

        if selectable_catalog_df.empty:
            selectable_catalog_df = full_stock_catalog_df.copy()

        selectable_catalog_df = selectable_catalog_df.sort_values(["name", "symbol"]).reset_index(drop=True)
        stock_options_by_symbol = selectable_catalog_df.set_index("symbol").to_dict("index")
        stock_symbol_options = selectable_catalog_df["symbol"].tolist()
        default_symbol = selected_symbol if selected_symbol in stock_options_by_symbol else (
            "005930" if "005930" in stock_options_by_symbol else stock_symbol_options[0]
        )
        default_index = stock_symbol_options.index(default_symbol)

        selected_symbol = st.selectbox(
            "종목 선택",
            options=stock_symbol_options,
            index=default_index,
            format_func=lambda symbol: (
                f"{stock_options_by_symbol[symbol]['name']} ({symbol}, {stock_options_by_symbol[symbol]['market']})"
            ),
        )

        selected_stock = stock_options_by_symbol[selected_symbol]
        ticker = selected_symbol
        company_name = str(selected_stock.get("news_query") or selected_stock["name"])

        sync_selected_ticker(ticker)

        st.text_input("선택된 티커", value=ticker, disabled=True)
        st.text_input("뉴스 검색어", value=company_name, disabled=True)
        st.text_input(
            "공유 링크용 쿼리",
            value=build_portfolio_share_suffix(selected_symbol),
            disabled=True,
        )
        st.caption("현재 앱 주소 뒤에 위 쿼리를 붙이면 포트폴리오 모드만 바로 열립니다.")

        news_days = st.slider("뉴스 조회 기간(일)", min_value=3, max_value=30, value=7)
        article_limit = st.slider("최대 기사 수", min_value=10, max_value=50, value=20, step=5)
        recommendation_limit = st.slider("표시 종목 수", min_value=3, max_value=12, value=8)
        scan_pool_size = st.slider("시장 스캔 후보 수", min_value=20, max_value=120, value=60, step=10)

        st.divider()
        st.subheader("자동 운용 설정")
        auto_trading_enabled = st.toggle(
            "상승 기대주 자동 탐색 + 모의 자동매매",
            value=bool(auto_runtime_state.get("enabled", False)),
        )
        auto_use_realtime = st.checkbox("자동 운용에 실시간 시세 반영", value=True)
        auto_budget_per_trade = st.number_input("1회 매수 예산", min_value=10000, value=300000, step=10000)
        auto_max_positions = st.slider("최대 보유 종목 수", min_value=1, max_value=10, value=3)
        auto_max_daily_buys = st.slider("하루 최대 매수 횟수", min_value=0, max_value=20, value=0)
        if auto_max_daily_buys == 0:
            st.caption("하루 최대 매수 횟수: 제한 없음")
        auto_stop_loss_pct = st.slider("자동 손절 기준(%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        auto_max_hold_hours = st.number_input("최대 보유 시간(시간)", min_value=1.0, value=24.0, step=1.0)

        with st.expander("자동 운용 고급 설정", expanded=False):
            auto_min_final_score = st.slider("최소 최종 점수", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
            auto_min_up_probability = st.slider("최소 상승 확률(%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
            auto_min_expected_return = st.slider("최소 예상 수익률(%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            auto_min_sentiment = st.slider("최소 평균 감성 점수", min_value=-1.0, max_value=1.0, value=0.05, step=0.01)
            auto_max_realtime_change = st.slider("추격매수 제한 등락률(%)", min_value=1.0, max_value=20.0, value=7.0, step=0.5)
            auto_target_profit_factor = st.slider("예상 수익률 반영 비율", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
            auto_min_target_profit = st.slider("최소 목표 수익률(%)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
            auto_max_target_profit = st.slider("최대 목표 수익률(%)", min_value=1.0, max_value=20.0, value=4.0, step=0.1)
            auto_scan_interval_seconds = st.number_input("자동 탐색 주기(초)", min_value=30, value=180, step=30)
            auto_buy_interval_seconds = st.number_input("자동 매수 판단 주기(초)", min_value=5, value=30, step=5)
            auto_monitor_interval_seconds = st.number_input("자동 감시 주기(초)", min_value=5, value=15, step=5)

        button_cols = st.columns(3)
        run_analysis_clicked = button_cols[0].button("분석 실행", use_container_width=True)
        run_scan_clicked = button_cols[1].button("시장 기회 스캔", use_container_width=True)
        reset_clicked = button_cols[2].button("화면 초기화", use_container_width=True)

        if run_analysis_clicked:
            st.session_state.analysis_ready = True

        if run_scan_clicked:
            st.session_state.scan_ready = True

        if reset_clicked:
            clear_page_state()
            st.rerun()

        sync_auto_resume_state(auto_trading_enabled)

        st.caption(
            f"현재 선택 가능한 종목은 {len(selectable_catalog_df):,}개입니다. "
            f"시장 스캔은 거래가 활발한 상위 {scan_pool_size}개 후보를 먼저 추린 뒤 "
            "가격 변동성과 예상 상승률 중심으로 보여줍니다."
        )

        st.divider()
        st.write("사용 API")
        st.write("- 네이버 뉴스 검색 API")
        st.write("- KIS Open API")
        st.write("- pykrx / KIS 종목 마스터")

    analysis_ready = st.session_state.analysis_ready
    scan_ready = st.session_state.scan_ready

    if app_mode == "포트폴리오 모니터":
        render_portfolio_monitor_fragment(
            settings,
            auto_trading_enabled=auto_trading_enabled,
            selected_symbol=selected_symbol,
            selected_market=selected_market,
            scan_pool_size=scan_pool_size,
            article_limit=article_limit,
            news_days=news_days,
            recommendation_limit=recommendation_limit,
            use_realtime=auto_use_realtime,
            max_positions=auto_max_positions,
            max_daily_buys=auto_max_daily_buys,
            budget_per_trade=float(auto_budget_per_trade),
            min_final_score=float(auto_min_final_score),
            min_up_probability=float(auto_min_up_probability),
            min_expected_return=float(auto_min_expected_return),
            min_sentiment=float(auto_min_sentiment),
            max_realtime_change=float(auto_max_realtime_change),
            target_profit_factor=float(auto_target_profit_factor),
            min_target_profit=float(auto_min_target_profit),
            max_target_profit=float(auto_max_target_profit),
            stop_loss_pct=float(auto_stop_loss_pct),
            max_hold_hours=float(auto_max_hold_hours),
            scan_interval_seconds=int(auto_scan_interval_seconds),
            buy_interval_seconds=int(auto_buy_interval_seconds),
            monitor_interval_seconds=int(auto_monitor_interval_seconds),
        )
        return

    render_auto_trading_fragment(
        settings,
        portfolio_view_only=False,
        auto_trading_enabled=auto_trading_enabled,
        selected_symbol=selected_symbol,
        selected_market=selected_market,
        scan_pool_size=scan_pool_size,
        article_limit=article_limit,
        news_days=news_days,
        recommendation_limit=recommendation_limit,
        use_realtime=auto_use_realtime,
        max_positions=auto_max_positions,
        max_daily_buys=auto_max_daily_buys,
        budget_per_trade=float(auto_budget_per_trade),
        min_final_score=float(auto_min_final_score),
        min_up_probability=float(auto_min_up_probability),
        min_expected_return=float(auto_min_expected_return),
        min_sentiment=float(auto_min_sentiment),
        max_realtime_change=float(auto_max_realtime_change),
        target_profit_factor=float(auto_target_profit_factor),
        min_target_profit=float(auto_min_target_profit),
        max_target_profit=float(auto_max_target_profit),
        stop_loss_pct=float(auto_stop_loss_pct),
        max_hold_hours=float(auto_max_hold_hours),
        scan_interval_seconds=int(auto_scan_interval_seconds),
        buy_interval_seconds=int(auto_buy_interval_seconds),
        monitor_interval_seconds=int(auto_monitor_interval_seconds),
    )

    if not analysis_ready and not scan_ready:
        if auto_trading_enabled:
            st.info("자동 운용이 활성화되어 있어 상승 기대주 탐색과 모의 자동매매가 계속 진행됩니다.")
            return
        st.info("개별 종목을 보려면 `분석 실행`, 시장 전체 후보를 보려면 `시장 기회 스캔`을 눌러주세요.")
        return

    if not render_missing_api_guide(settings, analysis_ready=analysis_ready):
        return

    if not ticker or not company_name:
        st.error("종목 티커와 회사명을 확인해 주세요.")
        return

    scan_candidates = build_market_scan_candidates(
        stock_catalog_df=full_stock_catalog_df,
        selected_market=selected_market,
        scan_pool_size=scan_pool_size,
    )

    recommendation_df = pd.DataFrame()
    mover_df = pd.DataFrame()

    if scan_ready or analysis_ready:
        with st.spinner("시장 후보군의 변동성과 상승 기대를 분석하는 중입니다..."):
            recommendation_df = scan_recommendation_universe(
                candidates=scan_candidates,
                settings=settings,
            )
            mover_df = build_mover_table(recommendation_df)

    if scan_ready:
        st.subheader("고변동성 상승 기대주")
        st.caption(
            f"{selected_market} 시장에서 거래가 활발한 후보 {len(scan_candidates)}개를 먼저 추린 뒤 "
            "가격 변동성, 상승 기대, 실시간 등락률을 함께 반영한 결과입니다."
        )
        render_recommendation_table(recommendation_df.head(recommendation_limit))
        st.markdown("**변동성 상위 후보**")
        render_top_movers_table(mover_df, limit=recommendation_limit)

        if not analysis_ready:
            return

    try:
        with st.spinner("주가와 뉴스 데이터를 불러오는 중입니다..."):
            stock_df = load_stock_data(
                ticker,
                getattr(settings, "kis_app_key", ""),
                getattr(settings, "kis_app_secret", ""),
            )

            latest_date = pd.to_datetime(stock_df["date"].max())

            news_query = build_news_query(ticker=ticker, company_name=company_name)
            news_df = load_news_data(
                query=news_query,
                client_id=getattr(settings, "naver_client_id", ""),
                client_secret=getattr(settings, "naver_client_secret", ""),
                page_size=article_limit,
            )

        news_df = filter_news_window(news_df, latest_date=latest_date, news_days=news_days)
        news_df, feature_df, signal = enrich_news_and_signal(stock_df=stock_df, news_df=news_df)

        current_quote = load_realtime_quote(
            ticker,
            getattr(settings, "kis_app_key", ""),
            getattr(settings, "kis_app_secret", ""),
        )

        current_price = safe_float(current_quote.get("current_price"), safe_float(stock_df["close"].iloc[-1]))
        forecast = calculate_price_forecast(stock_df=stock_df, signal=signal, current_price=current_price)

    except Exception as exc:  # noqa: BLE001
        st.error(f"대시보드 데이터를 불러오지 못했습니다: {exc}")

        suggestion_queries = [company_name, ticker]
        shown_any_suggestion = False
        seen_queries: set[str] = set()

        for query in suggestion_queries:
            normalized_query = query.strip()
            if not normalized_query or normalized_query in seen_queries:
                continue

            seen_queries.add(normalized_query)

            try:
                suggestions_df = load_symbol_suggestions(normalized_query)
            except Exception:
                continue

            if not suggestions_df.empty:
                st.subheader(f"심볼 후보: {normalized_query}")
                render_symbol_suggestions(suggestions_df)
                shown_any_suggestion = True

        if not shown_any_suggestion:
            st.info("심볼 후보를 불러오지 못했습니다. 국내 주식은 보통 6자리 종목 코드 형식입니다.")
        return

    currency_symbol = get_price_currency_symbol(ticker)

    render_overview_metrics(
        stock_df=stock_df,
        signal=signal,
        currency_symbol=currency_symbol,
        current_quote=current_quote,
        forecast=forecast,
    )

    st.subheader("종합 해석")
    st.write(format_signal_summary(signal))

    st.subheader("예상 시나리오")
    st.caption("아래 수치는 최근 뉴스 흐름과 가격 변동성을 바탕으로 계산한 실험적 추정치입니다.")
    render_quote_status(current_quote)
    render_forecast_metrics(forecast=forecast, currency_symbol=currency_symbol)

    st.subheader("모의투자 주문")
    if not has_mock_trading_config(settings):
        mock_missing = get_mock_trading_missing_fields(settings)
        st.info("모의투자 주문을 사용하려면 아래 환경변수를 추가해 주세요: " + ", ".join(mock_missing))
        st.code(
            "KIS_MOCK_APP_KEY=your_kis_mock_app_key\n"
            "KIS_MOCK_APP_SECRET=your_kis_mock_app_secret\n"
            "KIS_ACCOUNT_NO=12345678\n"
            "KIS_ACCOUNT_PRODUCT_CODE=01"
        )
    else:
        target_profit_pct = max(0.5, safe_float(forecast.get("expected_return_pct")) * 100 * 0.5)
        mock_balance = {}
        balance_error = ""

        try:
            mock_balance = inquire_mock_balance(settings)
        except Exception as exc:  # noqa: BLE001
            balance_error = str(exc)

        if mock_balance:
            summary = mock_balance["summary"]
            balance_cols = st.columns(3)
            balance_cols[0].metric("모의 평가금액", f"₩{safe_float(summary.get('stock_eval_amount')):,.0f}")
            balance_cols[1].metric("모의 평가손익", f"₩{safe_float(summary.get('profit_loss_amount')):,.0f}")
            balance_cols[2].metric("모의 총평가", f"₩{safe_float(summary.get('total_eval_amount')):,.0f}")
        elif balance_error:
            st.warning(balance_error)
            st.caption("잔고 조회가 실패해도 로컬 포지션 추적 기반으로 모의 주문과 자동매도 점검은 계속 사용할 수 있습니다.")

        try:
            orderable_cash = inquire_mock_orderable_cash(settings, ticker, current_price)
            st.caption(f"모의 주문 가능 현금: ₩{orderable_cash:,.0f}")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"주문 가능 금액 조회는 생략되었습니다: {exc}")

        with st.form("mock_buy_form"):
            order_cols = st.columns(3)
            buy_quantity = order_cols[0].number_input("매수 수량", min_value=1, value=1, step=1)
            auto_sell_enabled = order_cols[1].checkbox(
                "자동매도 등록",
                value=safe_float(forecast.get("expected_return_pct")) > 0,
            )
            target_profit_override = order_cols[2].number_input(
                "자동매도 목표(%)",
                min_value=0.1,
                value=float(target_profit_pct),
                step=0.1,
            )
            buy_clicked = st.form_submit_button("현재 종목 모의 매수", use_container_width=True)

        if buy_clicked:
            try:
                buy_payload = place_mock_cash_order(
                    settings=settings,
                    side="buy",
                    symbol=ticker,
                    quantity=int(buy_quantity),
                    order_type="market",
                )

                if auto_sell_enabled:
                    register_auto_sell_position(
                        symbol=ticker,
                        name=company_name,
                        market=str(selected_stock.get("market", "")),
                        quantity=int(buy_quantity),
                        entry_price=current_price,
                        expected_return_pct=safe_float(forecast.get("expected_return_pct")) * 100,
                        target_profit_pct=float(target_profit_override),
                        auto_sell_enabled=True,
                        order_payload=buy_payload,
                    )

                st.success("모의 매수 주문이 접수되었습니다.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"모의 매수 주문에 실패했습니다: {exc}")

        with st.form("auto_sell_check_form"):
            check_clicked = st.form_submit_button("자동매도 지금 점검", use_container_width=True)

        auto_sell_actions: list[dict[str, object]] = []
        if check_clicked:
            try:
                auto_sell_actions = evaluate_mock_auto_sell(settings)
            except Exception as exc:  # noqa: BLE001
                st.error(f"자동매도 점검에 실패했습니다: {exc}")

        st.markdown("**자동매도 등록 포지션**")
        active_positions_df = build_active_positions_frame(settings)
        render_mock_positions_dashboard_safe(active_positions_df, key_prefix="manual_positions")

        if auto_sell_actions:
            st.markdown("**자동매도 점검 결과**")
            render_auto_sell_actions(auto_sell_actions)

    chart_columns = st.columns(3)
    with chart_columns[0]:
        render_stock_chart(stock_df)
    with chart_columns[1]:
        render_news_volume_chart(feature_df)
    with chart_columns[2]:
        render_live_price_chart_fragment(
            ticker=ticker,
            current_quote=current_quote,
            fallback_price=safe_float(stock_df["close"].iloc[-1]),
            currency_symbol=currency_symbol,
        )

    if not recommendation_df.empty:
        st.subheader("시장 스캔")
        st.caption("시장 스캔은 API 호출 부담 때문에 뉴스 대신 가격/변동성 중심으로 계산했습니다.")
        scan_columns = st.columns(2)
        with scan_columns[0]:
            st.markdown("**고변동성 상승 기대주**")
            render_recommendation_table(recommendation_df.head(recommendation_limit))
        with scan_columns[1]:
            st.markdown("**변동성 상위 후보**")
            render_top_movers_table(mover_df, limit=recommendation_limit)

    st.subheader("일별 통합 지표")
    render_feature_table(feature_df, currency_symbol=currency_symbol)

    st.subheader("최근 뉴스 분석")
    render_news_table(news_df.sort_values("published_at", ascending=False))

def render_mock_positions_dashboard_safe(
    positions_df: pd.DataFrame,
    *,
    key_prefix: str = "positions_dashboard",
) -> None:
    if positions_df.empty:
        st.info("현재 등록된 모의투자 자동매도 포지션이 없습니다.")
        return

    display_df = positions_df.copy()
    numeric_columns = [
        "quantity",
        "entry_price",
        "current_price",
        "expected_return_pct",
        "target_profit_pct",
        "current_return_pct",
    ]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").fillna(0.0)

    display_df["position_buy_amount"] = display_df["quantity"] * display_df["entry_price"]
    display_df["position_eval_amount"] = display_df["quantity"] * display_df["current_price"]
    display_df["position_pnl_amount"] = display_df["position_eval_amount"] - display_df["position_buy_amount"]

    total_buy_amount = float(display_df["position_buy_amount"].sum())
    total_eval_amount = float(display_df["position_eval_amount"].sum())
    total_pnl_amount = float(display_df["position_pnl_amount"].sum())

    metric_cols = st.columns(3)
    metric_cols[0].metric("보유 종목 총 매수금액", f"₩{total_buy_amount:,.0f}")
    metric_cols[1].metric("보유 종목 현재 평가금액", f"₩{total_eval_amount:,.0f}")
    metric_cols[2].metric("보유 종목 평가손익", f"₩{total_pnl_amount:,.0f}")

    control_cols = st.columns([1.6, 1.0])
    sort_mode = control_cols[0].radio(
        "포지션 정렬",
        options=["평가금액 상위", "수익률 상위", "손실률 상위"],
        horizontal=True,
        key=f"{key_prefix}_sort_mode_safe",
    )
    highlight_enabled = control_cols[1].toggle(
        "손익 색상 강조",
        value=True,
        key=f"{key_prefix}_highlight_toggle_safe",
    )

    if sort_mode == "수익률 상위":
        display_df = display_df.sort_values(
            by=["current_return_pct", "position_pnl_amount"],
            ascending=[False, False],
        )
    elif sort_mode == "손실률 상위":
        display_df = display_df.sort_values(
            by=["current_return_pct", "position_pnl_amount"],
            ascending=[True, True],
        )
    else:
        display_df = display_df.sort_values(
            by=["position_eval_amount", "current_return_pct"],
            ascending=[False, False],
        )
    display_df = display_df.reset_index(drop=True)

    display_df = display_df.rename(
        columns={
            "name": "종목명",
            "symbol": "종목코드",
            "market": "시장",
            "quantity": "보유수량",
            "entry_price": "매수단가",
            "position_buy_amount": "총 매수금액",
            "current_price": "현재가",
            "position_eval_amount": "현재 평가금액",
            "position_pnl_amount": "평가손익",
            "expected_return_pct": "예상 상승률(%)",
            "target_profit_pct": "자동매도 목표(%)",
            "current_return_pct": "현재 수익률(%)",
            "auto_sell_enabled": "자동매도",
            "created_at": "등록시각",
        }
    )

    table_df = display_df[
        [
            "종목명",
            "종목코드",
            "시장",
            "보유수량",
            "매수단가",
            "총 매수금액",
            "현재가",
            "현재 평가금액",
            "평가손익",
            "현재 수익률(%)",
            "예상 상승률(%)",
            "자동매도 목표(%)",
            "자동매도",
            "등록시각",
        ]
    ].copy()
    table_df["자동매도"] = table_df["자동매도"].map(lambda value: "ON" if bool(value) else "OFF")

    st.caption("정렬 기준을 바꾸면서 종목별 투자금, 평가금액, 평가손익을 한눈에 비교할 수 있습니다.")

    if highlight_enabled:
        def _profit_style(value: object) -> str:
            number = safe_float(value)
            if number > 0:
                return "color: #15803d; font-weight: 700;"
            if number < 0:
                return "color: #b91c1c; font-weight: 700;"
            return "color: #475569;"

        def _auto_sell_style(value: object) -> str:
            if str(value).upper() == "ON":
                return "background-color: rgba(21, 128, 61, 0.10); color: #166534; font-weight: 700;"
            return "color: #64748b;"

        styler = table_df.style.format(
            {
                "보유수량": "{:,.0f}",
                "매수단가": "₩{:,.0f}",
                "총 매수금액": "₩{:,.0f}",
                "현재가": "₩{:,.0f}",
                "현재 평가금액": "₩{:,.0f}",
                "평가손익": "₩{:,.0f}",
                "현재 수익률(%)": "{:,.2f}",
                "예상 상승률(%)": "{:,.2f}",
                "자동매도 목표(%)": "{:,.2f}",
            }
        )
        if hasattr(styler, "map"):
            styler = styler.map(_profit_style, subset=["평가손익", "현재 수익률(%)", "예상 상승률(%)"])
            styler = styler.map(_auto_sell_style, subset=["자동매도"])
        else:
            styler = styler.applymap(_profit_style, subset=["평가손익", "현재 수익률(%)", "예상 상승률(%)"])
            styler = styler.applymap(_auto_sell_style, subset=["자동매도"])
        st.dataframe(styler, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "보유수량": st.column_config.NumberColumn("보유수량", format="%d"),
                "매수단가": st.column_config.NumberColumn("매수단가", format="₩%.0f"),
                "총 매수금액": st.column_config.NumberColumn("총 매수금액", format="₩%.0f"),
                "현재가": st.column_config.NumberColumn("현재가", format="₩%.0f"),
                "현재 평가금액": st.column_config.NumberColumn("현재 평가금액", format="₩%.0f"),
                "평가손익": st.column_config.NumberColumn("평가손익", format="₩%.0f"),
                "현재 수익률(%)": st.column_config.NumberColumn("현재 수익률(%)", format="%.2f"),
                "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
                "자동매도 목표(%)": st.column_config.NumberColumn("자동매도 목표(%)", format="%.2f"),
            },
        )


if __name__ == "__main__":
    main()
