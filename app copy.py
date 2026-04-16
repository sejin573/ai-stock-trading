from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.event_tags import add_event_tags
from analysis.forecast import calculate_price_forecast
from analysis.features import build_daily_feature_frame
from analysis.scoring import calculate_impact_signal, format_signal_summary
from analysis.sentiment import add_sentiment_columns
from services.news_service import fetch_company_news
from services.stock_service import (
    fetch_daily_stock_data,
    fetch_kis_realtime_quote,
    get_krx_stock_catalog,
    search_supported_symbols,
)
from services.trading_service import (
    build_active_positions_frame,
    evaluate_mock_auto_sell,
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
    ensure_required_keys,
    format_percentage,
    get_price_currency_symbol,
)


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
    {"symbol": item["symbol"], "name": item["display_name"], "market": "주요 종목", "news_query": item["news_query"]}
    for item in RECOMMENDATION_UNIVERSE
]
LIVE_CHART_HISTORY_LIMIT = 30


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


def enrich_news_and_signal(stock_df: pd.DataFrame, news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    enriched_news_df = add_sentiment_columns(news_df)
    enriched_news_df = add_event_tags(enriched_news_df)
    feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=enriched_news_df)
    signal = calculate_impact_signal(stock_df=stock_df, news_df=enriched_news_df, feature_df=feature_df)
    return enriched_news_df, feature_df, signal


def build_price_only_signal(stock_df: pd.DataFrame) -> dict[str, object]:
    empty_news_df = pd.DataFrame(columns=["published_at", "title", "sentiment_score", "event_tag_list"])
    feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=empty_news_df)
    signal = calculate_impact_signal(stock_df=stock_df, news_df=empty_news_df, feature_df=feature_df)
    return signal


def build_recommendation_row(
    candidate: dict[str, object],
    settings,
    news_days: int,
    article_limit: int,
) -> dict[str, object] | None:
    try:
        symbol = str(candidate["symbol"])
        display_name = str(candidate.get("display_name") or candidate.get("name") or symbol)
        stock_df = load_scan_stock_data(symbol)
        signal = build_price_only_signal(stock_df)

        realtime_quote: dict[str, object] = {}
        if settings.kis_app_key and settings.kis_app_secret:
            realtime_quote = load_realtime_quote(symbol, settings.kis_app_key, settings.kis_app_secret)

        current_price = float(realtime_quote.get("current_price", stock_df["close"].iloc[-1]))
        fallback_change_rate = float(stock_df["daily_return"].iloc[-1] * 100) if not stock_df.empty else 0.0
        realtime_change_rate = float(realtime_quote.get("change_rate", fallback_change_rate))
        forecast = calculate_price_forecast(
            stock_df=stock_df,
            signal=signal,
            current_price=current_price,
        )
        recent_volatility_pct = float(stock_df["daily_return"].tail(10).std() or 0.0) * 100
        expected_return_pct = float(forecast["expected_return_pct"]) * 100
        up_probability_pct = float(forecast["up_probability"]) * 100
        recent_return_5d_pct = compute_recent_return(stock_df["close"], periods=5) * 100
        market_cap_score = min(15.0, float(candidate.get("market_cap", 0.0) or 0.0) / 15000.0)
        roe_score = max(0.0, min(12.0, float(candidate.get("roe", 0.0) or 0.0) / 2.0))
        opportunity_score = (
            expected_return_pct * 3.4
            + up_probability_pct * 0.45
            + recent_volatility_pct * 1.8
            + float(signal["impact_score"]) * 0.18
            + max(0.0, realtime_change_rate) * 0.8
            + max(0.0, recent_return_5d_pct) * 0.9
            + market_cap_score * 0.2
            + roe_score * 0.2
        )

        return {
            "symbol": symbol,
            "name": display_name,
            "market": str(candidate.get("market", "")),
            "current_price": current_price,
            "realtime_change_rate": realtime_change_rate,
            "opportunity_score": round(opportunity_score, 1),
            "impact_score": int(signal["impact_score"]),
            "article_count": 0,
            "direction": str(forecast["direction"]),
            "expected_return_pct": expected_return_pct,
            "up_probability": up_probability_pct,
            "recent_volatility_pct": recent_volatility_pct,
            "recent_return_5d_pct": recent_return_5d_pct,
        }
    except Exception:
        return None


def scan_recommendation_universe(
    candidates: list[dict[str, object]],
    settings,
    news_days: int,
    article_limit: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = build_recommendation_row(
            candidate=candidate,
            settings=settings,
            news_days=news_days,
            article_limit=article_limit,
        )
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
        ["opportunity_score", "expected_return_pct", "recent_volatility_pct", "up_probability"],
        ascending=False,
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
    candidate_df["scan_priority"] = (
        candidate_df["liquidity_rank"] * 0.7
        + candidate_df["market_cap_rank"] * 0.2
        + candidate_df["roe_rank"] * 0.1
    )

    candidate_df = candidate_df.sort_values(
        ["scan_priority", "prev_volume", "market_cap"],
        ascending=[False, False, False],
    ).head(scan_pool_size)

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
    mover_df["volatility_priority"] = mover_df["recent_volatility_pct"]
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
        or abs(float(last_price) - float(price)) > 0
        or (current_time - pd.Timestamp(last_time)).total_seconds() >= 10
    )
    if should_append:
        history.append({"timestamp": current_time, "price": float(price)})
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
    latest_close = float(stock_df["close"].iloc[-1])
    realtime_price = float(current_quote.get("current_price", latest_close))
    change_rate = float(current_quote.get("change_rate", 0.0))
    recent_return = compute_recent_return(stock_df["close"], periods=5)
    avg_sentiment = float(signal["average_sentiment"])
    impact_score = int(signal["impact_score"])
    direction = str(forecast["direction"])

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
    metric_columns[0].metric("예상 등락률", format_percentage(float(forecast["expected_return_pct"]), show_sign=True))
    metric_columns[1].metric("상승 확률", f"{float(forecast['up_probability']) * 100:.1f}%")
    metric_columns[2].metric("예상 주가", f"{currency_symbol}{float(forecast['predicted_price']):,.2f}")


def render_quote_status(current_quote: dict[str, object]) -> None:
    if current_quote:
        st.caption("실시간 현재가는 KIS Open API 기준입니다.")
    else:
        st.caption("KIS 실시간 현재가를 불러오지 못해 최근 종가 기준으로 예측을 계산했습니다.")


def render_recommendation_table(recommendation_df: pd.DataFrame) -> None:
    if recommendation_df.empty:
        st.info("추천 후보를 계산하지 못했습니다.")
        return

    display_df = recommendation_df.rename(
        columns={
            "symbol": "티커",
            "name": "종목명",
            "current_price": "현재 기준가",
            "realtime_change_rate": "실시간 등락률(%)",
            "recommendation_score": "추천 점수",
            "impact_score": "영향 점수",
            "article_count": "최근 기사 수",
            "direction": "예상 방향",
            "expected_return_pct": "예상 등락률(%)",
            "up_probability": "상승 확률(%)",
        }
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재 기준가": st.column_config.NumberColumn("현재 기준가", format="₩%.2f"),
            "실시간 등락률(%)": st.column_config.NumberColumn("실시간 등락률(%)", format="%.2f"),
            "추천 점수": st.column_config.NumberColumn("추천 점수", format="%.1f"),
            "영향 점수": st.column_config.NumberColumn("영향 점수", format="%d"),
            "최근 기사 수": st.column_config.NumberColumn("최근 기사 수", format="%d"),
            "예상 등락률(%)": st.column_config.NumberColumn("예상 등락률(%)", format="%.2f"),
            "상승 확률(%)": st.column_config.NumberColumn("상승 확률(%)", format="%.1f"),
        },
    )


def render_top_movers_table(mover_df: pd.DataFrame, limit: int) -> None:
    if mover_df.empty:
        st.info("변동률 상위 종목을 계산하지 못했습니다.")
        return

    display_df = mover_df.head(limit).rename(
        columns={
            "symbol": "티커",
            "name": "종목명",
            "current_price": "현재 기준가",
            "realtime_change_rate": "실시간 등락률(%)",
            "direction": "예상 방향",
            "expected_return_pct": "예상 변동률(%)",
            "up_probability": "상승 확률(%)",
            "impact_score": "영향 점수",
        }
    )
    st.dataframe(
        display_df[
            ["티커", "종목명", "현재 기준가", "실시간 등락률(%)", "예상 방향", "예상 변동률(%)", "상승 확률(%)", "영향 점수"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재 기준가": st.column_config.NumberColumn("현재 기준가", format="₩%.2f"),
            "실시간 등락률(%)": st.column_config.NumberColumn("실시간 등락률(%)", format="%.2f"),
            "예상 변동률(%)": st.column_config.NumberColumn("예상 변동률(%)", format="%.2f"),
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
    live_price = float(current_quote.get("current_price", fallback_price))
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


def main() -> None:
    settings = get_settings()

    st.title("한국 주식 뉴스 영향 대시보드")
    st.caption(
        "네이버 뉴스, KIS Open API, pykrx를 조합해 한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 포트폴리오용 대시보드입니다."
    )

    with st.sidebar:
        st.header("분석 설정")
        full_stock_catalog_df = load_full_stock_catalog()
        using_fallback_catalog = len(full_stock_catalog_df) <= len(FALLBACK_STOCK_CATALOG)

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
        default_index = stock_symbol_options.index("005930") if "005930" in stock_symbol_options else 0

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
        st.text_input("선택된 티커", value=ticker, disabled=True)
        st.text_input("뉴스 검색어", value=company_name, disabled=True)
        news_days = st.slider("뉴스 조회 기간(일)", min_value=3, max_value=30, value=7)
        article_limit = st.slider("최대 기사 수", min_value=10, max_value=50, value=20, step=5)
        recommendation_limit = st.slider("추천 종목 수", min_value=3, max_value=12, value=6)
        run_analysis = st.button("분석 실행", use_container_width=True)
        run_scan = st.button("유망주 추천 스캔", use_container_width=True)
        st.caption(
            f"현재 선택 가능한 종목은 {len(selectable_catalog_df):,}개입니다. "
            "셀렉트박스에서 회사명을 검색해 바로 찾을 수 있고, 실시간 차트는 3초마다 새로고침됩니다."
        )

        st.divider()
        st.write("사용 API")
        st.write("- 네이버 뉴스 검색 API")
        st.write("- KIS Open API")
        st.write("- pykrx (보조/백테스트)")

    missing_keys = ensure_required_keys(settings)
    if missing_keys:
        st.warning(
            "대시보드를 실행하기 전에 `.env` 파일에 API 키를 입력해주세요.\n\n"
            f"누락된 키: {', '.join(missing_keys)}"
        )
        st.code(
            "NAVER_CLIENT_ID=your_naver_client_id\n"
            "NAVER_CLIENT_SECRET=your_naver_client_secret\n"
            "KIS_APP_KEY=your_kis_app_key\n"
            "KIS_APP_SECRET=your_kis_app_secret"
        )
        st.info("KIS 앱키가 없어도 pykrx로 한국 주식 일별 시세를 보조 조회할 수 있습니다.")
        return

    if not run_analysis:
        if not run_scan:
            st.info("종목을 입력한 뒤 `분석 실행`을 누르거나 `유망주 추천 스캔`으로 후보 종목을 확인해보세요.")
            return

    if not ticker or not company_name:
        st.error("종목 티커와 회사명을 모두 입력해주세요.")
        return

    if run_scan:
        with st.spinner("추천 후보 종목을 스캔하는 중입니다..."):
            recommendation_df = scan_recommendation_universe(
                settings=settings,
                news_days=news_days,
                article_limit=article_limit,
            )
        st.subheader("최근 유망주 추천")
        st.caption("대표 한국 종목 후보군을 기준으로 최근 뉴스와 가격 흐름을 함께 반영한 실험적 추천 결과입니다.")
        render_recommendation_table(recommendation_df.head(recommendation_limit))

        if not run_analysis:
            return

    try:
        with st.spinner("주가와 뉴스 데이터를 불러오는 중입니다..."):
            stock_df = load_stock_data(ticker, settings.kis_app_key, settings.kis_app_secret)
            latest_date = pd.to_datetime(stock_df["date"].max())
            news_query = build_news_query(ticker=ticker, company_name=company_name)
            news_df = load_news_data(
                query=news_query,
                client_id=settings.naver_client_id,
                client_secret=settings.naver_client_secret,
                page_size=article_limit,
            )

        news_df = filter_news_window(news_df, latest_date=latest_date, news_days=news_days)
        news_df, feature_df, signal = enrich_news_and_signal(stock_df=stock_df, news_df=news_df)
        current_quote = load_realtime_quote(ticker, settings.kis_app_key, settings.kis_app_secret)
        current_price = float(current_quote.get("current_price", stock_df["close"].iloc[-1]))
        forecast = calculate_price_forecast(stock_df=stock_df, signal=signal, current_price=current_price)
        recommendation_df = scan_recommendation_universe(
            settings=settings,
            news_days=news_days,
            article_limit=article_limit,
        )
        mover_df = build_mover_table(recommendation_df)

    except Exception as exc:  # noqa: BLE001
        st.error(f"대시보드 데이터를 불러오지 못했습니다: {exc}")

        error_message = str(exc)
        if "KIS" in error_message or "pykrx" in error_message or "한국 주식" in error_message:
            st.warning(
                "입력한 종목 티커를 KIS 또는 pykrx에서 확인하지 못했습니다. 한국 시장은 보통 6자리 종목 코드 형식으로 입력합니다."
            )
            st.markdown(
                "- KIS Open API 포털: https://apiportal.koreainvestment.com/\n"
                "- pykrx 문서: https://github.com/sharebook-kr/pykrx\n"
                "- 먼저 회사명이나 6자리 코드로 검색된 심볼을 아래에서 확인해보세요."
            )

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
                st.info("심볼 후보를 불러오지 못했습니다. 우선 `005930`, `000660`, `035420` 같은 대표적인 6자리 국내 티커로 확인해보세요.")
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
    st.caption("아래 수치는 최근 뉴스 흐름과 가격 변동성을 바탕으로 계산한 실험적 추정치이며, 투자 판단의 참고용입니다.")
    render_quote_status(current_quote)
    render_forecast_metrics(forecast=forecast, currency_symbol=currency_symbol)

    chart_columns = st.columns(3)
    with chart_columns[0]:
        render_stock_chart(stock_df)
    with chart_columns[1]:
        render_news_volume_chart(feature_df)
    with chart_columns[2]:
        render_live_price_chart_fragment(
            ticker=ticker,
            current_quote=current_quote,
            fallback_price=float(stock_df["close"].iloc[-1]),
            currency_symbol=currency_symbol,
        )

    st.subheader("시장 스캔")
    st.caption("대표 한국 종목 후보군을 대상으로 최근 뉴스와 가격 흐름을 함께 반영한 추천 및 변동률 상위 결과입니다.")
    scan_columns = st.columns(2)
    with scan_columns[0]:
        st.markdown("**추천 종목**")
        render_recommendation_table(recommendation_df.head(recommendation_limit))
    with scan_columns[1]:
        st.markdown("**변동률 주목 종목**")
        render_top_movers_table(mover_df, limit=recommendation_limit)

    st.subheader("일별 통합 지표")
    render_feature_table(feature_df, currency_symbol=currency_symbol)

    st.subheader("최근 뉴스 분석")
    render_news_table(news_df.sort_values("published_at", ascending=False))


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
            "expected_return_pct",
            "up_probability",
            "impact_score",
            "article_count",
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
            "expected_return_pct": "예상 상승률(%)",
            "up_probability": "상승 확률(%)",
            "impact_score": "영향 점수",
            "article_count": "최근 기사 수",
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
            "예상 상승률(%)": st.column_config.NumberColumn("예상 상승률(%)", format="%.2f"),
            "상승 확률(%)": st.column_config.NumberColumn("상승 확률(%)", format="%.1f"),
            "영향 점수": st.column_config.NumberColumn("영향 점수", format="%d"),
            "최근 기사 수": st.column_config.NumberColumn("최근 기사 수", format="%d"),
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


def main() -> None:
    settings = get_settings()

    st.title("한국 주식 뉴스 영향 대시보드")
    st.caption(
        "네이버 뉴스, KIS Open API, pykrx를 조합해 한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 포트폴리오용 대시보드입니다."
    )

    with st.sidebar:
        st.header("분석 설정")
        full_stock_catalog_df = load_full_stock_catalog()
        using_fallback_catalog = len(full_stock_catalog_df) <= len(FALLBACK_STOCK_CATALOG)

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
        default_index = stock_symbol_options.index("005930") if "005930" in stock_symbol_options else 0

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
        st.text_input("선택된 티커", value=ticker, disabled=True)
        st.text_input("뉴스 검색어", value=company_name, disabled=True)
        news_days = st.slider("뉴스 조회 기간(일)", min_value=3, max_value=30, value=7)
        article_limit = st.slider("최대 기사 수", min_value=10, max_value=50, value=20, step=5)
        recommendation_limit = st.slider("표시 종목 수", min_value=3, max_value=12, value=6)
        scan_pool_size = st.slider("시장 스캔 후보 수", min_value=20, max_value=120, value=40, step=10)
        run_analysis = st.button("분석 실행", use_container_width=True)
        run_scan = st.button("시장 기회 스캔", use_container_width=True)
        st.caption(
            f"현재 선택 가능한 종목은 {len(selectable_catalog_df):,}개입니다. "
            f"시장 스캔은 거래가 활발한 상위 {scan_pool_size}개 후보를 먼저 추린 뒤 "
            "변동성과 예상 상승률이 높은 종목을 보여줍니다."
        )

        st.divider()
        st.write("사용 API")
        st.write("- 네이버 뉴스 검색 API")
        st.write("- KIS Open API")
        st.write("- pykrx / KIS 종목 마스터")

    missing_keys = ensure_required_keys(settings)
    if missing_keys:
        st.warning(
            "대시보드를 실행하기 전에 `.env` 파일에 API 키를 입력해 주세요.\n\n"
            f"누락 항목: {', '.join(missing_keys)}"
        )
        st.code(
            "NAVER_CLIENT_ID=your_naver_client_id\n"
            "NAVER_CLIENT_SECRET=your_naver_client_secret\n"
            "KIS_APP_KEY=your_kis_app_key\n"
            "KIS_APP_SECRET=your_kis_app_secret"
        )
        st.info("KIS 키가 없으면 실시간 시세와 시장 스캔 정확도가 떨어질 수 있습니다.")
        return

    if not run_analysis and not run_scan:
        st.info("개별 종목을 보려면 `분석 실행`, 시장 전체 후보를 보려면 `시장 기회 스캔`을 눌러주세요.")
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

    if run_scan or run_analysis:
        with st.spinner("시장 후보군의 변동성과 상승 기대를 분석하는 중입니다..."):
            recommendation_df = scan_recommendation_universe(
                candidates=scan_candidates,
                settings=settings,
                news_days=news_days,
                article_limit=article_limit,
            )
            mover_df = build_mover_table(recommendation_df)

    if run_scan:
        st.subheader("고변동성 상승 기대주")
        st.caption(
            f"{selected_market} 시장에서 거래가 활발한 후보 {len(scan_candidates)}개를 먼저 추린 뒤 "
            "뉴스 흐름, 최근 변동성, 예상 상승률을 함께 반영한 결과입니다."
        )
        render_recommendation_table(recommendation_df.head(recommendation_limit))

        if not run_analysis:
            st.markdown("**변동성 상위 후보**")
            render_top_movers_table(mover_df, limit=recommendation_limit)
            return

    try:
        with st.spinner("주가와 뉴스 데이터를 불러오는 중입니다..."):
            stock_df = load_stock_data(ticker, settings.kis_app_key, settings.kis_app_secret)
            latest_date = pd.to_datetime(stock_df["date"].max())
            news_query = build_news_query(ticker=ticker, company_name=company_name)
            news_df = load_news_data(
                query=news_query,
                client_id=settings.naver_client_id,
                client_secret=settings.naver_client_secret,
                page_size=article_limit,
            )

        news_df = filter_news_window(news_df, latest_date=latest_date, news_days=news_days)
        news_df, feature_df, signal = enrich_news_and_signal(stock_df=stock_df, news_df=news_df)
        current_quote = load_realtime_quote(ticker, settings.kis_app_key, settings.kis_app_secret)
        current_price = float(current_quote.get("current_price", stock_df["close"].iloc[-1]))
        forecast = calculate_price_forecast(stock_df=stock_df, signal=signal, current_price=current_price)

    except Exception as exc:  # noqa: BLE001
        st.error(f"대시보드 데이터를 불러오지 못했습니다: {exc}")
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

    chart_columns = st.columns(3)
    with chart_columns[0]:
        render_stock_chart(stock_df)
    with chart_columns[1]:
        render_news_volume_chart(feature_df)
    with chart_columns[2]:
        render_live_price_chart_fragment(
            ticker=ticker,
            current_quote=current_quote,
            fallback_price=float(stock_df["close"].iloc[-1]),
            currency_symbol=currency_symbol,
        )

    if not recommendation_df.empty:
        st.subheader("시장 스캔")
        st.caption("시장 전체 후보 중 변동성이 크고 상승 기대가 높은 종목들을 함께 보여줍니다.")
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


def main() -> None:
    settings = get_settings()

    st.title("한국 주식 뉴스 영향 대시보드")
    st.caption("네이버 뉴스, KIS Open API, pykrx를 조합해 한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 포트폴리오용 대시보드입니다.")

    with st.sidebar:
        st.header("분석 설정")
        full_stock_catalog_df = load_full_stock_catalog()
        using_fallback_catalog = len(full_stock_catalog_df) <= len(FALLBACK_STOCK_CATALOG)
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
        default_index = stock_symbol_options.index("005930") if "005930" in stock_symbol_options else 0

        selected_symbol = st.selectbox(
            "종목 선택",
            options=stock_symbol_options,
            index=default_index,
            format_func=lambda symbol: f"{stock_options_by_symbol[symbol]['name']} ({symbol}, {stock_options_by_symbol[symbol]['market']})",
        )
        selected_stock = stock_options_by_symbol[selected_symbol]
        ticker = selected_symbol
        company_name = str(selected_stock.get("news_query") or selected_stock["name"])

        st.text_input("선택된 티커", value=ticker, disabled=True)
        st.text_input("뉴스 검색어", value=company_name, disabled=True)
        news_days = st.slider("뉴스 조회 기간(일)", min_value=3, max_value=30, value=7)
        article_limit = st.slider("최대 기사 수", min_value=10, max_value=50, value=20, step=5)
        recommendation_limit = st.slider("표시 종목 수", min_value=3, max_value=12, value=6)
        scan_pool_size = st.slider("시장 스캔 후보 수", min_value=20, max_value=120, value=40, step=10)
        run_analysis = st.button("분석 실행", use_container_width=True)
        run_scan = st.button("시장 기회 스캔", use_container_width=True)
        st.caption(
            f"현재 선택 가능한 종목은 {len(selectable_catalog_df):,}개입니다. "
            f"시장 스캔은 거래가 활발한 상위 {scan_pool_size}개 후보를 먼저 추린 뒤 변동성과 예상 상승률이 높은 종목을 보여줍니다."
        )

        st.divider()
        st.write("사용 API")
        st.write("- 네이버 뉴스 검색 API")
        st.write("- KIS Open API")
        st.write("- pykrx / KIS 종목 마스터")

    missing_keys = ensure_required_keys(settings)
    if missing_keys:
        st.warning(
            "대시보드를 실행하기 전에 `.env` 파일에 필수 API 키를 입력해 주세요.\n\n"
            f"누락 항목: {', '.join(missing_keys)}"
        )
        st.code(
            "NAVER_CLIENT_ID=your_naver_client_id\n"
            "NAVER_CLIENT_SECRET=your_naver_client_secret\n"
            "KIS_APP_KEY=your_kis_app_key\n"
            "KIS_APP_SECRET=your_kis_app_secret"
        )
        return

    if not run_analysis and not run_scan:
        st.info("개별 종목을 보려면 `분석 실행`, 시장 전체 후보를 보려면 `시장 기회 스캔`을 눌러주세요.")
        return

    scan_candidates = build_market_scan_candidates(
        stock_catalog_df=full_stock_catalog_df,
        selected_market=selected_market,
        scan_pool_size=scan_pool_size,
    )

    recommendation_df = pd.DataFrame()
    mover_df = pd.DataFrame()
    if run_scan or run_analysis:
        with st.spinner("시장 후보군의 변동성과 상승 기대를 분석하는 중입니다..."):
            recommendation_df = scan_recommendation_universe(
                candidates=scan_candidates,
                settings=settings,
                news_days=news_days,
                article_limit=article_limit,
            )
            mover_df = build_mover_table(recommendation_df)

    if run_scan:
        st.subheader("고변동성 상승 기대주")
        st.caption(
            f"{selected_market} 시장에서 거래가 활발한 후보 {len(scan_candidates)}개를 먼저 추린 뒤 "
            "가격 변동성과 상승 기대를 함께 반영한 결과입니다."
        )
        render_recommendation_table(recommendation_df.head(recommendation_limit))
        st.markdown("**변동성 상위 후보**")
        render_top_movers_table(mover_df, limit=recommendation_limit)
        if not run_analysis:
            return

    try:
        with st.spinner("주가와 뉴스 데이터를 불러오는 중입니다..."):
            stock_df = load_stock_data(ticker, settings.kis_app_key, settings.kis_app_secret)
            latest_date = pd.to_datetime(stock_df["date"].max())
            news_query = build_news_query(ticker=ticker, company_name=company_name)
            news_df = load_news_data(
                query=news_query,
                client_id=settings.naver_client_id,
                client_secret=settings.naver_client_secret,
                page_size=article_limit,
            )

        news_df = filter_news_window(news_df, latest_date=latest_date, news_days=news_days)
        news_df, feature_df, signal = enrich_news_and_signal(stock_df=stock_df, news_df=news_df)
        current_quote = load_realtime_quote(ticker, settings.kis_app_key, settings.kis_app_secret)
        current_price = float(current_quote.get("current_price", stock_df["close"].iloc[-1]))
        forecast = calculate_price_forecast(stock_df=stock_df, signal=signal, current_price=current_price)
    except Exception as exc:  # noqa: BLE001
        st.error(f"대시보드 데이터를 불러오지 못했습니다: {exc}")
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
        target_profit_pct = max(0.5, float(forecast["expected_return_pct"]) * 100 * 0.5)
        mock_balance = {}
        balance_error = ""
        try:
            mock_balance = inquire_mock_balance(settings)
        except Exception as exc:  # noqa: BLE001
            balance_error = str(exc)

        if mock_balance:
            summary = mock_balance["summary"]
            balance_cols = st.columns(3)
            balance_cols[0].metric("모의 평가금액", f"₩{summary['stock_eval_amount']:,.0f}")
            balance_cols[1].metric("모의 평가손익", f"₩{summary['profit_loss_amount']:,.0f}")
            balance_cols[2].metric("모의 총평가", f"₩{summary['total_eval_amount']:,.0f}")
        elif balance_error:
            st.warning(balance_error)
            st.caption("잔고 조회가 실패해도 로컬 포지션 추적 기반으로 모의 주문과 자동매도 점검은 계속 사용할 수 있습니다.")

        order_cols = st.columns(3)
        buy_quantity = order_cols[0].number_input("매수 수량", min_value=1, value=1, step=1)
        auto_sell_enabled = order_cols[1].checkbox("자동매도 등록", value=float(forecast["expected_return_pct"]) > 0)
        target_profit_override = order_cols[2].number_input("자동매도 목표(%)", min_value=0.1, value=float(target_profit_pct), step=0.1)

        try:
            orderable_cash = inquire_mock_orderable_cash(settings, ticker, current_price)
            st.caption(f"모의 주문 가능 현금: ₩{orderable_cash:,.0f}")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"주문 가능 금액 조회는 생략되었습니다: {exc}")

        action_cols = st.columns(2)
        buy_clicked = action_cols[0].button("현재 종목 모의 매수", use_container_width=True)
        check_clicked = action_cols[1].button("자동매도 지금 점검", use_container_width=True)

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
                        expected_return_pct=float(forecast["expected_return_pct"]) * 100,
                        target_profit_pct=float(target_profit_override),
                        auto_sell_enabled=True,
                        order_payload=buy_payload,
                    )
                st.success("모의 매수 주문이 접수되었습니다.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"모의 매수 주문에 실패했습니다: {exc}")

        auto_sell_actions: list[dict[str, object]] = []
        if check_clicked:
            try:
                auto_sell_actions = evaluate_mock_auto_sell(settings)
            except Exception as exc:  # noqa: BLE001
                st.error(f"자동매도 점검에 실패했습니다: {exc}")

        st.markdown("**자동매도 등록 포지션**")
        active_positions_df = build_active_positions_frame(settings)
        render_mock_positions_table(active_positions_df)
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
            fallback_price=float(stock_df["close"].iloc[-1]),
            currency_symbol=currency_symbol,
        )

    if not recommendation_df.empty:
        st.subheader("시장 스캔")
        st.caption("시장 전체 후보 중 변동성이 크고 상승 기대가 높은 종목들을 함께 보여줍니다.")
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


if __name__ == "__main__":
    main()
