from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.event_tags import add_event_tags
from analysis.features import build_daily_feature_frame
from analysis.forecast import calculate_price_forecast
from analysis.patterns import analyze_chart_patterns
from analysis.scoring import calculate_impact_signal
from analysis.sentiment import add_sentiment_columns
from services.learning_service import apply_learning_to_row, update_learning_from_history
from services.news_service import fetch_company_news
from services.stock_service import (
    fetch_daily_stock_data,
    fetch_kis_realtime_quote,
    get_krx_stock_catalog,
)
from utils.config import get_settings
from utils.helpers import build_news_query, compute_recent_return, is_krx_symbol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "candidate_snapshot.json"
STRATEGY_STATE_PATH = DATA_DIR / "strategy_state.json"
MOCK_TRADE_STATE_PATH = DATA_DIR / "mock_trade_state.json"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(payload: dict[str, Any]) -> None:
    ensure_data_dir()
    SNAPSHOT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_payload(path: Path) -> dict[str, Any]:
    ensure_data_dir()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def filter_news_window(news_df: pd.DataFrame, latest_date: pd.Timestamp, news_days: int) -> pd.DataFrame:
    if news_df.empty:
        return news_df.copy()

    start_bound = latest_date - timedelta(days=news_days)
    return news_df[news_df["published_at"] >= start_bound].copy().reset_index(drop=True)


def build_market_scan_candidates(
    stock_catalog_df: pd.DataFrame,
    selected_market: str,
    scan_pool_size: int,
) -> list[dict[str, Any]]:
    if stock_catalog_df.empty:
        return []

    candidate_df = stock_catalog_df.copy()
    if "symbol" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["symbol"].astype(str).map(is_krx_symbol)].copy()

    if selected_market != "전체" and "market" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["market"] == selected_market].copy()

    if candidate_df.empty:
        return []

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
            "symbol": str(row["symbol"]),
            "name": str(row["name"]),
            "display_name": str(row["name"]),
            "market": str(row.get("market", "")),
            "news_query": str(row.get("news_query", row["name"])),
            "market_cap": safe_float(row.get("market_cap", 0.0)),
            "roe": safe_float(row.get("roe", 0.0)),
        }
        for _, row in candidate_df.iterrows()
    ]


def build_price_only_signal(stock_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_news_df = pd.DataFrame(columns=["published_at", "title", "sentiment_score", "event_tag_list"])
    feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=empty_news_df)
    signal = calculate_impact_signal(stock_df=stock_df, news_df=empty_news_df, feature_df=feature_df)
    signal = {**signal, **analyze_chart_patterns(stock_df)}
    return feature_df, signal


def analyze_price_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    symbol = candidate["symbol"]

    try:
        stock_df = fetch_daily_stock_data(symbol=symbol, kis_app_key="", kis_app_secret="")
        if stock_df.empty:
            return None

        _, signal = build_price_only_signal(stock_df)
        chart_pattern = analyze_chart_patterns(stock_df)
        signal = {**signal, **chart_pattern}
        current_price = safe_float(stock_df["close"].iloc[-1])
        forecast = calculate_price_forecast(
            stock_df=stock_df,
            signal=signal,
            current_price=current_price,
        )

        recent_volatility_pct = (
            safe_float(stock_df["daily_return"].tail(10).std()) * 100
            if "daily_return" in stock_df.columns else 0.0
        )
        expected_return_pct = safe_float(forecast.get("expected_return_pct")) * 100
        up_probability_pct = safe_float(forecast.get("up_probability")) * 100
        recent_return_5d_pct = compute_recent_return(stock_df["close"], periods=5) * 100
        volatility_balance_bonus = compute_volatility_balance_bonus(recent_volatility_pct)
        latest_change_pct = (
            safe_float(stock_df["daily_return"].iloc[-1]) * 100
            if "daily_return" in stock_df.columns else 0.0
        )

        market_cap_score = min(15.0, safe_float(candidate.get("market_cap")) / 15000.0)
        roe_score = max(0.0, min(12.0, safe_float(candidate.get("roe")) / 2.0))
        pattern_score = safe_float(chart_pattern.get("pattern_score"))
        pattern_bias = safe_float(chart_pattern.get("pattern_bias"))
        pattern_label = str(chart_pattern.get("pattern_label", "중립"))

        base_score = (
            expected_return_pct * 3.35
            + up_probability_pct * 0.48
            + recent_volatility_pct * 0.60
            + volatility_balance_bonus
            + safe_float(signal.get("impact_score")) * 0.18
            + max(0.0, recent_return_5d_pct) * 0.9
            + pattern_score * 0.95
            + market_cap_score * 0.2
            + roe_score * 0.2
            - max(0.0, recent_volatility_pct - 9.0) * 0.85
        )

        row = {
            "symbol": symbol,
            "name": candidate["name"],
            "market": candidate["market"],
            "news_query": candidate["news_query"],
            "current_price": round(current_price, 2),
            "realtime_change_rate": round(latest_change_pct, 2),
            "recent_return_5d_pct": round(recent_return_5d_pct, 2),
            "recent_volatility_pct": round(recent_volatility_pct, 2),
            "volatility_balance_bonus": round(volatility_balance_bonus, 2),
            "base_score": round(base_score, 2),
            "impact_score": int(safe_float(signal.get("impact_score"))),
            "expected_return_pct": round(expected_return_pct, 2),
            "up_probability_pct": round(up_probability_pct, 2),
            "pattern_score": round(pattern_score, 2),
            "pattern_bias": round(pattern_bias, 3),
            "chart_pattern": pattern_label,
            "pattern_tags": chart_pattern.get("pattern_tags", []),
            "average_sentiment": 0.0,
            "article_count": 0,
            "top_tags": [],
            "direction": str(forecast.get("direction", "중립")),
        }
        return apply_learning_to_row(row, score_key="base_score")
    except Exception as exc:
        print(f"[scanner] 가격 분석 실패 {symbol}: {exc}")
        return None


def extract_top_tags(news_df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if "event_tag_list" in news_df.columns:
        for value in news_df["event_tag_list"]:
            if isinstance(value, list):
                tags.extend([str(tag).strip() for tag in value if str(tag).strip()])

    if not tags and "event_tags" in news_df.columns:
        for value in news_df["event_tags"].fillna(""):
            for tag in str(value).split(","):
                tag = tag.strip()
                if tag:
                    tags.append(tag)

    return [tag for tag, _ in Counter(tags).most_common(3)]


def enrich_candidate_with_news(
    candidate_row: dict[str, Any],
    news_days: int,
    article_limit: int,
    use_realtime: bool,
) -> dict[str, Any]:
    symbol = candidate_row["symbol"]
    company_name = candidate_row["name"]
    settings = get_settings()

    try:
        stock_df = fetch_daily_stock_data(symbol=symbol, kis_app_key="", kis_app_secret="")
        latest_date = pd.to_datetime(stock_df["date"].max())

        news_query = build_news_query(ticker=symbol, company_name=company_name)
        news_df = fetch_company_news(
            query=news_query,
            client_id=settings.naver_client_id,
            client_secret=settings.naver_client_secret,
            page_size=article_limit,
        )
        news_df = filter_news_window(news_df, latest_date=latest_date, news_days=news_days)

        if news_df.empty:
            return candidate_row

        news_df = add_sentiment_columns(news_df)
        news_df = add_event_tags(news_df)
        feature_df = build_daily_feature_frame(stock_df=stock_df, news_df=news_df)
        signal = calculate_impact_signal(stock_df=stock_df, news_df=news_df, feature_df=feature_df)
        chart_pattern = analyze_chart_patterns(stock_df)
        signal = {**signal, **chart_pattern}

        current_price = candidate_row["current_price"]
        realtime_change_rate = candidate_row["realtime_change_rate"]

        if use_realtime and settings.kis_app_key and settings.kis_app_secret:
            try:
                realtime_quote = fetch_kis_realtime_quote(symbol, settings.kis_app_key, settings.kis_app_secret)
                current_price = safe_float(realtime_quote.get("current_price"), current_price)
                realtime_change_rate = safe_float(realtime_quote.get("change_rate"), realtime_change_rate)
            except Exception:
                pass

        forecast = calculate_price_forecast(
            stock_df=stock_df,
            signal=signal,
            current_price=current_price,
        )

        avg_sentiment = safe_float(signal.get("average_sentiment"))
        impact_score = int(safe_float(signal.get("impact_score")))
        expected_return_pct = safe_float(forecast.get("expected_return_pct")) * 100
        up_probability_pct = safe_float(forecast.get("up_probability")) * 100
        article_count = int(len(news_df))
        top_tags = extract_top_tags(news_df)
        pattern_score = safe_float(chart_pattern.get("pattern_score"))
        pattern_bias = safe_float(chart_pattern.get("pattern_bias"))
        pattern_label = str(chart_pattern.get("pattern_label", "중립"))

        final_score = (
            safe_float(candidate_row["base_score"]) * 0.45
            + impact_score * 0.30
            + up_probability_pct * 0.18
            + max(0.0, avg_sentiment) * 10.0
            + pattern_score * 0.70
            + min(article_count, 15) * 0.35
        )

        enriched = candidate_row.copy()
        enriched.update(
            {
                "current_price": round(current_price, 2),
                "realtime_change_rate": round(realtime_change_rate, 2),
                "impact_score": impact_score,
                "expected_return_pct": round(expected_return_pct, 2),
                "up_probability_pct": round(up_probability_pct, 2),
                "average_sentiment": round(avg_sentiment, 3),
                "article_count": article_count,
                "top_tags": top_tags,
                "pattern_score": round(pattern_score, 2),
                "pattern_bias": round(pattern_bias, 3),
                "chart_pattern": pattern_label,
                "pattern_tags": chart_pattern.get("pattern_tags", []),
                "direction": str(forecast.get("direction", "중립")),
                "final_score": round(final_score, 2),
            }
        )
        return apply_learning_to_row(enriched, score_key="final_score")

    except Exception as exc:
        print(f"[scanner] 뉴스 보강 실패 {symbol}: {exc}")
        candidate_row["final_score"] = candidate_row.get("base_score", 0.0)
        return apply_learning_to_row(candidate_row, score_key="final_score")


def run_once(
    market: str,
    scan_pool_size: int,
    top_n_for_news: int,
    article_limit: int,
    news_days: int,
    max_results: int,
    use_realtime: bool,
) -> None:
    settings = get_settings()
    update_learning_from_history(
        strategy_payload=load_json_payload(STRATEGY_STATE_PATH),
        trade_state_payload=load_json_payload(MOCK_TRADE_STATE_PATH),
    )
    catalog_df = get_krx_stock_catalog()
    candidates = build_market_scan_candidates(catalog_df, market, scan_pool_size)

    if not candidates:
        raise ValueError("스캔할 종목 후보를 만들지 못했습니다.")

    base_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = analyze_price_candidate(candidate)
        if row:
            base_rows.append(row)

    if not base_rows:
        raise ValueError("가격 기반 후보 분석 결과가 비어 있습니다.")

    base_rows = sorted(base_rows, key=lambda x: x["base_score"], reverse=True)
    top_candidates = base_rows[:top_n_for_news]

    final_rows: list[dict[str, Any]] = []
    for row in top_candidates:
        final_rows.append(
            enrich_candidate_with_news(
                candidate_row=row,
                news_days=news_days,
                article_limit=article_limit,
                use_realtime=use_realtime,
            )
        )

    for row in final_rows:
        if "final_score" not in row:
            row["final_score"] = row.get("base_score", 0.0)

    final_rows = sorted(final_rows, key=lambda x: x["final_score"], reverse=True)[:max_results]

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "market": market,
        "scan_pool_size": scan_pool_size,
        "top_n_for_news": top_n_for_news,
        "article_limit": article_limit,
        "news_days": news_days,
        "max_results": max_results,
        "candidates": final_rows,
    }
    save_snapshot(payload)

    print("=" * 80)
    print(f"[scanner] snapshot saved -> {SNAPSHOT_PATH}")
    for idx, row in enumerate(final_rows, start=1):
        print(
            f"{idx}. {row['name']}({row['symbol']}) "
            f"score={row['final_score']:.2f} "
            f"up={row['up_probability_pct']:.1f}% "
            f"ret={row['expected_return_pct']:.2f}% "
            f"sent={row['average_sentiment']:.2f} "
            f"news={row['article_count']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="시장 스캐너")
    parser.add_argument("--market", type=str, default="전체", choices=["전체", "KOSPI", "KOSDAQ", "KONEX"])
    parser.add_argument("--scan-pool-size", type=int, default=40)
    parser.add_argument("--top-n-for-news", type=int, default=8)
    parser.add_argument("--article-limit", type=int, default=10)
    parser.add_argument("--news-days", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--use-realtime", action="store_true")
    parser.add_argument("--loop", type=int, default=0, help="초 단위 반복 실행. 0이면 1회 실행")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.loop <= 0:
        run_once(
            market=args.market,
            scan_pool_size=args.scan_pool_size,
            top_n_for_news=args.top_n_for_news,
            article_limit=args.article_limit,
            news_days=args.news_days,
            max_results=args.max_results,
            use_realtime=args.use_realtime,
        )
        return

    while True:
        try:
            run_once(
                market=args.market,
                scan_pool_size=args.scan_pool_size,
                top_n_for_news=args.top_n_for_news,
                article_limit=args.article_limit,
                news_days=args.news_days,
                max_results=args.max_results,
                use_realtime=args.use_realtime,
            )
        except Exception as exc:
            print(f"[scanner] error: {exc}")

        print(f"[scanner] sleeping {args.loop}s")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
