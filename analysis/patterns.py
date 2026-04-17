from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_zero(value: float, digits: int = 3) -> float:
    return round(_safe_float(value), digits)


def analyze_chart_patterns(stock_df: pd.DataFrame) -> dict[str, Any]:
    if stock_df.empty or "close" not in stock_df.columns:
        return {
            "pattern_score": 0.0,
            "pattern_bias": 0.0,
            "pattern_label": "중립",
            "pattern_tags": [],
            "pattern_summary": "차트 데이터가 부족해 패턴을 계산하지 못했습니다.",
        }

    working_df = stock_df.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    close_series = working_df["close"].dropna()
    if len(close_series) < 20:
        return {
            "pattern_score": 0.0,
            "pattern_bias": 0.0,
            "pattern_label": "중립",
            "pattern_tags": [],
            "pattern_summary": "차트 길이가 짧아 패턴 신호를 충분히 만들지 못했습니다.",
        }

    sma5 = close_series.rolling(5).mean()
    sma20 = close_series.rolling(20).mean()
    sma60 = close_series.rolling(60).mean()

    current_close = _safe_float(close_series.iloc[-1])
    prev_close = _safe_float(close_series.iloc[-2]) if len(close_series) >= 2 else current_close
    current_sma5 = _safe_float(sma5.iloc[-1], current_close)
    current_sma20 = _safe_float(sma20.iloc[-1], current_close)
    current_sma60 = _safe_float(sma60.iloc[-1], current_sma20)
    prev_sma20 = _safe_float(sma20.iloc[-2], current_sma20) if len(sma20.dropna()) >= 2 else current_sma20

    high_series = working_df["high"].dropna() if "high" in working_df.columns else close_series
    low_series = working_df["low"].dropna() if "low" in working_df.columns else close_series
    open_series = working_df["open"].dropna() if "open" in working_df.columns else close_series
    volume_series = working_df["volume"].dropna() if "volume" in working_df.columns else pd.Series(dtype=float)

    current_open = _safe_float(open_series.iloc[-1], current_close) if not open_series.empty else current_close
    current_high = _safe_float(high_series.iloc[-1], current_close) if not high_series.empty else current_close
    current_low = _safe_float(low_series.iloc[-1], current_close) if not low_series.empty else current_close

    recent_high_20 = _safe_float(high_series.tail(21).iloc[:-1].max(), current_high) if len(high_series) >= 21 else current_high
    recent_low_20 = _safe_float(low_series.tail(21).iloc[:-1].min(), current_low) if len(low_series) >= 21 else current_low
    avg_volume_20 = _safe_float(volume_series.tail(20).mean(), 0.0) if not volume_series.empty else 0.0
    current_volume = _safe_float(volume_series.iloc[-1], avg_volume_20) if not volume_series.empty else avg_volume_20

    body_size = abs(current_close - current_open)
    upper_wick = max(0.0, current_high - max(current_close, current_open))
    lower_wick = max(0.0, min(current_close, current_open) - current_low)
    candle_range = max(current_high - current_low, 1e-6)
    volume_ratio = current_volume / max(avg_volume_20, 1.0)

    score = 0.0
    tags: list[str] = []

    bullish_alignment = current_close > current_sma5 > current_sma20 and current_sma20 >= current_sma60
    bearish_alignment = current_close < current_sma5 < current_sma20 and current_sma20 <= current_sma60

    if bullish_alignment:
        score += 3.2
        tags.append("정배열 추세")
    elif bearish_alignment:
        score -= 3.4
        tags.append("역배열 약세")

    if current_sma20 > prev_sma20:
        score += 1.2
        tags.append("20일선 상승")
    elif current_sma20 < prev_sma20:
        score -= 1.0

    breakout = current_close > recent_high_20 and volume_ratio >= 1.15
    breakdown = current_close < recent_low_20
    if breakout:
        score += 4.0
        tags.append("20일 돌파")
    if breakdown:
        score -= 4.2
        tags.append("20일 지지 이탈")

    rebound = (
        current_close > current_open
        and current_close > prev_close
        and current_close >= current_sma5
        and current_low <= current_sma20 * 1.01
    )
    if rebound:
        score += 2.0
        tags.append("이평선 반등")

    overextended = current_close >= current_sma20 * 1.12
    if overextended:
        score -= 1.8
        tags.append("단기 과열")

    weak_close = (current_close - current_low) / candle_range <= 0.25
    if weak_close:
        score -= 1.4
        tags.append("종가 약세")

    if lower_wick > body_size * 1.4 and current_close > current_open:
        score += 1.2
        tags.append("아랫꼬리 매수세")

    if upper_wick > body_size * 1.8 and current_close < current_open:
        score -= 1.2
        tags.append("윗꼬리 매도세")

    if volume_ratio >= 1.5 and current_close > prev_close:
        score += 1.0
        tags.append("거래량 증가")
    elif volume_ratio <= 0.65:
        score -= 0.5

    pattern_score = max(-10.0, min(10.0, score))
    pattern_bias = max(-1.0, min(1.0, pattern_score / 10.0))

    if pattern_score >= 4.5:
        pattern_label = "강한 상승 패턴"
    elif pattern_score >= 1.5:
        pattern_label = "완만한 상승 패턴"
    elif pattern_score <= -4.5:
        pattern_label = "강한 하락 패턴"
    elif pattern_score <= -1.5:
        pattern_label = "약한 하락 패턴"
    else:
        pattern_label = "중립"

    if tags:
        summary = f"{pattern_label}: " + ", ".join(tags[:3])
    else:
        summary = "뚜렷한 차트 패턴이 확인되지 않았습니다."

    return {
        "pattern_score": _round_or_zero(pattern_score, 2),
        "pattern_bias": _round_or_zero(pattern_bias, 3),
        "pattern_label": pattern_label,
        "pattern_tags": tags[:4],
        "pattern_summary": summary,
        "volume_ratio": _round_or_zero(volume_ratio, 2),
    }
