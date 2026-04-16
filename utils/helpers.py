from __future__ import annotations

import re

import pandas as pd

from utils.config import Settings


def build_news_query(ticker: str, company_name: str) -> str:
    if company_name.strip():
        return company_name.strip()
    return normalize_krx_symbol(ticker)


def ensure_required_keys(settings: Settings) -> list[str]:
    missing_keys: list[str] = []
    if not settings.naver_client_id:
        missing_keys.append("NAVER_CLIENT_ID")
    if not settings.naver_client_secret:
        missing_keys.append("NAVER_CLIENT_SECRET")
    return missing_keys


def is_krx_symbol(symbol: str) -> bool:
    normalized_symbol = symbol.strip().upper()
    return bool(re.fullmatch(r"\d{6}(\.(KS|KQ))?", normalized_symbol))


def normalize_krx_symbol(symbol: str) -> str:
    normalized_symbol = symbol.strip().upper()
    return normalized_symbol.split(".", maxsplit=1)[0]


def get_price_currency_symbol(symbol: str) -> str:
    return "₩" if is_krx_symbol(symbol) else "$"


def compute_recent_return(close_series: pd.Series, periods: int = 5) -> float:
    if len(close_series) <= periods:
        return 0.0
    start_price = float(close_series.iloc[-periods - 1])
    end_price = float(close_series.iloc[-1])
    if start_price == 0:
        return 0.0
    return (end_price / start_price) - 1


def format_percentage(value: float, show_sign: bool = False) -> str:
    if show_sign:
        return f"{value:+.2%}"
    return f"{value:.2%}"
