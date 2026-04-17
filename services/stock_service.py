from __future__ import annotations

import io
import json
import sys
import types
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.helpers import is_krx_symbol, normalize_krx_symbol

def _load_pykrx_stock_module():
    try:
        from pykrx import stock as stock_module

        return stock_module
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
        if exc.name not in {"matplotlib", "matplotlib.font_manager", "matplotlib.pyplot"}:
            return None

        # pykrx imports matplotlib in __init__ only to configure Korean fonts.
        # Our app uses the stock data module only, so a tiny stub keeps the
        # dependency lightweight and avoids breaking the full stock catalog.
        matplotlib_module = types.ModuleType("matplotlib")
        font_manager_module = types.ModuleType("matplotlib.font_manager")
        pyplot_module = types.ModuleType("matplotlib.pyplot")

        class _FontEntry:
            def __init__(self, fname: str = "", name: str = "") -> None:
                self.fname = fname
                self.name = name

        font_manager_module.FontEntry = _FontEntry
        font_manager_module.fontManager = types.SimpleNamespace(ttflist=[])
        pyplot_module.rcParams = {}
        pyplot_module.rc = lambda *args, **kwargs: None

        matplotlib_module.font_manager = font_manager_module
        matplotlib_module.pyplot = pyplot_module

        sys.modules.setdefault("matplotlib", matplotlib_module)
        sys.modules.setdefault("matplotlib.font_manager", font_manager_module)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_module)

        try:
            from pykrx import stock as stock_module

            return stock_module
        except Exception:  # noqa: BLE001
            return None
    except ImportError:  # pragma: no cover - optional dependency in development
        return None


pykrx_stock = _load_pykrx_stock_module()


KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_MOCK_BASE_URL = "https://openapivts.koreainvestment.com:29443"
KIS_MASTER_BASE_URL = "https://new.real.download.dws.co.kr/common/master"
KIS_TOKEN_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "kis_token_cache.json"
KIS_TOKEN_REFRESH_BUFFER_SECONDS = 600
KIS_MASTER_FILES = {
    "KOSPI": ("kospi_code.mst.zip", "kospi_code.mst", 228),
    "KOSDAQ": ("kosdaq_code.mst.zip", "kosdaq_code.mst", 222),
    "KONEX": ("konex_code.mst.zip", "konex_code.mst", 184),
}
KOSPI_MASTER_WIDTHS = [
    2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8, 15, 21, 2, 7, 1, 1, 1, 1, 1, 9, 9, 9,
    5, 9, 8, 9, 3, 1, 1, 1,
]
KOSDAQ_MASTER_WIDTHS = [
    2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8, 15, 21, 2, 7, 1, 1, 1, 1,
    9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1,
]


def _ensure_token_cache_dir() -> None:
    KIS_TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_token_cache() -> dict[str, Any]:
    _ensure_token_cache_dir()
    if not KIS_TOKEN_CACHE_PATH.exists():
        return {}

    try:
        payload = json.loads(KIS_TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_token_cache(payload: dict[str, Any]) -> None:
    _ensure_token_cache_dir()
    KIS_TOKEN_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _kis_token_cache_key(base_url: str, app_key: str) -> str:
    return f"{base_url}:{app_key}"


def _parse_token_expiry_epoch(payload: dict[str, Any], now: datetime) -> int:
    expires_in = int(float(payload.get("expires_in") or 0))
    token_expired_at = str(payload.get("access_token_token_expired", "")).strip()

    if token_expired_at:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
            try:
                parsed = datetime.strptime(token_expired_at, fmt)
                return int(parsed.timestamp())
            except ValueError:
                continue

    if expires_in > 0:
        return int((now + timedelta(seconds=expires_in)).timestamp())

    return int((now + timedelta(hours=23)).timestamp())


def _build_derived_columns(stock_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = stock_df.sort_values("date").reset_index(drop=True)
    stock_df["daily_return"] = stock_df["close"].pct_change()
    stock_df["ma_5"] = stock_df["close"].rolling(5).mean()
    stock_df["ma_20"] = stock_df["close"].rolling(20).mean()
    stock_df["volume_change"] = stock_df["volume"].pct_change()
    stock_df["volatility_5d"] = stock_df["daily_return"].rolling(5).std()
    return stock_df.dropna(subset=["close"]).reset_index(drop=True)


def _download_kis_master_rows(market: str) -> list[str]:
    zip_name, file_name, _ = KIS_MASTER_FILES[market]
    response = requests.get(f"{KIS_MASTER_BASE_URL}/{zip_name}", timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        raw_bytes = zip_file.read(file_name)

    return raw_bytes.decode("cp949", errors="ignore").splitlines()


def _to_number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_kospi_master_rows(rows: list[str]) -> pd.DataFrame:
    head_lines: list[str] = []
    tail_lines: list[str] = []

    for row in rows:
        cleaned_row = row.rstrip("\r\n")
        if not cleaned_row:
            continue
        head = cleaned_row[:-228]
        tail = cleaned_row[-228:]
        head_lines.append(f"{head[0:9].rstrip()},{head[9:21].rstrip()},{head[21:].strip()}")
        tail_lines.append(tail)

    if not head_lines:
        return pd.DataFrame(columns=["symbol", "standard_code", "name", "market", "news_query", "reference_price", "prev_volume", "market_cap", "roe"])

    head_df = pd.read_csv(io.StringIO("\n".join(head_lines)), header=None, names=["symbol", "standard_code", "name"], encoding="cp949")
    tail_columns = [f"field_{index}" for index in range(len(KOSPI_MASTER_WIDTHS))]
    tail_df = pd.read_fwf(io.StringIO("\n".join(tail_lines)), widths=KOSPI_MASTER_WIDTHS, names=tail_columns)
    parsed_df = pd.concat([head_df, tail_df], axis=1)
    parsed_df["market"] = "KOSPI"
    parsed_df["news_query"] = parsed_df["name"]
    parsed_df["reference_price"] = parsed_df["field_31"].map(_to_number)
    parsed_df["prev_volume"] = parsed_df["field_47"].map(_to_number)
    parsed_df["market_cap"] = parsed_df["field_65"].map(_to_number)
    parsed_df["roe"] = parsed_df["field_63"].map(_to_number)
    return parsed_df[["symbol", "standard_code", "name", "market", "news_query", "reference_price", "prev_volume", "market_cap", "roe"]]


def _parse_kosdaq_master_rows(rows: list[str]) -> pd.DataFrame:
    head_lines: list[str] = []
    tail_lines: list[str] = []

    for row in rows:
        cleaned_row = row.rstrip("\r\n")
        if not cleaned_row:
            continue
        head = cleaned_row[:-222]
        tail = cleaned_row[-222:]
        head_lines.append(f"{head[0:9].rstrip()},{head[9:21].rstrip()},{head[21:].strip()}")
        tail_lines.append(tail)

    if not head_lines:
        return pd.DataFrame(columns=["symbol", "standard_code", "name", "market", "news_query", "reference_price", "prev_volume", "market_cap", "roe"])

    head_df = pd.read_csv(io.StringIO("\n".join(head_lines)), header=None, names=["symbol", "standard_code", "name"], encoding="cp949")
    tail_columns = [f"field_{index}" for index in range(len(KOSDAQ_MASTER_WIDTHS))]
    tail_df = pd.read_fwf(io.StringIO("\n".join(tail_lines)), widths=KOSDAQ_MASTER_WIDTHS, names=tail_columns)
    parsed_df = pd.concat([head_df, tail_df], axis=1)
    parsed_df["market"] = "KOSDAQ"
    parsed_df["news_query"] = parsed_df["name"]
    parsed_df["reference_price"] = parsed_df["field_26"].map(_to_number)
    parsed_df["prev_volume"] = parsed_df["field_42"].map(_to_number)
    parsed_df["market_cap"] = parsed_df["field_60"].map(_to_number)
    parsed_df["roe"] = parsed_df["field_58"].map(_to_number)
    return parsed_df[["symbol", "standard_code", "name", "market", "news_query", "reference_price", "prev_volume", "market_cap", "roe"]]


def _parse_konex_master_rows(rows: list[str]) -> pd.DataFrame:
    parsed_rows: list[dict[str, object]] = []

    for row in rows:
        cleaned_row = row.rstrip("\r\n")
        if not cleaned_row:
            continue

        parsed_rows.append(
            {
                "symbol": cleaned_row[0:9].strip(),
                "standard_code": cleaned_row[9:21].strip(),
                "name": cleaned_row[21:-184].strip(),
                "market": "KONEX",
                "news_query": cleaned_row[21:-184].strip(),
                "reference_price": _to_number(cleaned_row[-182:-173]),
                "prev_volume": _to_number(cleaned_row[-142:-130]),
                "market_cap": _to_number(cleaned_row[-12:-3]),
                "roe": _to_number(cleaned_row[-29:-20]),
            }
        )

    return pd.DataFrame(parsed_rows)


def _parse_kis_master_rows(market: str, rows: list[str]) -> list[dict[str, str]]:
    if market == "KOSPI":
        return _parse_kospi_master_rows(rows).to_dict("records")
    if market == "KOSDAQ":
        return _parse_kosdaq_master_rows(rows).to_dict("records")
    if market == "KONEX":
        return _parse_konex_master_rows(rows).to_dict("records")
    return []


def _load_catalog_from_kis_master(markets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for market in markets:
        if market not in KIS_MASTER_FILES:
            continue
        try:
            master_rows = _download_kis_master_rows(market)
            rows.extend(_parse_kis_master_rows(market, master_rows))
        except Exception:  # noqa: BLE001
            continue

    if not rows:
        return pd.DataFrame(columns=["symbol", "name", "market", "news_query", "reference_price", "prev_volume", "market_cap", "roe"])

    catalog_df = pd.DataFrame(rows)
    catalog_df = catalog_df.drop_duplicates(subset=["symbol"]).sort_values(["market", "name", "symbol"])
    return catalog_df.reset_index(drop=True)


def issue_kis_access_token(
    app_key: str,
    app_secret: str,
    base_url: str = KIS_BASE_URL,
    *,
    force_refresh: bool = False,
) -> str:
    now = datetime.now()
    cache = _load_token_cache()
    cache_key = _kis_token_cache_key(base_url, app_key)
    cached = cache.get(cache_key, {})
    cached_token = str(cached.get("access_token", "")).strip()
    expires_at_epoch = int(float(cached.get("expires_at_epoch") or 0))

    if (
        not force_refresh
        and cached_token
        and expires_at_epoch > int(now.timestamp()) + KIS_TOKEN_REFRESH_BUFFER_SECONDS
    ):
        return cached_token

    headers = {"Content-Type": "application/json; charset=UTF-8"}
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    response = requests.post(
        f"{base_url}/oauth2/tokenP",
        headers=headers,
        json=body,
        timeout=30,
    )
    payload = response.json()
    if response.status_code >= 400:
        message = payload.get("error_description") or payload.get("msg1") or response.text
        raise ValueError(f"KIS 토큰 발급 실패: {message}")

    access_token = payload.get("access_token", "").strip()
    if not access_token:
        message = payload.get("msg1") or payload.get("message") or "KIS access token was not returned."
        raise ValueError(message)

    expires_at_epoch = _parse_token_expiry_epoch(payload, now)
    cache[cache_key] = {
        "access_token": access_token,
        "issued_at": now.isoformat(timespec="seconds"),
        "expires_at_epoch": expires_at_epoch,
        "access_token_token_expired": str(payload.get("access_token_token_expired", "")).strip(),
        "expires_in": int(float(payload.get("expires_in") or 0)),
    }
    _save_token_cache(cache)
    return access_token


def fetch_kis_stock_data(symbol: str, app_key: str, app_secret: str) -> pd.DataFrame:
    ticker = normalize_krx_symbol(symbol)
    access_token = issue_kis_access_token(app_key=app_key, app_secret=app_secret)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010400",
        "custtype": "P",
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": ticker,
        "fid_org_adj_prc": "1",
        "fid_period_div_code": "D",
    }

    response = requests.get(
        f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price",
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    output = payload.get("output", [])
    if not output:
        message = payload.get("msg1") or "KIS did not return daily price data."
        raise ValueError(message)

    stock_df = pd.DataFrame(output)
    column_map = {
        "stck_bsop_date": "date",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_clpr": "close",
        "acml_vol": "volume",
    }
    stock_df = stock_df.rename(columns=column_map)

    required_columns = list(column_map.values())
    missing_columns = [column for column in required_columns if column not in stock_df.columns]
    if missing_columns:
        raise ValueError(f"KIS 응답에서 필요한 컬럼이 누락되었습니다: {', '.join(missing_columns)}")

    stock_df = stock_df[required_columns].copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"], format="%Y%m%d", errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    stock_df[numeric_columns] = stock_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return _build_derived_columns(stock_df)


def fetch_kis_realtime_quote(symbol: str, app_key: str, app_secret: str) -> dict[str, Any]:
    ticker = normalize_krx_symbol(symbol)
    access_token = issue_kis_access_token(app_key=app_key, app_secret=app_secret)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST01010100",
        "custtype": "P",
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": ticker,
    }

    response = requests.get(
        f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    output = payload.get("output", {})

    if not output:
        message = payload.get("msg1") or "KIS did not return realtime quote data."
        raise ValueError(message)

    current_price = pd.to_numeric(output.get("stck_prpr"), errors="coerce")
    change_price = pd.to_numeric(output.get("prdy_vrss"), errors="coerce")
    change_rate = pd.to_numeric(output.get("prdy_ctrt"), errors="coerce")
    open_price = pd.to_numeric(output.get("stck_oprc"), errors="coerce")
    high_price = pd.to_numeric(output.get("stck_hgpr"), errors="coerce")
    low_price = pd.to_numeric(output.get("stck_lwpr"), errors="coerce")
    volume = pd.to_numeric(output.get("acml_vol"), errors="coerce")

    if pd.isna(current_price):
        raise ValueError("KIS 현재가 응답에서 현재가를 확인하지 못했습니다.")

    return {
        "current_price": float(current_price),
        "change_price": float(change_price) if not pd.isna(change_price) else 0.0,
        "change_rate": float(change_rate) if not pd.isna(change_rate) else 0.0,
        "open_price": float(open_price) if not pd.isna(open_price) else None,
        "high_price": float(high_price) if not pd.isna(high_price) else None,
        "low_price": float(low_price) if not pd.isna(low_price) else None,
        "volume": int(volume) if not pd.isna(volume) else None,
    }


def fetch_krx_stock_data(symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    if pykrx_stock is None:
        raise ValueError(
            "한국 주식 조회에는 `pykrx`가 필요합니다. `pip install -r requirements.txt` 후 다시 실행해주세요."
        )

    ticker = normalize_krx_symbol(symbol)
    base_end_date = pd.Timestamp.today().normalize()
    stock_df = pd.DataFrame()

    for offset in range(0, 8):
        end_date = base_end_date - pd.Timedelta(days=offset)
        start_date = end_date - pd.Timedelta(days=lookback_days * 2)
        try:
            stock_df = pykrx_stock.get_market_ohlcv(
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
                ticker,
            )
        except Exception:  # noqa: BLE001
            stock_df = pd.DataFrame()
        if not stock_df.empty:
            break

    if stock_df.empty:
        raise ValueError("pykrx에서 해당 한국 종목의 OHLCV 데이터를 찾지 못했습니다.")

    stock_df = stock_df.reset_index().rename(
        columns={
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        }
    )

    required_columns = ["date", "open", "high", "low", "close", "volume"]
    stock_df = stock_df[required_columns].copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    numeric_columns = ["open", "high", "low", "close", "volume"]
    stock_df[numeric_columns] = stock_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return _build_derived_columns(stock_df)


def fetch_daily_stock_data(
    symbol: str,
    kis_app_key: str = "",
    kis_app_secret: str = "",
) -> pd.DataFrame:
    if not is_krx_symbol(symbol):
        raise ValueError("현재 버전은 한국 주식 티커(예: 005930, 000660) 전용으로 구성되어 있습니다.")

    if kis_app_key and kis_app_secret:
        try:
            return fetch_kis_stock_data(symbol=symbol, app_key=kis_app_key, app_secret=kis_app_secret)
        except Exception:  # noqa: BLE001
            return fetch_krx_stock_data(symbol=symbol)

    return fetch_krx_stock_data(symbol=symbol)


def get_krx_stock_catalog(markets: list[str] | None = None) -> pd.DataFrame:
    if pykrx_stock is None:
        return _load_catalog_from_kis_master(markets or ["KOSPI", "KOSDAQ", "KONEX"])

    target_markets = markets or ["KOSPI", "KOSDAQ", "KONEX"]
    rows: list[dict[str, Any]] = []
    try:
        business_day = pykrx_stock.get_nearest_business_day_in_a_week()

        for market in target_markets:
            try:
                tickers = pykrx_stock.get_market_ticker_list(business_day, market=market)
            except Exception:  # noqa: BLE001
                continue

            for ticker in tickers:
                try:
                    name = pykrx_stock.get_market_ticker_name(ticker)
                except Exception:  # noqa: BLE001
                    continue
                rows.append(
                    {
                        "symbol": ticker,
                        "name": name,
                        "market": market,
                        "news_query": name,
                    }
                )
    except Exception:  # noqa: BLE001
        rows = []

    if not rows:
        return _load_catalog_from_kis_master(target_markets)

    catalog_df = pd.DataFrame(rows)
    catalog_df = catalog_df.drop_duplicates(subset=["symbol"]).sort_values(["market", "name", "symbol"])
    return catalog_df.reset_index(drop=True)


def search_supported_symbols(keywords: str, limit: int = 10) -> pd.DataFrame:
    try:
        catalog_df = get_krx_stock_catalog()
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "market"])

    if catalog_df.empty:
        return pd.DataFrame(columns=["symbol", "name", "market"])

    keyword = keywords.strip().lower()
    filtered_df = catalog_df[
        catalog_df.apply(lambda row: keyword in f"{row['symbol']} {row['name']}".lower(), axis=1)
    ]
    return filtered_df[["symbol", "name", "market"]].head(limit).reset_index(drop=True)
