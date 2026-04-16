from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from services.stock_service import (
    KIS_MOCK_BASE_URL,
    fetch_daily_stock_data,
    fetch_kis_realtime_quote,
    issue_kis_access_token,
)
from utils.config import Settings
from utils.helpers import normalize_krx_symbol


TRADE_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_trade_state.json"
TOKEN_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_kis_token_cache.json"

# 보수적으로 1시간 재사용.
# 핵심은 "매 요청마다 새 토큰 발급"을 막는 것.
MOCK_TOKEN_REUSE_SECONDS = 3600


def get_mock_trading_missing_fields(settings: Settings) -> list[str]:
    missing_fields: list[str] = []
    if not settings.kis_mock_app_key:
        missing_fields.append("KIS_MOCK_APP_KEY")
    if not settings.kis_mock_app_secret:
        missing_fields.append("KIS_MOCK_APP_SECRET")
    if not settings.kis_account_no:
        missing_fields.append("KIS_ACCOUNT_NO")
    if not settings.kis_account_product_code:
        missing_fields.append("KIS_ACCOUNT_PRODUCT_CODE")
    return missing_fields


def has_mock_trading_config(settings: Settings) -> bool:
    return not get_mock_trading_missing_fields(settings)


def _ensure_trade_state_dir() -> None:
    TRADE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _ensure_token_cache_dir() -> None:
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_trade_state() -> dict[str, Any]:
    _ensure_trade_state_dir()
    if not TRADE_STATE_PATH.exists():
        return {"positions": [], "history": []}

    try:
        return json.loads(TRADE_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"positions": [], "history": []}


def _save_trade_state(state: dict[str, Any]) -> None:
    _ensure_trade_state_dir()
    TRADE_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_token_cache() -> dict[str, Any]:
    _ensure_token_cache_dir()
    if not TOKEN_CACHE_PATH.exists():
        return {}

    try:
        payload = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_token_cache(state: dict[str, Any]) -> None:
    _ensure_token_cache_dir()
    TOKEN_CACHE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_token_cache_key(settings: Settings) -> str:
    # 앱키별/환경별로 분리
    return f"{KIS_MOCK_BASE_URL}:{settings.kis_mock_app_key}"


def _clear_mock_token_cache(settings: Settings) -> None:
    cache = _load_token_cache()
    cache.pop(_mock_token_cache_key(settings), None)
    _save_token_cache(cache)


def _get_mock_access_token(settings: Settings, force_refresh: bool = False) -> str:
    if not settings.kis_mock_app_key or not settings.kis_mock_app_secret:
        raise ValueError("모의투자 앱키 설정이 완료되지 않았습니다.")

    cache = _load_token_cache()
    cache_key = _mock_token_cache_key(settings)
    now_ts = int(time.time())

    cached = cache.get(cache_key, {})
    cached_token = str(cached.get("access_token", "")).strip()
    issued_at = int(cached.get("issued_at", 0) or 0)

    if not force_refresh and cached_token and (now_ts - issued_at) < MOCK_TOKEN_REUSE_SECONDS:
        return cached_token

    access_token = issue_kis_access_token(
        settings.kis_mock_app_key,
        settings.kis_mock_app_secret,
        base_url=KIS_MOCK_BASE_URL,
    )

    cache[cache_key] = {
        "access_token": access_token,
        "issued_at": now_ts,
    }
    _save_token_cache(cache)
    return access_token


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _kis_error_message(action: str, response: requests.Response, payload: dict[str, Any]) -> str:
    status_code = response.status_code
    rt_cd = str(payload.get("rt_cd", "")).strip()
    msg_cd = str(payload.get("msg_cd", "")).strip()
    msg1 = str(payload.get("msg1", "")).strip()

    detail_parts: list[str] = []
    if status_code >= 400:
        detail_parts.append(f"HTTP {status_code}")
    if rt_cd:
        detail_parts.append(f"rt_cd={rt_cd}")
    if msg_cd:
        detail_parts.append(f"msg_cd={msg_cd}")

    detail = ", ".join(detail_parts)
    if msg1:
        return f"{action}에 실패했습니다. {msg1}" + (f" ({detail})" if detail else "")

    if status_code >= 500 or rt_cd == "1":
        return (
            f"{action}에 실패했습니다. KIS 모의투자 서버가 비정상 응답을 반환했습니다."
            + (f" ({detail})" if detail else "")
        )

    return f"{action}에 실패했습니다." + (f" ({detail})" if detail else "")


def _is_auth_failure(response: requests.Response, payload: dict[str, Any]) -> bool:
    msg1 = str(payload.get("msg1", "")).lower()
    msg_cd = str(payload.get("msg_cd", "")).lower()

    if response.status_code in (401, 403):
        return True

    auth_keywords = ["토큰", "token", "access", "expired", "만료", "인증", "unauthorized"]
    combined = f"{msg1} {msg_cd}"
    return any(keyword in combined for keyword in auth_keywords)


def _request_hashkey(settings: Settings, body: dict[str, Any]) -> str:
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "appKey": settings.kis_mock_app_key,
        "appSecret": settings.kis_mock_app_secret,
    }
    response = requests.post(
        f"{KIS_MOCK_BASE_URL}/uapi/hashkey",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    hashkey = str(payload.get("HASH", "")).strip()
    if not hashkey:
        raise ValueError("KIS hashkey 생성에 실패했습니다.")
    return hashkey


def _mock_headers(
    settings: Settings,
    tr_id: str,
    body: dict[str, Any] | None = None,
    force_refresh_token: bool = False,
) -> dict[str, str]:
    access_token = _get_mock_access_token(settings, force_refresh=force_refresh_token)

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "authorization": f"Bearer {access_token}",
        "appKey": settings.kis_mock_app_key,
        "appSecret": settings.kis_mock_app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }

    if body:
        headers["hashkey"] = _request_hashkey(settings, body)

    return headers


def _mock_get(
    settings: Settings,
    tr_id: str,
    path: str,
    params: dict[str, Any],
    action_name: str,
) -> dict[str, Any]:
    last_response: requests.Response | None = None
    last_payload: dict[str, Any] = {}

    for attempt in range(2):
        headers = _mock_headers(settings, tr_id=tr_id, force_refresh_token=(attempt == 1))
        response = requests.get(
            f"{KIS_MOCK_BASE_URL}{path}",
            headers=headers,
            params=params,
            timeout=30,
        )
        payload = _safe_json(response)

        if (response.status_code < 400 and payload.get("rt_cd") == "0"):
            return payload

        last_response = response
        last_payload = payload

        if attempt == 0 and _is_auth_failure(response, payload):
            _clear_mock_token_cache(settings)
            continue
        break

    if last_response is None:
        raise ValueError(f"{action_name}에 실패했습니다.")

    raise ValueError(_kis_error_message(action_name, last_response, last_payload))


def _mock_post(
    settings: Settings,
    tr_id: str,
    path: str,
    body: dict[str, Any],
    action_name: str,
) -> dict[str, Any]:
    last_response: requests.Response | None = None
    last_payload: dict[str, Any] = {}

    for attempt in range(2):
        headers = _mock_headers(
            settings,
            tr_id=tr_id,
            body=body,
            force_refresh_token=(attempt == 1),
        )
        response = requests.post(
            f"{KIS_MOCK_BASE_URL}{path}",
            headers=headers,
            json=body,
            timeout=30,
        )
        payload = _safe_json(response)

        if (response.status_code < 400 and payload.get("rt_cd") == "0"):
            return payload

        last_response = response
        last_payload = payload

        if attempt == 0 and _is_auth_failure(response, payload):
            _clear_mock_token_cache(settings)
            continue
        break

    if last_response is None:
        raise ValueError(f"{action_name}에 실패했습니다.")

    raise ValueError(_kis_error_message(action_name, last_response, last_payload))


def _get_quote_credentials(settings: Settings) -> tuple[str, str]:
    if settings.kis_app_key and settings.kis_app_secret:
        return settings.kis_app_key, settings.kis_app_secret
    return settings.kis_mock_app_key, settings.kis_mock_app_secret


def get_reference_price(settings: Settings, symbol: str) -> float:
    normalized_symbol = normalize_krx_symbol(symbol)
    quote_app_key, quote_app_secret = _get_quote_credentials(settings)

    if quote_app_key and quote_app_secret:
        try:
            quote = fetch_kis_realtime_quote(normalized_symbol, quote_app_key, quote_app_secret)
            return float(quote["current_price"])
        except Exception:
            pass

    stock_df = fetch_daily_stock_data(normalized_symbol, kis_app_key="", kis_app_secret="")
    return float(stock_df["close"].iloc[-1])


def inquire_mock_balance(settings: Settings) -> dict[str, Any]:
    if not has_mock_trading_config(settings):
        raise ValueError("모의투자 계좌 설정이 완료되지 않았습니다.")

    params = {
        "CANO": settings.kis_account_no,
        "ACNT_PRDT_CD": settings.kis_account_product_code,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    payload = _mock_get(
        settings=settings,
        tr_id="VTTC8434R",
        path="/uapi/domestic-stock/v1/trading/inquire-balance",
        params=params,
        action_name="모의 잔고 조회",
    )

    holdings: list[dict[str, Any]] = []
    for row in payload.get("output1", []):
        quantity = int(float(row.get("hldg_qty") or 0))
        if quantity <= 0:
            continue
        holdings.append(
            {
                "symbol": row.get("pdno", ""),
                "name": row.get("prdt_name", ""),
                "quantity": quantity,
                "avg_price": float(row.get("pchs_avg_pric") or 0),
                "eval_amount": float(row.get("evlu_amt") or 0),
                "profit_loss_amount": float(row.get("evlu_pfls_amt") or 0),
                "profit_loss_rate": float(row.get("evlu_pfls_rt") or 0),
            }
        )

    summary_row = payload.get("output2", [{}])
    summary = summary_row[0] if summary_row else {}
    return {
        "holdings": holdings,
        "summary": {
            "stock_eval_amount": float(summary.get("scts_evlu_amt") or 0),
            "profit_loss_amount": float(summary.get("evlu_pfls_smtl_amt") or 0),
            "total_eval_amount": float(summary.get("tot_evlu_amt") or 0),
            "cash_balance_amount": float(summary.get("dnca_tot_amt") or 0),
            "deposit_received_amount": float(summary.get("tot_dncl_amt") or 0),
        },
    }


def inquire_mock_orderable_cash(settings: Settings, symbol: str, price: float) -> float:
    if not has_mock_trading_config(settings):
        raise ValueError("모의투자 계좌 설정이 완료되지 않았습니다.")

    params = {
        "CANO": settings.kis_account_no,
        "ACNT_PRDT_CD": settings.kis_account_product_code,
        "PDNO": normalize_krx_symbol(symbol),
        "ORD_UNPR": str(int(price)),
        "ORD_DVSN": "00",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "Y",
    }

    payload = _mock_get(
        settings=settings,
        tr_id="VTTC8908R",
        path="/uapi/domestic-stock/v1/trading/inquire-psbl-order",
        params=params,
        action_name="모의 주문 가능 금액 조회",
    )
    return float(payload.get("output", {}).get("ord_psbl_cash") or 0)


def place_mock_cash_order(
    settings: Settings,
    side: str,
    symbol: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict[str, Any]:
    if not has_mock_trading_config(settings):
        raise ValueError("모의투자 계좌 설정이 완료되지 않았습니다.")
    if side not in {"buy", "sell"}:
        raise ValueError("side는 buy 또는 sell 이어야 합니다.")
    if quantity <= 0:
        raise ValueError("주문 수량은 1주 이상이어야 합니다.")

    normalized_symbol = normalize_krx_symbol(symbol)
    order_division = "01" if order_type == "market" else "00"
    order_price = "0" if order_type == "market" else str(int(limit_price or 0))
    if order_type != "market" and int(order_price) <= 0:
        raise ValueError("지정가 주문에는 유효한 가격이 필요합니다.")

    body = {
        "CANO": settings.kis_account_no,
        "ACNT_PRDT_CD": settings.kis_account_product_code,
        "PDNO": normalized_symbol,
        "ORD_DVSN": order_division,
        "ORD_QTY": str(quantity),
        "ORD_UNPR": order_price,
        "EXCG_ID_DVSN_CD": "KRX",
        "SLL_TYPE": "",
        "CNDT_PRIC": "",
    }

    tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
    return _mock_post(
        settings=settings,
        tr_id=tr_id,
        path="/uapi/domestic-stock/v1/trading/order-cash",
        body=body,
        action_name="모의 주문",
    )


def register_auto_sell_position(
    symbol: str,
    name: str,
    market: str,
    quantity: int,
    entry_price: float,
    expected_return_pct: float,
    target_profit_pct: float,
    auto_sell_enabled: bool,
    order_payload: dict[str, Any],
) -> None:
    state = _load_trade_state()
    position = {
        "id": f"{normalize_krx_symbol(symbol)}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "symbol": normalize_krx_symbol(symbol),
        "name": name,
        "market": market,
        "quantity": int(quantity),
        "entry_price": float(entry_price),
        "expected_return_pct": float(expected_return_pct),
        "target_profit_pct": float(target_profit_pct),
        "auto_sell_enabled": bool(auto_sell_enabled),
        "status": "active",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "buy_order_response": order_payload,
    }
    state.setdefault("positions", []).append(position)
    _save_trade_state(state)


def load_active_positions() -> list[dict[str, Any]]:
    state = _load_trade_state()
    return [row for row in state.get("positions", []) if row.get("status") == "active"]


def build_active_positions_frame(settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position in load_active_positions():
        current_price = get_reference_price(settings, position["symbol"])
        entry_price = float(position["entry_price"])
        current_return_pct = ((current_price / entry_price) - 1) * 100 if entry_price else 0.0
        rows.append(
            {
                "id": position["id"],
                "symbol": position["symbol"],
                "name": position["name"],
                "market": position.get("market", ""),
                "quantity": int(position["quantity"]),
                "entry_price": entry_price,
                "current_price": current_price,
                "expected_return_pct": float(position["expected_return_pct"]),
                "target_profit_pct": float(position["target_profit_pct"]),
                "current_return_pct": current_return_pct,
                "auto_sell_enabled": bool(position.get("auto_sell_enabled", False)),
                "created_at": position.get("created_at", ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "symbol",
                "name",
                "market",
                "quantity",
                "entry_price",
                "current_price",
                "expected_return_pct",
                "target_profit_pct",
                "current_return_pct",
                "auto_sell_enabled",
                "created_at",
            ]
        )

    return pd.DataFrame(rows).sort_values(["current_return_pct", "target_profit_pct"], ascending=[False, False])


def _close_position(position_id: str, sell_payload: dict[str, Any], sell_price: float, sold_at: str) -> None:
    state = _load_trade_state()
    for row in state.get("positions", []):
        if row.get("id") != position_id:
            continue
        row["status"] = "closed"
        row["closed_at"] = sold_at
        row["sell_price"] = float(sell_price)
        row["sell_order_response"] = sell_payload
        state.setdefault("history", []).append(row.copy())
        break

    state["positions"] = [row for row in state.get("positions", []) if row.get("status") == "active"]
    _save_trade_state(state)


def close_mock_position(position_id: str, sell_payload: dict[str, Any], sell_price: float, sold_at: str) -> None:
    _close_position(
        position_id=position_id,
        sell_payload=sell_payload,
        sell_price=sell_price,
        sold_at=sold_at,
    )


def evaluate_mock_auto_sell(settings: Settings) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    active_positions = load_active_positions()

    for position in active_positions:
        if not position.get("auto_sell_enabled", False):
            continue

        current_price = get_reference_price(settings, position["symbol"])
        entry_price = float(position["entry_price"])
        current_return_pct = ((current_price / entry_price) - 1) * 100 if entry_price else 0.0
        target_profit_pct = float(position["target_profit_pct"])

        if current_return_pct < target_profit_pct:
            actions.append(
                {
                    "symbol": position["symbol"],
                    "name": position["name"],
                    "quantity": int(position["quantity"]),
                    "entry_price": entry_price,
                    "status": "waiting",
                    "current_return_pct": current_return_pct,
                    "target_profit_pct": target_profit_pct,
                }
            )
            continue

        sell_payload = place_mock_cash_order(
            settings=settings,
            side="sell",
            symbol=position["symbol"],
            quantity=int(position["quantity"]),
            order_type="market",
        )
        closed_at = datetime.now().isoformat(timespec="seconds")
        _close_position(position["id"], sell_payload, current_price, closed_at)
        quantity = int(position["quantity"])
        realized_pnl = (current_price - entry_price) * quantity

        actions.append(
            {
                "symbol": position["symbol"],
                "name": position["name"],
                "quantity": quantity,
                "entry_price": entry_price,
                "status": "sold",
                "current_return_pct": current_return_pct,
                "target_profit_pct": target_profit_pct,
                "sell_price": current_price,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": current_return_pct,
                "closed_at": closed_at,
            }
        )

    return actions
