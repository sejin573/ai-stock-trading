# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from services.trading_service import (
    _close_position,
    build_active_positions_frame,
    evaluate_mock_auto_sell,
    has_mock_trading_config,
    place_mock_cash_order,
)
from services.learning_service import update_learning_from_history
from utils.config import get_settings
from utils.helpers import normalize_krx_symbol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRATEGY_STATE_PATH = DATA_DIR / "strategy_state.json"
MOCK_TRADE_STATE_PATH = DATA_DIR / "mock_trade_state.json"
KST = ZoneInfo("Asia/Seoul")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_strategy_state() -> dict[str, Any]:
    ensure_data_dir()
    if not STRATEGY_STATE_PATH.exists():
        return {"daily_buys": {}, "bought_symbols": {}, "orders": []}
    try:
        payload = json.loads(STRATEGY_STATE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"daily_buys": {}, "bought_symbols": {}, "orders": []}
    except json.JSONDecodeError:
        return {"daily_buys": {}, "bought_symbols": {}, "orders": []}


def save_strategy_state(state: dict[str, Any]) -> None:
    ensure_data_dir()
    STRATEGY_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_payload(path: Path) -> dict[str, Any]:
    ensure_data_dir()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def is_market_open() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False

    current_hhmm = now.hour * 100 + now.minute
    return 905 <= current_hhmm <= 1515


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def record_sell(
    state: dict[str, Any],
    symbol: str,
    name: str,
    reason: str,
    quantity: int,
    price: float,
    ordered_at: str | None = None,
    entry_price: float | None = None,
    realized_pnl: float | None = None,
    realized_pnl_pct: float | None = None,
) -> None:
    state.setdefault("orders", [])
    order_row = {
        "type": "sell",
        "symbol": normalize_krx_symbol(symbol),
        "name": name,
        "reason": reason,
        "quantity": int(quantity),
        "price": float(price),
        "ordered_at": ordered_at or datetime.now(KST).isoformat(timespec="seconds"),
    }
    if entry_price is not None:
        order_row["entry_price"] = float(entry_price)
    if realized_pnl is not None:
        order_row["realized_pnl"] = float(realized_pnl)
    if realized_pnl_pct is not None:
        order_row["realized_pnl_pct"] = float(realized_pnl_pct)
    state["orders"].append(order_row)


def run_once(
    stop_loss_pct: float,
    max_hold_hours: float,
    ignore_market_hours: bool,
) -> None:
    settings = get_settings()

    if not has_mock_trading_config(settings):
        raise ValueError("모의투자 계좌 설정이 완료되지 않았습니다.")

    if not ignore_market_hours and not is_market_open():
        print("[position_monitor] 장중이 아니므로 자동매도 점검을 건너뜁니다.")
        return

    state = load_strategy_state()

    profit_actions = evaluate_mock_auto_sell(settings)
    for action in profit_actions:
        if action.get("status") == "sold":
            record_sell(
                state=state,
                symbol=str(action.get("symbol", "")),
                name=str(action.get("name", "")),
                reason="target_profit",
                quantity=int(safe_float(action.get("quantity"), 0)),
                price=safe_float(action.get("sell_price")),
                ordered_at=str(action.get("closed_at", "")) or None,
                entry_price=safe_float(action.get("entry_price")),
                realized_pnl=safe_float(action.get("realized_pnl")),
                realized_pnl_pct=safe_float(action.get("realized_pnl_pct")),
            )
            print(
                f"[position_monitor] TAKE PROFIT {action.get('name')}({action.get('symbol')}) "
                f"qty={int(safe_float(action.get('quantity'), 0))} "
                f"price={safe_float(action.get('sell_price')):.2f} "
                f"return={safe_float(action.get('realized_pnl_pct')):.2f}%"
            )

    active_df = build_active_positions_frame(settings)
    if active_df.empty:
        save_strategy_state(state)
        print("[position_monitor] active position 없음")
        return

    now = datetime.now(KST)

    for _, row in active_df.iterrows():
        symbol = str(row["symbol"])
        name = str(row["name"])
        position_id = str(row["id"])
        quantity = int(row["quantity"])
        current_price = safe_float(row["current_price"])
        current_return_pct = safe_float(row["current_return_pct"])
        created_at = str(row.get("created_at", ""))

        hold_hours = 0.0
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=KST)
                hold_hours = (now - created_dt.astimezone(KST)).total_seconds() / 3600.0
            except ValueError:
                hold_hours = 0.0

        should_stop_loss = current_return_pct <= -abs(stop_loss_pct)
        should_time_exit = hold_hours >= max_hold_hours

        if not should_stop_loss and not should_time_exit:
            continue

        reason = "stop_loss" if should_stop_loss else "time_exit"

        try:
            sell_payload = place_mock_cash_order(
                settings=settings,
                side="sell",
                symbol=symbol,
                quantity=quantity,
                order_type="market",
            )
            _close_position(
                position_id=position_id,
                sell_payload=sell_payload,
                sell_price=current_price,
                sold_at=now.isoformat(timespec="seconds"),
            )
            record_sell(
                state=state,
                symbol=symbol,
                name=name,
                reason=reason,
                quantity=quantity,
                price=current_price,
            )
            print(
                f"[position_monitor] SELL {reason} {name}({symbol}) "
                f"qty={quantity} price={current_price:.2f} return={current_return_pct:.2f}%"
            )
        except Exception as exc:
            print(f"[position_monitor] 매도 실패 {symbol}: {exc}")

    save_strategy_state(state)
    update_learning_from_history(
        strategy_payload=load_json_payload(STRATEGY_STATE_PATH),
        trade_state_payload=load_json_payload(MOCK_TRADE_STATE_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="포지션 자동 감시기")
    parser.add_argument("--stop-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-hold-hours", type=float, default=24.0)
    parser.add_argument("--ignore-market-hours", action="store_true")
    parser.add_argument("--loop", type=int, default=0, help="초 단위 반복 실행. 0이면 1회 실행")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.loop <= 0:
        run_once(
            stop_loss_pct=args.stop_loss_pct,
            max_hold_hours=args.max_hold_hours,
            ignore_market_hours=args.ignore_market_hours,
        )
        return

    while True:
        try:
            run_once(
                stop_loss_pct=args.stop_loss_pct,
                max_hold_hours=args.max_hold_hours,
                ignore_market_hours=args.ignore_market_hours,
            )
        except Exception as exc:
            print(f"[position_monitor] error: {exc}")

        print(f"[position_monitor] sleeping {args.loop}s")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
