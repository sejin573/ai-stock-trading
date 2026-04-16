from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from services.trading_service import (
    get_reference_price,
    has_mock_trading_config,
    inquire_mock_orderable_cash,
    load_active_positions,
    place_mock_cash_order,
    register_auto_sell_position,
)
from utils.config import get_settings
from utils.helpers import normalize_krx_symbol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "candidate_snapshot.json"
STRATEGY_STATE_PATH = DATA_DIR / "strategy_state.json"
KST = ZoneInfo("Asia/Seoul")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_snapshot() -> dict[str, Any]:
    ensure_data_dir()
    if not SNAPSHOT_PATH.exists():
        return {}
    try:
        payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


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


def get_today_key() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def get_active_symbols() -> set[str]:
    return {normalize_krx_symbol(row["symbol"]) for row in load_active_positions()}


def choose_trade_candidates(
    snapshot: dict[str, Any],
    active_symbols: set[str],
    bought_today: set[str],
    min_final_score: float,
    min_up_probability: float,
    min_expected_return: float,
    min_sentiment: float,
    max_realtime_change: float,
) -> list[dict[str, Any]]:
    candidates = snapshot.get("candidates", [])
    filtered: list[dict[str, Any]] = []

    for row in candidates:
        symbol = normalize_krx_symbol(row.get("symbol", ""))
        if not symbol:
            continue
        if symbol in active_symbols:
            continue
        if symbol in bought_today:
            continue

        final_score = safe_float(row.get("final_score"))
        up_probability_pct = safe_float(row.get("up_probability_pct"))
        expected_return_pct = safe_float(row.get("expected_return_pct"))
        average_sentiment = safe_float(row.get("average_sentiment"))
        realtime_change_rate = safe_float(row.get("realtime_change_rate"))

        if final_score < min_final_score:
            continue
        if up_probability_pct < min_up_probability:
            continue
        if expected_return_pct < min_expected_return:
            continue
        if average_sentiment < min_sentiment:
            continue
        if realtime_change_rate >= max_realtime_change:
            continue

        filtered.append(row)

    return sorted(filtered, key=lambda x: safe_float(x.get("final_score")), reverse=True)


def record_buy(
    state: dict[str, Any],
    symbol: str,
    name: str,
    quantity: int,
    price: float,
    target_profit_pct: float,
    candidate_row: dict[str, Any] | None = None,
) -> None:
    today_key = get_today_key()
    state.setdefault("daily_buys", {})
    state.setdefault("bought_symbols", {})
    state.setdefault("orders", [])

    state["daily_buys"][today_key] = int(state["daily_buys"].get(today_key, 0)) + 1
    state["bought_symbols"].setdefault(today_key, [])
    if symbol not in state["bought_symbols"][today_key]:
        state["bought_symbols"][today_key].append(symbol)

    order_row = {
        "type": "buy",
        "symbol": symbol,
        "name": name,
        "quantity": int(quantity),
        "price": float(price),
        "target_profit_pct": float(target_profit_pct),
        "ordered_at": datetime.now(KST).isoformat(timespec="seconds"),
    }
    if candidate_row:
        for key in [
            "final_score",
            "base_score",
            "up_probability_pct",
            "expected_return_pct",
            "average_sentiment",
            "recent_volatility_pct",
            "realtime_change_rate",
            "impact_score",
            "article_count",
        ]:
            if key in candidate_row:
                order_row[key] = candidate_row[key]
    state["orders"].append(order_row)


def run_once(
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
    ignore_market_hours: bool,
) -> None:
    settings = get_settings()

    if not has_mock_trading_config(settings):
        raise ValueError("모의투자 계좌 설정이 완료되지 않았습니다.")

    if not ignore_market_hours and not is_market_open():
        print("[auto_trader] 장중이 아니므로 매수 실행을 건너뜁니다.")
        return

    snapshot = load_snapshot()
    if not snapshot.get("candidates"):
        print("[auto_trader] candidate_snapshot.json이 비어 있습니다.")
        return

    state = load_strategy_state()
    today_key = get_today_key()
    daily_buy_count = int(state.get("daily_buys", {}).get(today_key, 0))
    bought_today = set(state.get("bought_symbols", {}).get(today_key, []))
    active_symbols = get_active_symbols()

    if len(active_symbols) >= max_positions:
        print("[auto_trader] 최대 보유 종목 수에 도달했습니다.")
        return

    if max_daily_buys > 0 and daily_buy_count >= max_daily_buys:
        print("[auto_trader] 일일 최대 매수 횟수에 도달했습니다.")
        return

    candidates = choose_trade_candidates(
        snapshot=snapshot,
        active_symbols=active_symbols,
        bought_today=bought_today,
        min_final_score=min_final_score,
        min_up_probability=min_up_probability,
        min_expected_return=min_expected_return,
        min_sentiment=min_sentiment,
        max_realtime_change=max_realtime_change,
    )

    if not candidates:
        print("[auto_trader] 현재 조건을 만족하는 매수 후보가 없습니다.")
        return

    for row in candidates:
        symbol = normalize_krx_symbol(row["symbol"])
        name = str(row.get("name", symbol))
        market = str(row.get("market", ""))
        current_price = get_reference_price(settings, symbol)

        if current_price <= 0:
            print(f"[auto_trader] 현재가 조회 실패 {symbol}")
            continue

        try:
            orderable_cash = inquire_mock_orderable_cash(settings, symbol, current_price)
        except Exception as exc:
            print(f"[auto_trader] 주문 가능 금액 조회 실패 {symbol}: {exc}")
            continue

        available_budget = min(float(budget_per_trade), float(orderable_cash))
        quantity = int(available_budget // current_price)

        if quantity < 1:
            print(f"[auto_trader] 예산 부족으로 매수 불가 {symbol}")
            continue

        expected_return_pct = safe_float(row.get("expected_return_pct"))
        target_profit_pct = max(
            min_target_profit,
            min(max_target_profit, expected_return_pct * target_profit_factor),
        )

        try:
            order_payload = place_mock_cash_order(
                settings=settings,
                side="buy",
                symbol=symbol,
                quantity=quantity,
                order_type="market",
            )

            register_auto_sell_position(
                symbol=symbol,
                name=name,
                market=market,
                quantity=quantity,
                entry_price=current_price,
                expected_return_pct=expected_return_pct,
                target_profit_pct=target_profit_pct,
                auto_sell_enabled=True,
                order_payload=order_payload,
            )

            record_buy(
                state=state,
                symbol=symbol,
                name=name,
                quantity=quantity,
                price=current_price,
                target_profit_pct=target_profit_pct,
                candidate_row=row,
            )
            save_strategy_state(state)

            print(
                f"[auto_trader] BUY {name}({symbol}) "
                f"qty={quantity} price={current_price:.2f} "
                f"target_profit={target_profit_pct:.2f}%"
            )
            return

        except Exception as exc:
            print(f"[auto_trader] 주문 실패 {symbol}: {exc}")

    print("[auto_trader] 매수 가능한 종목을 끝까지 찾지 못했습니다.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="자동 매수 실행기")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-daily-buys", type=int, default=0, help="0이면 하루 매수 횟수 제한 없음")
    parser.add_argument("--budget-per-trade", type=float, default=300000)
    parser.add_argument("--min-final-score", type=float, default=55.0)
    parser.add_argument("--min-up-probability", type=float, default=60.0)
    parser.add_argument("--min-expected-return", type=float, default=1.0)
    parser.add_argument("--min-sentiment", type=float, default=0.05)
    parser.add_argument("--max-realtime-change", type=float, default=7.0)
    parser.add_argument("--target-profit-factor", type=float, default=0.6)
    parser.add_argument("--min-target-profit", type=float, default=1.5)
    parser.add_argument("--max-target-profit", type=float, default=4.0)
    parser.add_argument("--ignore-market-hours", action="store_true")
    parser.add_argument("--loop", type=int, default=0, help="초 단위 반복 실행. 0이면 1회 실행")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.loop <= 0:
        run_once(
            max_positions=args.max_positions,
            max_daily_buys=args.max_daily_buys,
            budget_per_trade=args.budget_per_trade,
            min_final_score=args.min_final_score,
            min_up_probability=args.min_up_probability,
            min_expected_return=args.min_expected_return,
            min_sentiment=args.min_sentiment,
            max_realtime_change=args.max_realtime_change,
            target_profit_factor=args.target_profit_factor,
            min_target_profit=args.min_target_profit,
            max_target_profit=args.max_target_profit,
            ignore_market_hours=args.ignore_market_hours,
        )
        return

    while True:
        try:
            run_once(
                max_positions=args.max_positions,
                max_daily_buys=args.max_daily_buys,
                budget_per_trade=args.budget_per_trade,
                min_final_score=args.min_final_score,
                min_up_probability=args.min_up_probability,
                min_expected_return=args.min_expected_return,
                min_sentiment=args.min_sentiment,
                max_realtime_change=args.max_realtime_change,
                target_profit_factor=args.target_profit_factor,
                min_target_profit=args.min_target_profit,
                max_target_profit=args.max_target_profit,
                ignore_market_hours=args.ignore_market_hours,
            )
        except Exception as exc:
            print(f"[auto_trader] error: {exc}")

        print(f"[auto_trader] sleeping {args.loop}s")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
