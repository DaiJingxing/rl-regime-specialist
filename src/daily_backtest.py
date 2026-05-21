from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import StockEnv
from .metrics import compute_daily_metrics


@dataclass
class DailyBacktestResult:
    metrics: dict
    equity_curve: list[float]
    positions: list[float]
    trades: list[dict]


def _apply_transaction_cost(equity: float, old_position: int, new_position: int, cost_bps: float) -> float:
    if old_position == new_position:
        return equity
    return equity * (1.0 - abs(new_position - old_position) * cost_bps / 10_000.0)


def _finalize(equity_curve: list[float], positions: list[float], trades: list[dict]) -> DailyBacktestResult:
    return DailyBacktestResult(
        metrics=compute_daily_metrics(equity_curve, positions, trades=len(trades)),
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
    )


def slice_daily_result(result: DailyBacktestResult, start_index: int) -> DailyBacktestResult:
    if start_index <= 0:
        return result
    if start_index >= len(result.equity_curve):
        return _finalize([1.0], [0.0], [])
    base = max(float(result.equity_curve[start_index]), 1e-12)
    equity_curve = [float(value) / base for value in result.equity_curve[start_index:]]
    positions = [float(value) for value in result.positions[start_index:]]
    trades = []
    for trade in result.trades:
        if int(trade.get("exit_index", 0)) < start_index:
            continue
        adjusted = dict(trade)
        adjusted["entry_index"] = max(0, int(adjusted.get("entry_index", 0)) - start_index)
        adjusted["exit_index"] = max(0, int(adjusted.get("exit_index", 0)) - start_index)
        trades.append(adjusted)
    return _finalize(equity_curve, positions, trades)


def _cash_growth(annual_cash_rate: float) -> float:
    return float((1.0 + annual_cash_rate) ** (1.0 / 252.0))


def daily_buy_and_hold(prices: np.ndarray, cost_bps: float = 5.0, annual_cash_rate: float = 0.0) -> DailyBacktestResult:
    del annual_cash_rate
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 2:
        return _finalize([1.0], [0.0], [])
    equity = _apply_transaction_cost(1.0, 0, 1, cost_bps)
    equity_curve = [equity]
    positions = [1.0]
    for t in range(1, len(prices)):
        equity *= float(prices[t] / prices[t - 1])
        equity_curve.append(equity)
        positions.append(1.0)
    equity_curve[-1] = _apply_transaction_cost(equity_curve[-1], 1, 0, cost_bps)
    trades = [{"entry_index": 0, "exit_index": len(prices) - 1, "holding_days": len(prices) - 1}]
    return _finalize(equity_curve, positions, trades)


def daily_rule_backtest(
    prices: np.ndarray,
    rule: str,
    episode_length: int = 160,
    take_profit: float = 0.08,
    stop_loss: float = -0.04,
    trail: float = 0.05,
    cost_bps: float = 5.0,
    annual_cash_rate: float = 0.0,
) -> DailyBacktestResult:
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 3:
        return _finalize([1.0], [0.0], [])

    equity = 1.0
    cash_growth = _cash_growth(annual_cash_rate)
    equity_curve = [equity]
    positions = [0.0]
    trades: list[dict] = []
    position = 0
    entry_index = 0
    entry_price = float(prices[0])
    peak = entry_price

    for t in range(1, len(prices)):
        if position == 0 and (t - 1) % episode_length == 0:
            position = 1
            entry_index = t - 1
            entry_price = float(prices[t - 1])
            peak = entry_price
            equity = _apply_transaction_cost(equity, 0, 1, cost_bps)

        if position == 1:
            equity *= float(prices[t] / prices[t - 1])
            price = float(prices[t])
            peak = max(peak, price)
            trade_return = price / entry_price - 1.0
            max_holding_exit = t - entry_index >= episode_length - 1
            if rule == "fixed_tp_sl":
                exit_now = trade_return >= take_profit or trade_return <= stop_loss or max_holding_exit
            elif rule == "trailing_stop":
                exit_now = price / peak - 1.0 <= -trail or max_holding_exit
            else:
                raise ValueError(f"Unknown daily rule: {rule}")

            if exit_now:
                equity = _apply_transaction_cost(equity, 1, 0, cost_bps)
                position = 0
                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": t,
                        "return": trade_return,
                        "holding_days": t - entry_index,
                    }
                )
        elif position == 0:
            equity *= cash_growth

        equity_curve.append(equity)
        positions.append(float(position))

    if position == 1:
        equity_curve[-1] = _apply_transaction_cost(equity_curve[-1], 1, 0, cost_bps)
        trades.append(
            {
                "entry_index": entry_index,
                "exit_index": len(prices) - 1,
                "return": float(prices[-1] / entry_price - 1.0),
                "holding_days": len(prices) - 1 - entry_index,
            }
        )
    return _finalize(equity_curve, positions, trades)


def daily_agent_backtest(
    prices: np.ndarray,
    agent,
    episode_length: int = 160,
    cost_bps: float = 5.0,
    annual_cash_rate: float = 0.0,
) -> DailyBacktestResult:
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 3:
        return _finalize([1.0], [0.0], [])

    equity = 1.0
    cash_growth = _cash_growth(annual_cash_rate)
    equity_curve = [equity]
    positions = [0.0]
    trades: list[dict] = []
    position = 0
    entry_index = 0
    env: StockEnv | None = None
    state = None

    for t in range(1, len(prices)):
        if position == 0 and (t - 1) % episode_length == 0:
            segment = prices[t - 1 : min(len(prices), t - 1 + episode_length)]
            if len(segment) >= 3:
                env = StockEnv(segment)
                state = env.reset()
                position = 1
                entry_index = t - 1
                equity = _apply_transaction_cost(equity, 0, 1, cost_bps)

        if position == 1 and env is not None and state is not None:
            equity *= float(prices[t] / prices[t - 1])
            if hasattr(agent, "act_greedy"):
                action = agent.act_greedy(state)
            else:
                action = agent.act(state, greedy=True)
            state, _, done, info = env.step(action)
            if done:
                equity = _apply_transaction_cost(equity, 1, 0, cost_bps)
                position = 0
                exit_index = entry_index + int(info["exit_step"])
                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": min(exit_index, len(prices) - 1),
                        "return": float(info["trade_return"]),
                        "holding_days": int(info["exit_step"]),
                    }
                )
                env = None
                state = None
        elif position == 0:
            equity *= cash_growth

        equity_curve.append(equity)
        positions.append(float(position))

    if position == 1:
        equity_curve[-1] = _apply_transaction_cost(equity_curve[-1], 1, 0, cost_bps)
        trades.append({"entry_index": entry_index, "exit_index": len(prices) - 1, "holding_days": len(prices) - 1 - entry_index})
    return _finalize(equity_curve, positions, trades)
