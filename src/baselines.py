from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult
from .metrics import compute_metrics


@dataclass
class RuleResult:
    trades: list[dict]
    metrics: dict
    equity_curve: list[float]


def _run_rule(prices: np.ndarray, exit_fn, episode_length: int = 120, stride: int = 20) -> BacktestResult:
    trades: list[dict] = []
    equity_curve = [1.0]
    trade_returns: list[float] = []
    holding_days: list[int] = []

    for start in range(0, max(1, len(prices) - 2), stride):
        end = min(len(prices), start + episode_length)
        if end - start < 3:
            continue
        segment = prices[start:end]
        entry = float(segment[0])
        peak = entry
        exit_offset = len(segment) - 1
        for t in range(1, len(segment)):
            price = float(segment[t])
            peak = max(peak, price)
            if exit_fn(price, entry, peak, t, len(segment)):
                exit_offset = t
                break
        exit_price = float(segment[exit_offset])
        trade_return = exit_price / entry - 1.0
        trades.append(
            {
                "entry_index": start,
                "exit_index": start + exit_offset,
                "entry_price": entry,
                "exit_price": exit_price,
                "return": trade_return,
                "holding_days": exit_offset,
            }
        )
        trade_returns.append(trade_return)
        holding_days.append(exit_offset)
        equity_curve.append(equity_curve[-1] * (1.0 + trade_return))

    return BacktestResult(trades, compute_metrics(trade_returns, holding_days, equity_curve), equity_curve)


def fixed_tp_sl(
    prices: np.ndarray,
    take_profit: float = 0.08,
    stop_loss: float = -0.04,
    episode_length: int = 120,
    stride: int = 20,
) -> BacktestResult:
    return _run_rule(
        prices,
        lambda price, entry, peak, t, n: price / entry - 1.0 >= take_profit or price / entry - 1.0 <= stop_loss,
        episode_length=episode_length,
        stride=stride,
    )


def trailing_stop(prices: np.ndarray, trail: float = 0.05, episode_length: int = 120, stride: int = 20) -> BacktestResult:
    return _run_rule(
        prices,
        lambda price, entry, peak, t, n: price / peak - 1.0 <= -trail,
        episode_length=episode_length,
        stride=stride,
    )


def buy_and_hold(prices: np.ndarray) -> BacktestResult:
    prices = np.asarray(prices, dtype=np.float32)
    ret = float(prices[-1] / prices[0] - 1.0) if len(prices) > 1 else 0.0
    equity_curve = [1.0, 1.0 + ret]
    trades = [
        {
            "entry_index": 0,
            "exit_index": len(prices) - 1,
            "entry_price": float(prices[0]),
            "exit_price": float(prices[-1]),
            "return": ret,
            "holding_days": len(prices) - 1,
        }
    ]
    return BacktestResult(trades, compute_metrics([ret], [len(prices) - 1], equity_curve), equity_curve)


class HardRouterAgent:
    """Simulation-only hard router using simple trend and volatility heuristics."""

    def __init__(self, bull_agent, bear_agent, sideways_agent, trend_threshold: float = 0.02, vol_threshold: float = 0.025) -> None:
        self.bull_agent = bull_agent
        self.bear_agent = bear_agent
        self.sideways_agent = sideways_agent
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold

    def act_greedy(self, state: np.ndarray) -> int:
        sigma_t = float(state[3])
        trend = float(state[5])
        if trend > self.trend_threshold:
            return self.bull_agent.act(state, greedy=True)
        if trend < -self.trend_threshold:
            return self.bear_agent.act(state, greedy=True)
        if sigma_t > self.vol_threshold:
            return self.sideways_agent.act(state, greedy=True)
        return self.sideways_agent.act(state, greedy=True)
