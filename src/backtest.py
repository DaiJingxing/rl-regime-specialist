from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import StockEnv
from .metrics import compute_metrics


@dataclass
class BacktestResult:
    trades: list[dict]
    metrics: dict
    equity_curve: list[float]


def backtest(prices: np.ndarray, agent, episode_length: int = 120, stride: int = 20) -> BacktestResult:
    """Run repeated sell-timing episodes across a chronological price series."""
    prices = np.asarray(prices, dtype=np.float32)
    trades: list[dict] = []
    equity_curve = [1.0]
    holding_days: list[int] = []
    trade_returns: list[float] = []

    if len(prices) < 3:
        return BacktestResult(trades=[], metrics=compute_metrics([], [], equity_curve), equity_curve=equity_curve)

    starts = range(0, max(1, len(prices) - 2), stride)
    for start in starts:
        end = min(len(prices), start + episode_length)
        if end - start < 3:
            continue
        env = StockEnv(prices[start:end])
        state = env.reset()
        done = False
        info = {}
        while not done:
            if hasattr(agent, "act_greedy"):
                action = agent.act_greedy(state)
            else:
                action = agent.act(state, greedy=True)
            state, _, done, info = env.step(action)

        trade_return = float(info["trade_return"])
        holding = int(info["exit_step"])
        trade = {
            "entry_index": start,
            "exit_index": start + holding,
            "entry_price": float(env.entry_price),
            "exit_price": float(info["price"]),
            "return": trade_return,
            "holding_days": holding,
        }
        trades.append(trade)
        trade_returns.append(trade_return)
        holding_days.append(holding)
        equity_curve.append(equity_curve[-1] * (1.0 + trade_return))

    return BacktestResult(
        trades=trades,
        metrics=compute_metrics(trade_returns, holding_days, equity_curve),
        equity_curve=equity_curve,
    )

