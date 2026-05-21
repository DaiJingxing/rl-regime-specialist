from __future__ import annotations

import numpy as np


def max_drawdown(equity_curve: np.ndarray) -> float:
    equity = np.asarray(equity_curve, dtype=np.float32)
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(peaks, 1e-12) - 1.0
    return float(np.min(drawdowns))


def compute_metrics(trade_returns: list[float], holding_days: list[int], equity_curve: list[float] | np.ndarray) -> dict:
    returns = np.asarray(trade_returns, dtype=np.float32)
    equity = np.asarray(equity_curve, dtype=np.float32)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total_return = float(equity[-1] - 1.0) if equity.size else 0.0
    average_trade_return = float(np.mean(returns)) if returns.size else 0.0
    win_rate = float(np.mean(returns > 0)) if returns.size else 0.0
    profit_loss_ratio = float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else 0.0

    if equity.size > 2:
        equity_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
        volatility = float(np.std(equity_returns))
        sharpe_ratio = float(np.sqrt(252) * np.mean(equity_returns) / volatility) if volatility > 1e-12 else 0.0
    else:
        sharpe_ratio = 0.0

    return {
        "total_return": total_return,
        "average_trade_return": average_trade_return,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown(equity),
        "num_trades": int(returns.size),
        "average_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
    }


def compute_daily_metrics(
    equity_curve: list[float] | np.ndarray,
    positions: list[float] | np.ndarray | None = None,
    trades: int = 0,
    periods_per_year: int = 252,
) -> dict:
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "num_trades": int(trades),
            "turnover": 0.0,
            "time_in_market": 0.0,
        }

    daily_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    total_return = float(equity[-1] / max(equity[0], 1e-12) - 1.0)
    years = max((equity.size - 1) / periods_per_year, 1e-12)
    cagr = float((equity[-1] / max(equity[0], 1e-12)) ** (1.0 / years) - 1.0)
    annual_volatility = float(np.std(daily_returns) * np.sqrt(periods_per_year))
    sharpe_ratio = float(np.sqrt(periods_per_year) * np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) > 1e-12 else 0.0

    downside = daily_returns[daily_returns < 0.0]
    downside_vol = float(np.std(downside) * np.sqrt(periods_per_year)) if downside.size else 0.0
    sortino_ratio = float(np.sqrt(periods_per_year) * np.mean(daily_returns) / np.std(downside)) if downside.size and np.std(downside) > 1e-12 else 0.0
    drawdown = max_drawdown(equity)
    calmar_ratio = float(cagr / abs(drawdown)) if drawdown < -1e-12 else 0.0

    if positions is None:
        position_arr = np.ones_like(equity)
    else:
        position_arr = np.asarray(positions, dtype=np.float64)
    turnover = float(np.sum(np.abs(np.diff(position_arr)))) if position_arr.size > 1 else 0.0
    time_in_market = float(np.mean(position_arr > 0.0)) if position_arr.size else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": drawdown,
        "calmar_ratio": calmar_ratio,
        "num_trades": int(trades),
        "turnover": turnover,
        "time_in_market": time_in_market,
    }


def active_metrics(strategy_equity: list[float] | np.ndarray, benchmark_equity: list[float] | np.ndarray) -> dict:
    strategy = np.asarray(strategy_equity, dtype=np.float64)
    benchmark = np.asarray(benchmark_equity, dtype=np.float64)
    n = min(strategy.size, benchmark.size)
    if n < 3:
        return {"active_return": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}
    strategy_returns = np.diff(strategy[:n]) / np.maximum(strategy[: n - 1], 1e-12)
    benchmark_returns = np.diff(benchmark[:n]) / np.maximum(benchmark[: n - 1], 1e-12)
    active = strategy_returns - benchmark_returns
    tracking_error = float(np.std(active) * np.sqrt(252))
    information_ratio = float(np.sqrt(252) * np.mean(active) / np.std(active)) if np.std(active) > 1e-12 else 0.0
    return {
        "active_return": float(strategy[n - 1] / max(strategy[0], 1e-12) - benchmark[n - 1] / max(benchmark[0], 1e-12)),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }
