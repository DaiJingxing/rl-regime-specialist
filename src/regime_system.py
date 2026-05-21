from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .daily_backtest import DailyBacktestResult, _apply_transaction_cost, _finalize, slice_daily_result
from .env import StockEnv
from .regime_classifier import RegimeClassifier, market_features


@dataclass(frozen=True)
class RegimeEntryConfig:
    bull_threshold: float = 0.45
    bear_threshold: float = 0.50
    sideways_threshold: float = 0.40
    trend_momentum_days: int = 20
    range_drawdown: float = -0.03
    rebound_days: int = 5
    cooldown_days: int = 3
    max_holding_days: int = 60
    risk_exit_threshold: float = 0.65
    range_take_profit: float | None = None
    trend_stop_drawdown: float = -0.08
    use_soft_exit_in_trend: bool = False
    cost_bps: float = 5.0
    annual_cash_rate: float = 0.0
    device: str = "cpu"


def _ma(prices: np.ndarray, t: int, window: int) -> float:
    return float(np.mean(prices[t - window + 1 : t + 1]))


def _ret(prices: np.ndarray, t: int, window: int) -> float:
    if t - window < 0 or prices[t - window] <= 0:
        return 0.0
    return float(prices[t] / prices[t - window] - 1.0)


def _drawdown(prices: np.ndarray, t: int, window: int) -> float:
    high = float(np.max(prices[t - window + 1 : t + 1]))
    return float(prices[t] / max(high, 1e-12) - 1.0)


def _entry_signal(prices: np.ndarray, t: int, classifier: RegimeClassifier, cfg: RegimeEntryConfig) -> tuple[bool, str, np.ndarray]:
    if t < 200:
        return False, "warmup", np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    probs = classifier.probabilities(market_features(prices, t), device=cfg.device)
    bull_p, bear_p, sideways_p = map(float, probs)
    if bear_p >= cfg.bear_threshold:
        return False, "risk_off", probs

    price = float(prices[t])
    ma20 = _ma(prices, t, 20)
    ma60 = _ma(prices, t, 60)
    trend_entry = bull_p >= cfg.bull_threshold and price > ma60 and _ret(prices, t, cfg.trend_momentum_days) > 0.0
    if trend_entry:
        return True, "trend_up", probs

    range_entry = (
        sideways_p >= cfg.sideways_threshold
        and _drawdown(prices, t, 20) <= cfg.range_drawdown
        and _ret(prices, t, cfg.rebound_days) > 0.0
    )
    if range_entry:
        return True, "range_bound", probs

    return False, "no_signal", probs


def regime_entry_soft_exit_backtest(
    prices: np.ndarray,
    classifier: RegimeClassifier,
    exit_agent,
    config: RegimeEntryConfig | None = None,
    evaluation_start: int = 0,
) -> DailyBacktestResult:
    cfg = config or RegimeEntryConfig()
    prices = np.asarray(prices, dtype=np.float32)
    if len(prices) < 203:
        return _finalize([1.0], [0.0], [])

    equity = 1.0
    cash_growth = float((1.0 + cfg.annual_cash_rate) ** (1.0 / 252.0))
    equity_curve = [equity]
    positions = [0.0]
    trades: list[dict] = []
    position = 0
    entry_index = 0
    entry_price = 0.0
    peak_price = 0.0
    entry_regime = "none"
    last_exit = -10_000
    env: StockEnv | None = None
    state = None

    for t in range(1, len(prices)):
        if t < evaluation_start:
            equity *= cash_growth
            equity_curve.append(equity)
            positions.append(0.0)
            continue

        if position == 0 and t >= 200 and t - last_exit >= cfg.cooldown_days:
            should_enter, regime, probs = _entry_signal(prices, t - 1, classifier, cfg)
            if should_enter and t < len(prices) - 2:
                segment = prices[t - 1 : min(len(prices), t - 1 + cfg.max_holding_days)]
                if len(segment) >= 3:
                    env = StockEnv(segment)
                    state = env.reset()
                    position = 1
                    entry_index = t - 1
                    entry_price = float(prices[t - 1])
                    peak_price = entry_price
                    entry_regime = regime
                    equity = _apply_transaction_cost(equity, 0, 1, cfg.cost_bps)

        if position == 1 and env is not None and state is not None:
            equity *= float(prices[t] / prices[t - 1])
            peak_price = max(peak_price, float(prices[t]))
            probs = classifier.probabilities(market_features(prices, max(200, t - 1)), device=cfg.device) if t >= 200 else np.zeros(3)
            if entry_regime == "trend_up" and not cfg.use_soft_exit_in_trend:
                action = StockEnv.HOLD
            elif hasattr(exit_agent, "act_greedy"):
                action = exit_agent.act_greedy(state)
            else:
                action = exit_agent.act(state, greedy=True)
            state, _, done, info = env.step(action)
            trade_return = float(prices[t] / entry_price - 1.0)
            trend_drawdown = float(prices[t] / max(peak_price, 1e-12) - 1.0)
            risk_exit = float(probs[1]) >= cfg.risk_exit_threshold
            range_take_profit = (
                cfg.range_take_profit is not None
                and entry_regime == "range_bound"
                and trade_return >= cfg.range_take_profit
            )
            trend_soft_exit = entry_regime == "trend_up" and cfg.use_soft_exit_in_trend and done
            trend_stop_exit = entry_regime == "trend_up" and trend_drawdown <= cfg.trend_stop_drawdown
            non_trend_soft_exit = entry_regime != "trend_up" and done
            time_exit = t - entry_index >= cfg.max_holding_days - 1
            if risk_exit or range_take_profit or trend_soft_exit or trend_stop_exit or non_trend_soft_exit or time_exit:
                equity = _apply_transaction_cost(equity, 1, 0, cfg.cost_bps)
                position = 0
                last_exit = t
                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": t,
                        "return": trade_return,
                        "holding_days": t - entry_index,
                        "entry_regime": entry_regime,
                        "exit_reason": "risk_exit"
                        if risk_exit
                        else "range_take_profit"
                        if range_take_profit
                        else "trend_stop"
                        if trend_stop_exit
                        else "time_exit"
                        if time_exit
                        else "soft_exit",
                    }
                )
                env = None
                state = None
        elif position == 0:
            equity *= cash_growth

        equity_curve.append(equity)
        positions.append(float(position))

    if position == 1:
        equity_curve[-1] = _apply_transaction_cost(equity_curve[-1], 1, 0, cfg.cost_bps)
        trades.append(
            {
                "entry_index": entry_index,
                "exit_index": len(prices) - 1,
                "return": float(prices[-1] / entry_price - 1.0),
                "holding_days": len(prices) - 1 - entry_index,
                "entry_regime": entry_regime,
                "exit_reason": "end",
            }
        )
    result = _finalize(equity_curve, positions, trades)
    return slice_daily_result(result, evaluation_start)
