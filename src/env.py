from __future__ import annotations

import numpy as np


class StockEnv:
    """Single-position sell-timing environment with actions: 0=hold, 1=sell.

    Each episode starts with one long position at index 0 and ends when the agent sells
    or reaches the final price. Rewards are realized only on exit, with small holding
    and drawdown penalties to discourage unlimited exposure.
    """

    HOLD = 0
    SELL = 1

    def __init__(
        self,
        prices: np.ndarray,
        window: int = 20,
        max_holding_days: int | None = None,
        holding_penalty: float = 0.0002,
        drawdown_penalty: float = 0.01,
    ) -> None:
        prices = np.asarray(prices, dtype=np.float32)
        if prices.ndim != 1 or len(prices) < 3:
            raise ValueError("prices must be a 1D array with at least 3 points")
        self.prices = prices
        self.window = max(2, int(window))
        self.max_holding_days = max_holding_days or len(prices) - 1
        self.holding_penalty = holding_penalty
        self.drawdown_penalty = drawdown_penalty
        self.reset()

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        self.t = 1
        self.entry_price = float(self.prices[0])
        self.peak_price = self.entry_price
        self.done = False
        self.exit_price: float | None = None
        self.exit_step: int | None = None
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Cannot step a finished environment. Call reset().")

        price = float(self.prices[self.t])
        self.peak_price = max(self.peak_price, price)
        forced_exit = self.t >= len(self.prices) - 1 or self.t >= self.max_holding_days
        sell = int(action) == self.SELL or forced_exit

        reward = -self.holding_penalty
        if sell:
            trade_return = price / self.entry_price - 1.0
            drawdown = price / self.peak_price - 1.0
            reward = trade_return + self.drawdown_penalty * drawdown
            self.done = True
            self.exit_price = price
            self.exit_step = self.t
        else:
            self.t += 1

        return self._get_state(), float(reward), self.done, self._info()

    def _get_state(self) -> np.ndarray:
        idx = min(self.t, len(self.prices) - 1)
        price = float(self.prices[idx])
        prev_price = float(self.prices[max(0, idx - 1)])
        r_t = price / prev_price - 1.0 if prev_price > 0 else 0.0
        unrealized = price / self.entry_price - 1.0
        self.peak_price = max(getattr(self, "peak_price", price), price)
        drawdown = price / self.peak_price - 1.0 if self.peak_price > 0 else 0.0
        holding = idx / max(1, self.max_holding_days)

        start = max(0, idx - self.window + 1)
        window_prices = self.prices[start : idx + 1]
        returns = np.diff(window_prices) / window_prices[:-1] if len(window_prices) > 1 else np.array([0.0])
        sigma_t = float(np.std(returns)) if returns.size else 0.0
        trend = price / float(window_prices[0]) - 1.0 if len(window_prices) else 0.0

        return np.array([r_t, drawdown, holding, sigma_t, unrealized, trend], dtype=np.float32)

    def _info(self) -> dict:
        price = float(self.prices[min(self.t, len(self.prices) - 1)])
        return {
            "step": min(self.t, len(self.prices) - 1),
            "price": price,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_step": self.exit_step,
            "trade_return": price / self.entry_price - 1.0,
        }

