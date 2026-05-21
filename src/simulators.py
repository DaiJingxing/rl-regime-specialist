from __future__ import annotations

import numpy as np


def generate_gbm(
    n_steps: int,
    start_price: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a geometric Brownian motion price path with daily log returns."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=sigma, size=n_steps - 1)
    log_prices = np.empty(n_steps, dtype=np.float32)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + np.cumsum(shocks)
    return np.exp(log_prices).astype(np.float32)


def generate_regime_mix(
    n_steps: int,
    start_price: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a longer synthetic path that alternates bull, bear, and sideways regimes."""
    rng = np.random.default_rng(seed)
    regimes = [
        (0.0010, 0.020),
        (-0.0005, 0.020),
        (0.0000, 0.025),
        (0.0008, 0.018),
        (-0.0003, 0.023),
    ]
    segment = max(20, n_steps // len(regimes))
    prices = []
    current = start_price
    remaining = n_steps
    for idx, (mu, sigma) in enumerate(regimes):
        length = segment if idx < len(regimes) - 1 else remaining
        path = generate_gbm(length, current, mu, sigma, seed=int(rng.integers(1_000_000)))
        if prices:
            path = path[1:]
        prices.extend(path.tolist())
        current = float(path[-1])
        remaining = n_steps - len(prices)
        if remaining <= 0:
            break
    return np.asarray(prices[:n_steps], dtype=np.float32)

