from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_price_csv(path: str, price_col: str = "Close") -> np.ndarray:
    """Load prices from CSV, sort by date when a date-like column exists, and drop missing values."""
    df = pd.read_csv(path)
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found. Available columns: {list(df.columns)}")

    date_candidates = [col for col in df.columns if col.lower() in {"date", "datetime", "timestamp", "time"}]
    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    prices = pd.to_numeric(df[price_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    prices = prices[np.isfinite(prices)]
    if prices.size < 30:
        raise ValueError("Need at least 30 valid price points.")
    return prices


def split_prices(prices: np.ndarray, train_ratio: float = 0.6, validation_ratio: float = 0.2) -> PriceSplits:
    """Chronologically split prices without shuffling."""
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    if len(prices) < 50:
        raise ValueError("Need at least 50 prices for train/validation/test split.")

    train_end = int(len(prices) * train_ratio)
    validation_end = int(len(prices) * (train_ratio + validation_ratio))
    if train_end < 10 or validation_end <= train_end or validation_end >= len(prices):
        raise ValueError("Invalid split ratios for available price length.")
    return PriceSplits(
        train=prices[:train_end].astype(np.float32),
        validation=prices[train_end:validation_end].astype(np.float32),
        test=prices[validation_end:].astype(np.float32),
    )

