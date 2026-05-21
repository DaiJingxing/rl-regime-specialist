from __future__ import annotations

import os
from pathlib import Path

_mpl_config = Path("outputs") / ".matplotlib"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_curves(results: dict[str, object], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result.equity_curve, label=name)
    plt.title("Equity Curves")
    plt.xlabel("Trade")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trades(prices: np.ndarray, trades: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", color="black", linewidth=1)
    entries = [trade["entry_index"] for trade in trades]
    exits = [trade["exit_index"] for trade in trades]
    plt.scatter(entries, prices[entries], marker="^", color="green", label="Entry", s=30)
    plt.scatter(exits, prices[exits], marker="v", color="red", label="Exit", s=30)
    plt.title("Trades")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
