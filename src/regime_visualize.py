from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rl-regime-specialist-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import HardRouterAgent
from .daily_backtest import DailyBacktestResult, daily_agent_backtest, daily_buy_and_hold, daily_rule_backtest
from .data_loader import split_prices
from .dqn import DQNConfig, load_agent
from .experiments import Candidate, _load_ensemble
from .regime_classifier import load_regime_classifier
from .regime_system import RegimeEntryConfig, regime_entry_soft_exit_backtest


METHOD_LABELS = {
    "regime_entry_soft_exit": "RL Regime Specialist",
    "buy_and_hold": "Buy & Hold",
    "exit_only_soft_60": "Exit-Only Soft Router",
    "trailing_stop_60": "Trailing Stop",
    "fixed_tp_sl_60": "Fixed TP/SL",
    "hard_router_60": "Hard Router",
}

METHOD_COLORS = {
    "regime_entry_soft_exit": "#2563eb",
    "buy_and_hold": "#111827",
    "exit_only_soft_60": "#7c3aed",
    "trailing_stop_60": "#f97316",
    "fixed_tp_sl_60": "#16a34a",
    "hard_router_60": "#64748b",
}


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#525252")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def _mean_nav(results: list[DailyBacktestResult]) -> np.ndarray:
    min_len = min(len(result.equity_curve) for result in results)
    curves = np.asarray([result.equity_curve[:min_len] for result in results], dtype=np.float64)
    return np.mean(curves, axis=0)


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    preferred = [
        "regime_entry_soft_exit",
        "buy_and_hold",
        "exit_only_soft_60",
        "trailing_stop_60",
        "fixed_tp_sl_60",
        "hard_router_60",
    ]
    present = set(df["method"])
    return [method for method in preferred if method in present] + sorted(present - set(preferred))


def plot_metric_bars(df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(df)
    values = [float(df.loc[df["method"] == method, metric].iloc[0]) for method in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(x, values, color=[_color(method) for method in methods], width=0.58)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(method) for method in methods], rotation=12, ha="right")
    _style_axes(ax)
    for bar, value in zip(bars, values):
        text = f"{value * 100:.1f}%" if metric in {"total_return", "cagr", "max_drawdown", "time_in_market"} else f"{value:.2f}"
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_nav(method_results: dict[str, list[DailyBacktestResult]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for method, results in method_results.items():
        nav = _mean_nav(results)
        ax.plot(nav, label=_label(method), color=_color(method), linewidth=2.2)
    ax.set_title("Average NAV on Test Set", fontsize=14, weight="bold")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Net asset value")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_return_and_sharpe(df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(df)
    labels = [_label(method) for method in methods]
    total_returns = [float(df.loc[df["method"] == method, "total_return"].iloc[0]) * 100.0 for method in methods]
    sharpes = [float(df.loc[df["method"] == method, "sharpe_ratio"].iloc[0]) for method in methods]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, total_returns, width, label="Total return", color="#2563eb")
    ax2.bar(x + width / 2, sharpes, width, label="Sharpe", color="#14b8a6")
    ax1.set_title("Return and Sharpe on Test Set", fontsize=14, weight="bold")
    ax1.set_ylabel("Total return (%)")
    ax2.set_ylabel("Sharpe ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    _style_axes(ax1)
    ax2.spines["top"].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_regime_research_charts(output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    aggregate_path = out / "test_metrics_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics: {aggregate_path}")
    df = pd.read_csv(aggregate_path)
    figure_dir = out / "figures"
    paths = {
        "return": figure_dir / "returns_test.png",
        "sharpe": figure_dir / "sharpe_test.png",
        "drawdown": figure_dir / "drawdown_test.png",
        "return_sharpe": figure_dir / "return_sharpe_test.png",
    }
    plot_metric_bars(df, "total_return", "Total Return on Test Set", "Total return", paths["return"])
    plot_metric_bars(df, "sharpe_ratio", "Sharpe Ratio on Test Set", "Sharpe ratio", paths["sharpe"])
    plot_metric_bars(df, "max_drawdown", "Maximum Drawdown on Test Set", "Max drawdown", paths["drawdown"])
    plot_return_and_sharpe(df, paths["return_sharpe"])
    return {name: str(path) for name, path in paths.items()}


def _load_close_prices(path: Path) -> np.ndarray:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if close.size < 50:
        raise ValueError(f"Not enough close prices in {path}")
    return close


def _ticker_path(data_dir: Path, ticker: str) -> Path:
    return data_dir / f"{ticker.replace('.', '_').lower()}.csv"


def rebuild_regime_results(output_dir: str | Path, data_dir: str | Path = "data/stooq") -> dict[str, list[DailyBacktestResult]]:
    out = Path(output_dir)
    data_path = Path(data_dir)
    summary = pd.read_json(out / "summary.json", typ="series")
    tickers = list(summary["tickers"])
    exit_candidate_data = dict(summary["exit_candidate"])
    entry_config_data = dict(summary["best_entry_config"])

    exit_candidate = Candidate(**exit_candidate_data)
    device = entry_config_data.get("device", "cpu")
    model_dir = str(out / "models" / exit_candidate.name)
    router_path = str(out / "models" / exit_candidate.name / "router.pt")
    exit_agent = _load_ensemble(model_dir, router_path, exit_candidate, device)
    classifier = load_regime_classifier(str(out / "models" / "regime_classifier.pt"), device=device)
    cfg = DQNConfig(device=device, lr=exit_candidate.dqn_lr, gamma=exit_candidate.gamma, hidden_dim=exit_candidate.hidden_dim)
    hard_router = HardRouterAgent(
        load_agent(str(Path(model_dir) / "bull_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(model_dir) / "bear_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(model_dir) / "sideways_agent.pt"), cfg, map_location=device),
    )
    entry_cfg = RegimeEntryConfig(**entry_config_data)

    methods: dict[str, list[DailyBacktestResult]] = {
        "regime_entry_soft_exit": [],
        "buy_and_hold": [],
        "exit_only_soft_60": [],
        "trailing_stop_60": [],
        "fixed_tp_sl_60": [],
        "hard_router_60": [],
    }
    for ticker in tickers:
        prices = _load_close_prices(_ticker_path(data_path, ticker))
        test_prices = split_prices(prices).test
        methods["regime_entry_soft_exit"].append(
            regime_entry_soft_exit_backtest(test_prices, classifier, exit_agent, entry_cfg)
        )
        methods["buy_and_hold"].append(
            daily_buy_and_hold(test_prices, cost_bps=entry_cfg.cost_bps, annual_cash_rate=entry_cfg.annual_cash_rate)
        )
        methods["exit_only_soft_60"].append(
            daily_agent_backtest(test_prices, exit_agent, episode_length=60, cost_bps=entry_cfg.cost_bps, annual_cash_rate=entry_cfg.annual_cash_rate)
        )
        methods["trailing_stop_60"].append(
            daily_rule_backtest(test_prices, "trailing_stop", episode_length=60, cost_bps=entry_cfg.cost_bps, annual_cash_rate=entry_cfg.annual_cash_rate)
        )
        methods["fixed_tp_sl_60"].append(
            daily_rule_backtest(test_prices, "fixed_tp_sl", episode_length=60, cost_bps=entry_cfg.cost_bps, annual_cash_rate=entry_cfg.annual_cash_rate)
        )
        methods["hard_router_60"].append(
            daily_agent_backtest(test_prices, hard_router, episode_length=60, cost_bps=entry_cfg.cost_bps, annual_cash_rate=entry_cfg.annual_cash_rate)
        )
    return methods


def write_nav_data(method_results: dict[str, list[DailyBacktestResult]], output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    navs = {method: _mean_nav(results) for method, results in method_results.items()}
    min_len = min(len(nav) for nav in navs.values())
    df = pd.DataFrame({"day": np.arange(min_len)})
    for method, nav in navs.items():
        df[method] = nav[:min_len]
    df.to_csv(output, index=False)
    return str(output)


def write_full_regime_visualization(output_dir: str | Path, data_dir: str | Path = "data/stooq") -> dict[str, str]:
    out = Path(output_dir)
    aggregate_df = pd.read_csv(out / "test_metrics_aggregate.csv")
    method_results = rebuild_regime_results(out, data_dir=data_dir)
    figure_dir = out / "figures"
    paths = {
        "nav": figure_dir / "nav_test.png",
        "return": figure_dir / "returns_test.png",
        "sharpe": figure_dir / "sharpe_test.png",
        "drawdown": figure_dir / "drawdown_test.png",
        "return_sharpe": figure_dir / "return_sharpe_test.png",
        "nav_csv": out / "average_nav_test.csv",
    }
    plot_nav(method_results, paths["nav"])
    plot_metric_bars(aggregate_df, "total_return", "Total Return on Test Set", "Total return", paths["return"])
    plot_metric_bars(aggregate_df, "sharpe_ratio", "Sharpe Ratio on Test Set", "Sharpe ratio", paths["sharpe"])
    plot_metric_bars(aggregate_df, "max_drawdown", "Maximum Drawdown on Test Set", "Max drawdown", paths["drawdown"])
    plot_return_and_sharpe(aggregate_df, paths["return_sharpe"])
    write_nav_data(method_results, paths["nav_csv"])
    return {name: str(path) for name, path in paths.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create charts from RL Regime Specialist backtest outputs.")
    parser.add_argument("output_dir", nargs="?", default="outputs/regime_trading_system_warmup_start_cash_tune")
    args = parser.parse_args()
    for name, path in write_full_regime_visualization(args.output_dir).items():
        print(f"{name}={path}")
