from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random

import numpy as np
import torch

from .baselines import HardRouterAgent
from .daily_backtest import daily_agent_backtest, daily_buy_and_hold, daily_rule_backtest
from .data_download import DEFAULT_TICKERS, download_stooq_daily, load_downloaded_close_series
from .data_loader import split_prices
from .dqn import DQNConfig, load_agent
from .experiments import Candidate, _load_ensemble
from .metrics import active_metrics
from .regime_classifier import RegimeTrainConfig, load_regime_classifier, train_regime_classifier
from .regime_system import RegimeEntryConfig, regime_entry_soft_exit_backtest
from .train_router import RouterTrainConfig, train_router_on_price_sets
from .train_specialists import train_specialists


@dataclass(frozen=True)
class RegimeSystemSummary:
    data_rows: int
    tickers: list[str]
    exit_candidate: dict
    best_entry_config: dict
    validation_score: float
    test_total_returns: dict[str, float]
    output_dir: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metrics(results: list) -> dict[str, float]:
    keys = results[0].metrics.keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [result.metrics[key] for result in results]
        out[key] = float(np.mean(values)) if isinstance(values[0], float) else int(np.sum(values))
    return out


def _score(metrics_rows: list[dict]) -> float:
    if not metrics_rows:
        return -1e9
    cagr = np.asarray([row["cagr"] for row in metrics_rows], dtype=np.float32)
    sortino = np.asarray([row["sortino_ratio"] for row in metrics_rows], dtype=np.float32)
    calmar = np.asarray([row["calmar_ratio"] for row in metrics_rows], dtype=np.float32)
    maxdd = np.asarray([row["max_drawdown"] for row in metrics_rows], dtype=np.float32)
    active = np.asarray([row.get("active_return", 0.0) for row in metrics_rows], dtype=np.float32)
    time_in_market = np.asarray([row["time_in_market"] for row in metrics_rows], dtype=np.float32)
    exposure_bonus = -np.abs(np.mean(time_in_market) - 0.55)
    return float(
        3.0 * np.mean(cagr)
        + 0.4 * np.mean(sortino)
        + 0.4 * np.mean(calmar)
        + 0.7 * np.mean(maxdd)
        + 1.2 * np.mean(active)
        + 0.3 * exposure_bonus
        - 0.25 * np.std(cagr)
    )


def _entry_grid(cost_bps: float, device: str) -> list[RegimeEntryConfig]:
    configs: list[RegimeEntryConfig] = []
    for bull_threshold in (0.25, 0.35, 0.45):
        for bear_threshold in (0.55, 0.65):
            for max_holding_days in (90, 160, 252):
                for trend_stop_drawdown in (-0.08, -0.12):
                    configs.append(
                        RegimeEntryConfig(
                            bull_threshold=bull_threshold,
                            bear_threshold=bear_threshold,
                            sideways_threshold=0.35,
                            range_drawdown=-0.02,
                            rebound_days=5,
                            cooldown_days=1,
                            max_holding_days=max_holding_days,
                            risk_exit_threshold=0.70,
                            range_take_profit=None,
                            trend_stop_drawdown=trend_stop_drawdown,
                            use_soft_exit_in_trend=False,
                            cost_bps=cost_bps,
                            device=device,
                        )
                    )
    for range_drawdown in (-0.02, -0.04):
        configs.append(
            RegimeEntryConfig(
                bull_threshold=0.45,
                bear_threshold=0.55,
                sideways_threshold=0.35,
                range_drawdown=range_drawdown,
                rebound_days=5,
                cooldown_days=2,
                max_holding_days=60,
                risk_exit_threshold=0.65,
                range_take_profit=None,
                trend_stop_drawdown=-0.08,
                use_soft_exit_in_trend=True,
                cost_bps=cost_bps,
                device=device,
            )
        )
    return configs


def _benchmark_methods(prices: np.ndarray, exit_agent, hard_router, cost_bps: float, annual_cash_rate: float) -> dict:
    return {
        "buy_and_hold": daily_buy_and_hold(prices, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate),
        "exit_only_soft_60": daily_agent_backtest(prices, exit_agent, episode_length=60, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate),
        "trailing_stop_60": daily_rule_backtest(prices, "trailing_stop", episode_length=60, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate),
        "fixed_tp_sl_60": daily_rule_backtest(prices, "fixed_tp_sl", episode_length=60, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate),
        "hard_router_60": daily_agent_backtest(prices, hard_router, episode_length=60, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate),
    }


def run_regime_trading_system(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
    seed: int = 7,
    device: str = "cpu",
    output_dir: str = "outputs/regime_trading_system",
    data_dir: str = "data/stooq",
    cost_bps: float = 5.0,
    annual_cash_rate: float = 0.0368,
) -> RegimeSystemSummary:
    _set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloads = download_stooq_daily(tickers or DEFAULT_TICKERS, data_dir, start=start, end=end, min_rows=500)
    series_by_ticker = load_downloaded_close_series(downloads)
    if len(series_by_ticker) < 3:
        raise RuntimeError("Need at least three usable assets.")
    splits_by_ticker = {ticker: split_prices(s.to_numpy(dtype=np.float32)) for ticker, s in series_by_ticker.items()}
    train_sets = [splits.train for splits in splits_by_ticker.values()]
    validation_sets = {ticker: splits.validation for ticker, splits in splits_by_ticker.items()}
    test_sets = {ticker: splits.test for ticker, splits in splits_by_ticker.items()}

    exit_candidate = Candidate("exit_lr1e-3_g97_h128_l60", 1e-3, 0.97, 128, 80, 1e-3, 90, 60)
    model_dir = str(out / "models" / exit_candidate.name)
    router_path = str(out / "models" / exit_candidate.name / "router.pt")
    dqn_cfg = DQNConfig(device=device, lr=exit_candidate.dqn_lr, gamma=exit_candidate.gamma, hidden_dim=exit_candidate.hidden_dim)
    train_specialists(model_dir, exit_candidate.specialist_episodes, exit_candidate.episode_length, seed, device, config=dqn_cfg)
    train_router_on_price_sets(
        train_sets,
        model_dir,
        router_path,
        RouterTrainConfig(episodes=exit_candidate.router_episodes, gamma=exit_candidate.gamma, lr=exit_candidate.router_lr, device=device, episode_length=60),
        seed=seed,
    )
    exit_agent = _load_ensemble(model_dir, router_path, exit_candidate, device)
    classifier_path = str(out / "models" / "regime_classifier.pt")
    train_regime_classifier(
        train_sets,
        classifier_path,
        RegimeTrainConfig(horizon=60, bull_return=0.06, bear_return=-0.06, epochs=35, lr=1e-3, hidden_dim=64, device=device),
    )
    classifier = load_regime_classifier(classifier_path, device=device)
    cfg = DQNConfig(device=device, lr=exit_candidate.dqn_lr, gamma=exit_candidate.gamma, hidden_dim=exit_candidate.hidden_dim)
    hard_router = HardRouterAgent(
        load_agent(str(Path(model_dir) / "bull_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(model_dir) / "bear_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(model_dir) / "sideways_agent.pt"), cfg, map_location=device),
    )

    validation_rows: list[dict] = []
    best: tuple[float, RegimeEntryConfig] | None = None
    for idx, entry_cfg in enumerate(_entry_grid(cost_bps, device)):
        entry_cfg = RegimeEntryConfig(**{**asdict(entry_cfg), "annual_cash_rate": annual_cash_rate})
        metric_rows = []
        results = []
        for prices in validation_sets.values():
            result = regime_entry_soft_exit_backtest(prices, classifier, exit_agent, entry_cfg)
            benchmark = daily_buy_and_hold(prices, cost_bps=cost_bps, annual_cash_rate=annual_cash_rate)
            metrics = {**result.metrics, **active_metrics(result.equity_curve, benchmark.equity_curve)}
            metric_rows.append(metrics)
            results.append(result)
        score = _score(metric_rows)
        row = {"entry_config": idx, "score": score, **asdict(entry_cfg), **_mean_metrics(results)}
        validation_rows.append(row)
        if best is None or score > best[0]:
            best = (score, entry_cfg)
    if best is None:
        raise RuntimeError("No entry config completed validation.")

    best_score, best_cfg = best
    by_asset_rows: list[dict] = []
    aggregate: dict[str, list] = {
        "regime_entry_soft_exit": [],
        "buy_and_hold": [],
        "exit_only_soft_60": [],
        "trailing_stop_60": [],
        "fixed_tp_sl_60": [],
        "hard_router_60": [],
    }
    for ticker, prices in test_sets.items():
        methods = _benchmark_methods(prices, exit_agent, hard_router, cost_bps, annual_cash_rate)
        methods["regime_entry_soft_exit"] = regime_entry_soft_exit_backtest(prices, classifier, exit_agent, best_cfg)
        benchmark = methods["buy_and_hold"]
        for method, result in methods.items():
            aggregate[method].append(result)
            active = {} if method == "buy_and_hold" else active_metrics(result.equity_curve, benchmark.equity_curve)
            by_asset_rows.append({"ticker": ticker, "method": method, **result.metrics, **active})

    aggregate_rows = [{"method": method, **_mean_metrics(results)} for method, results in aggregate.items()]
    active_rows = []
    for method, results in aggregate.items():
        if method == "buy_and_hold":
            continue
        rows = [
            active_metrics(result.equity_curve, benchmark.equity_curve)
            for result, benchmark in zip(results, aggregate["buy_and_hold"])
        ]
        active_rows.append(
            {
                "method": method,
                "active_return": float(np.mean([row["active_return"] for row in rows])),
                "tracking_error": float(np.mean([row["tracking_error"] for row in rows])),
                "information_ratio": float(np.mean([row["information_ratio"] for row in rows])),
            }
        )

    _write_csv(out / "validation_entry_configs.csv", validation_rows)
    _write_csv(out / "test_metrics_by_asset.csv", by_asset_rows)
    _write_csv(out / "test_metrics_aggregate.csv", aggregate_rows)
    _write_csv(out / "active_vs_buy_hold.csv", active_rows)
    summary = RegimeSystemSummary(
        data_rows=sum(item.rows for item in downloads),
        tickers=list(series_by_ticker),
        exit_candidate=asdict(exit_candidate),
        best_entry_config=asdict(best_cfg),
        validation_score=best_score,
        test_total_returns={row["method"]: row["total_return"] for row in aggregate_rows},
        output_dir=str(out),
    )
    with (out / "summary.json").open("w") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary
