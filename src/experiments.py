from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random

import numpy as np
import torch

from .backtest import BacktestResult
from .baselines import HardRouterAgent
from .data_download import DEFAULT_TICKERS, download_stooq_daily, load_downloaded_close_series
from .data_loader import split_prices
from .daily_backtest import DailyBacktestResult, daily_agent_backtest, daily_buy_and_hold, daily_rule_backtest
from .dqn import DQNConfig, load_agent
from .metrics import active_metrics
from .plots import plot_equity_curves
from .router import EnsembleAgent
from .train_router import RouterTrainConfig, load_router, train_router_on_price_sets
from .train_specialists import train_specialists


@dataclass(frozen=True)
class Candidate:
    name: str
    dqn_lr: float
    gamma: float
    hidden_dim: int
    specialist_episodes: int
    router_lr: float
    router_episodes: int
    episode_length: int


@dataclass(frozen=True)
class ExperimentSummary:
    data_rows: int
    tickers: list[str]
    best_candidate: dict
    validation_score: float
    test_metrics: dict[str, float]
    output_dir: str


def candidate_grid(preset: str, device: str, episode_lengths: list[int] | None = None) -> list[Candidate]:
    del device
    lengths = episode_lengths or []
    if preset == "quick":
        quick_lengths = lengths or [140]
        return [
            Candidate(f"q_lr1e-3_g99_l{length}", 1e-3, 0.99, 64, 35, 1e-3, 35, length)
            for length in quick_lengths
        ] + [
            Candidate(f"q_lr5e-4_g97_l{length}", 5e-4, 0.97, 64, 35, 5e-4, 35, length)
            for length in quick_lengths
        ]
    if preset == "thorough":
        specialist_episodes = 140
        router_episodes = 160
        thorough_lengths = lengths or [180]
        return [
            Candidate(f"t_lr{lr}_g{gamma}_h{hidden}_l{length}", lr, gamma, hidden, specialist_episodes, lr, router_episodes, length)
            for lr in (1e-3, 5e-4)
            for gamma in (0.97, 0.99)
            for hidden in (64, 128)
            for length in thorough_lengths
        ]
    balanced_lengths = lengths or [160]
    return [
        Candidate(f"b_lr1e-3_g99_h64_l{length}", 1e-3, 0.99, 64, 80, 1e-3, 90, length)
        for length in balanced_lengths
    ] + [
        Candidate(f"b_lr5e-4_g99_h64_l{length}", 5e-4, 0.99, 64, 80, 5e-4, 90, length)
        for length in balanced_lengths
    ] + [
        Candidate(f"b_lr1e-3_g97_h128_l{length}", 1e-3, 0.97, 128, 80, 1e-3, 90, length)
        for length in balanced_lengths
    ] + [
        Candidate(f"b_lr5e-4_g97_h128_l{length}", 5e-4, 0.97, 128, 80, 5e-4, 90, length)
        for length in balanced_lengths
    ]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _score(metrics_by_asset: list[dict]) -> float:
    if not metrics_by_asset:
        return -1e9
    sharpe = np.asarray([m["sharpe_ratio"] for m in metrics_by_asset], dtype=np.float32)
    cagr = np.asarray([m["cagr"] for m in metrics_by_asset], dtype=np.float32)
    drawdowns = np.asarray([m["max_drawdown"] for m in metrics_by_asset], dtype=np.float32)
    calmar = np.asarray([m["calmar_ratio"] for m in metrics_by_asset], dtype=np.float32)
    active = np.asarray([m.get("active_return", 0.0) for m in metrics_by_asset], dtype=np.float32)
    return float(np.mean(sharpe) + 2.0 * np.mean(cagr) + 0.25 * np.mean(calmar) + np.mean(drawdowns) + np.mean(active) - 0.5 * np.std(sharpe))


def _mean_metrics(results: list[BacktestResult] | list[DailyBacktestResult]) -> dict[str, float]:
    keys = results[0].metrics.keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [result.metrics[key] for result in results]
        out[key] = float(np.mean(values)) if isinstance(values[0], float) else int(np.sum(values))
    return out


def _daily_methods(prices: np.ndarray, ensemble, hard_router, episode_length: int, cost_bps: float) -> dict[str, DailyBacktestResult]:
    return {
        "soft_router_ensemble": daily_agent_backtest(prices, ensemble, episode_length=episode_length, cost_bps=cost_bps),
        "simulation_hard_router": daily_agent_backtest(prices, hard_router, episode_length=episode_length, cost_bps=cost_bps),
        "fixed_tp_sl": daily_rule_backtest(prices, "fixed_tp_sl", episode_length=episode_length, cost_bps=cost_bps),
        "trailing_stop": daily_rule_backtest(prices, "trailing_stop", episode_length=episode_length, cost_bps=cost_bps),
        "buy_and_hold": daily_buy_and_hold(prices, cost_bps=cost_bps),
    }


def _load_ensemble(model_dir: str, router_path: str, candidate: Candidate, device: str) -> EnsembleAgent:
    cfg = DQNConfig(device=device, lr=candidate.dqn_lr, gamma=candidate.gamma, hidden_dim=candidate.hidden_dim)
    bull = load_agent(str(Path(model_dir) / "bull_agent.pt"), config=cfg, map_location=device)
    bear = load_agent(str(Path(model_dir) / "bear_agent.pt"), config=cfg, map_location=device)
    sideways = load_agent(str(Path(model_dir) / "sideways_agent.pt"), config=cfg, map_location=device)
    router = load_router(router_path, device=device)
    return EnsembleAgent(bull, bear, sideways, router, device=device)


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_optimization(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
    preset: str = "balanced",
    seed: int = 7,
    device: str = "cpu",
    output_dir: str = "outputs/optimization",
    data_dir: str = "data/stooq",
    cost_bps: float = 5.0,
    episode_lengths: list[int] | None = None,
) -> ExperimentSummary:
    _set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloads = download_stooq_daily(tickers or DEFAULT_TICKERS, data_dir, start=start, end=end, min_rows=500)
    series_by_ticker = load_downloaded_close_series(downloads)
    if len(series_by_ticker) < 3:
        raise RuntimeError("Downloaded fewer than three usable assets; cannot run a robust multi-asset experiment.")

    splits_by_ticker = {ticker: split_prices(s.to_numpy(dtype=np.float32)) for ticker, s in series_by_ticker.items()}
    train_sets = [splits.train for splits in splits_by_ticker.values()]
    validation_sets = {ticker: splits.validation for ticker, splits in splits_by_ticker.items()}
    test_sets = {ticker: splits.test for ticker, splits in splits_by_ticker.items()}

    candidate_rows: list[dict] = []
    best: tuple[float, Candidate, str, str] | None = None
    for idx, candidate in enumerate(candidate_grid(preset, device, episode_lengths=episode_lengths)):
        candidate_seed = seed + idx * 1000
        _set_seed(candidate_seed)
        model_dir = str(out / "models" / candidate.name)
        router_path = str(out / "models" / candidate.name / "router.pt")
        dqn_cfg = DQNConfig(device=device, lr=candidate.dqn_lr, gamma=candidate.gamma, hidden_dim=candidate.hidden_dim)
        train_specialists(model_dir, candidate.specialist_episodes, candidate.episode_length, candidate_seed, device, config=dqn_cfg)
        router_cfg = RouterTrainConfig(
            episodes=candidate.router_episodes,
            gamma=candidate.gamma,
            lr=candidate.router_lr,
            device=device,
            episode_length=candidate.episode_length,
        )
        train_router_on_price_sets(train_sets, model_dir, router_path, router_cfg, seed=candidate_seed)
        ensemble = _load_ensemble(model_dir, router_path, candidate, device)
        validation_results = []
        validation_metric_rows = []
        for prices in validation_sets.values():
            result = daily_agent_backtest(prices, ensemble, episode_length=candidate.episode_length, cost_bps=cost_bps)
            benchmark = daily_buy_and_hold(prices, cost_bps=cost_bps)
            metrics = {**result.metrics, **active_metrics(result.equity_curve, benchmark.equity_curve)}
            validation_results.append(result)
            validation_metric_rows.append(metrics)
        score = _score(validation_metric_rows)
        row = {"candidate": candidate.name, "score": score, **asdict(candidate), **_mean_metrics(validation_results)}
        candidate_rows.append(row)
        if best is None or score > best[0]:
            best = (score, candidate, model_dir, router_path)

    if best is None:
        raise RuntimeError("No candidate completed training.")

    best_score, best_candidate, best_model_dir, best_router_path = best
    best_ensemble = _load_ensemble(best_model_dir, best_router_path, best_candidate, device)
    cfg = DQNConfig(device=device, lr=best_candidate.dqn_lr, gamma=best_candidate.gamma, hidden_dim=best_candidate.hidden_dim)
    hard_router = HardRouterAgent(
        load_agent(str(Path(best_model_dir) / "bull_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(best_model_dir) / "bear_agent.pt"), cfg, map_location=device),
        load_agent(str(Path(best_model_dir) / "sideways_agent.pt"), cfg, map_location=device),
    )

    report_rows: list[dict] = []
    aggregate: dict[str, list[DailyBacktestResult]] = {
        "soft_router_ensemble": [],
        "simulation_hard_router": [],
        "fixed_tp_sl": [],
        "trailing_stop": [],
        "buy_and_hold": [],
    }
    for ticker, prices in test_sets.items():
        methods = _daily_methods(prices, best_ensemble, hard_router, best_candidate.episode_length, cost_bps)
        benchmark = methods["buy_and_hold"]
        for method, result in methods.items():
            aggregate[method].append(result)
            active = {} if method == "buy_and_hold" else active_metrics(result.equity_curve, benchmark.equity_curve)
            report_rows.append({"ticker": ticker, "method": method, **result.metrics, **active})

    aggregate_rows = [{"method": method, **_mean_metrics(results)} for method, results in aggregate.items()]
    benchmark_by_asset = aggregate["buy_and_hold"]
    benchmark_lookup = {ticker: result for ticker, result in zip(test_sets, benchmark_by_asset)}
    active_rows = []
    for ticker, prices in test_sets.items():
        del prices
        benchmark = benchmark_lookup[ticker]
        for method in ("soft_router_ensemble", "simulation_hard_router", "fixed_tp_sl", "trailing_stop"):
            result = aggregate[method][list(test_sets).index(ticker)]
            active_rows.append({"ticker": ticker, "method": method, **active_metrics(result.equity_curve, benchmark.equity_curve)})
    active_aggregate = []
    for method in ("soft_router_ensemble", "simulation_hard_router", "fixed_tp_sl", "trailing_stop"):
        rows = [row for row in active_rows if row["method"] == method]
        active_aggregate.append(
            {
                "method": method,
                "active_return": float(np.mean([row["active_return"] for row in rows])),
                "tracking_error": float(np.mean([row["tracking_error"] for row in rows])),
                "information_ratio": float(np.mean([row["information_ratio"] for row in rows])),
            }
        )
    _write_metrics_csv(out / "validation_candidates.csv", candidate_rows)
    _write_metrics_csv(out / "test_metrics_by_asset.csv", report_rows)
    _write_metrics_csv(out / "test_metrics_aggregate.csv", aggregate_rows)
    _write_metrics_csv(out / "active_vs_buy_hold.csv", active_aggregate)
    plot_equity_curves({row["method"]: type("Result", (), {"equity_curve": [1.0, 1.0 + row["total_return"]]})() for row in aggregate_rows}, str(out / "aggregate_equity.png"))

    summary = ExperimentSummary(
        data_rows=sum(item.rows for item in downloads),
        tickers=list(series_by_ticker),
        best_candidate=asdict(best_candidate),
        validation_score=best_score,
        test_metrics={row["method"]: row["total_return"] for row in aggregate_rows},
        output_dir=str(out),
    )
    with (out / "summary.json").open("w") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary
