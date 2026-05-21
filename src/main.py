from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .backtest import backtest
from .baselines import HardRouterAgent, buy_and_hold, fixed_tp_sl, trailing_stop
from .data_loader import load_price_csv, split_prices
from .dqn import DQNAgent, DQNConfig, load_agent, train_dqn_on_envs
from .env import StockEnv
from .experiments import run_optimization
from .plots import plot_equity_curves, plot_trades
from .regime_pipeline import run_regime_trading_system
from .router import EnsembleAgent
from .simulators import generate_gbm, generate_regime_mix
from .train_router import RouterTrainConfig, load_router, train_router
from .train_specialists import train_specialists


def _load_or_simulate_prices(args) -> np.ndarray:
    if args.use_simulated_real:
        return generate_regime_mix(args.synthetic_steps, seed=args.seed)
    return load_price_csv(args.data, price_col=args.price_col)


def _print_metrics(results: dict[str, object]) -> None:
    names = list(results)
    metric_names = list(next(iter(results.values())).metrics)
    print(",".join(["method", *metric_names]))
    for name in names:
        metrics = results[name].metrics
        values = [name] + [f"{metrics[key]:.6f}" if isinstance(metrics[key], float) else str(metrics[key]) for key in metric_names]
        print(",".join(values))


def _train_single_dqn(prices: np.ndarray, episodes: int, device: str) -> DQNAgent:
    agent = DQNAgent(6, 2, DQNConfig(device=device))

    def env_factory():
        if len(prices) <= 120:
            return StockEnv(prices)
        start = int(np.random.randint(0, len(prices) - 120))
        return StockEnv(prices[start : start + 120])

    train_dqn_on_envs(agent, env_factory, episodes)
    return agent


def _load_ensemble(model_dir: str, router_path: str, device: str) -> EnsembleAgent:
    cfg = DQNConfig(device=device)
    bull = load_agent(str(Path(model_dir) / "bull_agent.pt"), config=cfg, map_location=device)
    bear = load_agent(str(Path(model_dir) / "bear_agent.pt"), config=cfg, map_location=device)
    sideways = load_agent(str(Path(model_dir) / "sideways_agent.pt"), config=cfg, map_location=device)
    router = load_router(router_path, device=device)
    return EnsembleAgent(bull, bear, sideways, router, device=device)


def command_train_specialists(args) -> None:
    saved = train_specialists(args.model_dir, args.episodes, args.n_steps, args.seed, args.device)
    for name, path in saved.items():
        print(f"{name}: {path}")


def command_train_router(args) -> None:
    prices = _load_or_simulate_prices(args)
    splits = split_prices(prices)
    router_path = args.router_path or str(Path(args.model_dir) / "router.pt")
    path = train_router(
        splits.train,
        model_dir=args.model_dir,
        output_path=router_path,
        config=RouterTrainConfig(episodes=args.episodes, device=args.device),
    )
    print(path)


def command_evaluate(args) -> None:
    prices = _load_or_simulate_prices(args)
    splits = split_prices(prices)
    router_path = args.router_path or str(Path(args.model_dir) / "router.pt")
    cfg = DQNConfig(device=args.device)
    bull = load_agent(str(Path(args.model_dir) / "bull_agent.pt"), config=cfg, map_location=args.device)
    bear = load_agent(str(Path(args.model_dir) / "bear_agent.pt"), config=cfg, map_location=args.device)
    sideways = load_agent(str(Path(args.model_dir) / "sideways_agent.pt"), config=cfg, map_location=args.device)
    ensemble = _load_ensemble(args.model_dir, router_path, args.device)
    single_dqn = _train_single_dqn(splits.train, args.single_dqn_episodes, args.device)
    hard_router = HardRouterAgent(bull, bear, sideways)

    results = {
        "fixed_tp_sl": fixed_tp_sl(splits.test),
        "trailing_stop": trailing_stop(splits.test),
        "buy_and_hold": buy_and_hold(splits.test),
        "single_real_dqn": backtest(splits.test, single_dqn),
        "simulation_hard_router": backtest(splits.test, hard_router),
        "soft_router_ensemble": backtest(splits.test, ensemble),
    }
    _print_metrics(results)
    plot_equity_curves(results, str(Path(args.output_dir) / "equity_curves.png"))
    plot_trades(splits.test, results["soft_router_ensemble"].trades, str(Path(args.output_dir) / "soft_router_trades.png"))


def command_run_all(args) -> None:
    command_train_specialists(args)
    command_train_router(args)
    command_evaluate(args)


def command_optimize_backtest(args) -> None:
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()] if args.tickers else None
    episode_lengths = [int(item.strip()) for item in args.episode_lengths.split(",") if item.strip()] if args.episode_lengths else None
    summary = run_optimization(
        tickers=tickers,
        start=args.start,
        end=args.end,
        preset=args.preset,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cost_bps=args.cost_bps,
        episode_lengths=episode_lengths,
    )
    print(f"downloaded_rows={summary.data_rows}")
    print(f"tickers={','.join(summary.tickers)}")
    print(f"best_candidate={summary.best_candidate}")
    print(f"validation_score={summary.validation_score:.6f}")
    for method, total_return in summary.test_metrics.items():
        print(f"{method}_test_total_return={total_return:.6f}")
    print(f"outputs={summary.output_dir}")


def command_regime_system(args) -> None:
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()] if args.tickers else None
    summary = run_regime_trading_system(
        tickers=tickers,
        start=args.start,
        end=args.end,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        cost_bps=args.cost_bps,
        annual_cash_rate=args.annual_cash_rate,
    )
    print(f"downloaded_rows={summary.data_rows}")
    print(f"tickers={','.join(summary.tickers)}")
    print(f"exit_candidate={summary.exit_candidate}")
    print(f"best_entry_config={summary.best_entry_config}")
    print(f"validation_score={summary.validation_score:.6f}")
    for method, total_return in summary.test_total_returns.items():
        print(f"{method}_test_total_return={total_return:.6f}")
    print(f"outputs={summary.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate simulation-pretrained specialist agents with a soft router.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs")

    subparsers = parser.add_subparsers(dest="command", required=True)

    specialists = subparsers.add_parser("train-specialists")
    specialists.add_argument("--episodes", type=int, default=300)
    specialists.add_argument("--n-steps", type=int, default=120)
    specialists.set_defaults(func=command_train_specialists)

    router = subparsers.add_parser("train-router")
    router.add_argument("--data", default="data/real_prices.csv")
    router.add_argument("--price-col", default="Close")
    router.add_argument("--use-simulated-real", action="store_true")
    router.add_argument("--synthetic-steps", type=int, default=1500)
    router.add_argument("--episodes", type=int, default=250)
    router.add_argument("--router-path", default=None)
    router.set_defaults(func=command_train_router)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--data", default="data/real_prices.csv")
    evaluate.add_argument("--price-col", default="Close")
    evaluate.add_argument("--use-simulated-real", action="store_true")
    evaluate.add_argument("--synthetic-steps", type=int, default=1500)
    evaluate.add_argument("--router-path", default=None)
    evaluate.add_argument("--single-dqn-episodes", type=int, default=150)
    evaluate.set_defaults(func=command_evaluate)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--data", default="data/real_prices.csv")
    run_all.add_argument("--price-col", default="Close")
    run_all.add_argument("--use-simulated-real", action="store_true")
    run_all.add_argument("--synthetic-steps", type=int, default=1500)
    run_all.add_argument("--episodes", type=int, default=150)
    run_all.add_argument("--n-steps", type=int, default=120)
    run_all.add_argument("--router-path", default=None)
    run_all.add_argument("--single-dqn-episodes", type=int, default=100)
    run_all.set_defaults(func=command_run_all)

    optimize = subparsers.add_parser("optimize-backtest")
    optimize.add_argument("--tickers", default="")
    optimize.add_argument("--start", default="2005-01-01")
    optimize.add_argument("--end", default=None)
    optimize.add_argument("--preset", choices=["quick", "balanced", "thorough"], default="balanced")
    optimize.add_argument("--data-dir", default="data/stooq")
    optimize.add_argument("--cost-bps", type=float, default=5.0)
    optimize.add_argument("--episode-lengths", default="")
    optimize.set_defaults(func=command_optimize_backtest)

    regime = subparsers.add_parser("regime-system")
    regime.add_argument("--tickers", default="")
    regime.add_argument("--start", default="2005-01-01")
    regime.add_argument("--end", default=None)
    regime.add_argument("--data-dir", default="data/stooq")
    regime.add_argument("--cost-bps", type=float, default=5.0)
    regime.add_argument("--annual-cash-rate", type=float, default=0.0368)
    regime.set_defaults(func=command_regime_system)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
