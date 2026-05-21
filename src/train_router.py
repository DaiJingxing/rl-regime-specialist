from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dqn import ReplayBuffer, load_agent
from .env import StockEnv
from .router import EnsembleAgent, SoftRouter


@dataclass
class RouterTrainConfig:
    episodes: int = 250
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_size: int = 10_000
    device: str = "cpu"
    episode_length: int = 160


def _router_q_for_actions(ensemble: EnsembleAgent, states: torch.Tensor) -> torch.Tensor:
    q_batches = []
    for state in states:
        q_batches.append(ensemble.get_q_values(state))
    return torch.stack(q_batches, dim=0)


def train_router(
    train_prices: np.ndarray,
    model_dir: str = "models",
    output_path: str = "models/router.pt",
    config: RouterTrainConfig | None = None,
) -> str:
    cfg = config or RouterTrainConfig()
    device = torch.device(cfg.device)
    bull = load_agent(str(Path(model_dir) / "bull_agent.pt"), map_location=cfg.device)
    bear = load_agent(str(Path(model_dir) / "bear_agent.pt"), map_location=cfg.device)
    sideways = load_agent(str(Path(model_dir) / "sideways_agent.pt"), map_location=cfg.device)
    router = SoftRouter(state_dim=6).to(device)
    ensemble = EnsembleAgent(bull, bear, sideways, router, device=cfg.device)
    optimizer = torch.optim.Adam(router.parameters(), lr=cfg.lr)
    memory = ReplayBuffer(cfg.replay_size)
    epsilon = cfg.epsilon_start

    for _ in range(cfg.episodes):
        env = StockEnv(train_prices)
        state = env.reset()
        done = False
        while not done:
            action = ensemble.act(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state

            if len(memory) >= cfg.batch_size:
                states, actions, rewards, next_states, dones = memory.sample(cfg.batch_size)
                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                q = _router_q_for_actions(ensemble, states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = _router_q_for_actions(ensemble, next_states_t).max(dim=1, keepdim=True).values
                    target = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q

                loss = F.mse_loss(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dim": 6, "router": router.state_dict()}, output_path)
    return output_path


def train_router_on_price_sets(
    train_price_sets: Sequence[np.ndarray],
    model_dir: str = "models",
    output_path: str = "models/router.pt",
    config: RouterTrainConfig | None = None,
    seed: int = 7,
) -> str:
    """Train one router by sampling full episodes from multiple chronological assets."""
    cfg = config or RouterTrainConfig()
    valid_sets = [np.asarray(prices, dtype=np.float32) for prices in train_price_sets if len(prices) >= 30]
    if not valid_sets:
        raise ValueError("Need at least one price set with 30 or more rows to train the router.")

    rng = random.Random(seed)
    device = torch.device(cfg.device)
    bull = load_agent(str(Path(model_dir) / "bull_agent.pt"), map_location=cfg.device)
    bear = load_agent(str(Path(model_dir) / "bear_agent.pt"), map_location=cfg.device)
    sideways = load_agent(str(Path(model_dir) / "sideways_agent.pt"), map_location=cfg.device)
    router = SoftRouter(state_dim=6).to(device)
    ensemble = EnsembleAgent(bull, bear, sideways, router, device=cfg.device)
    optimizer = torch.optim.Adam(router.parameters(), lr=cfg.lr)
    memory = ReplayBuffer(cfg.replay_size)
    epsilon = cfg.epsilon_start

    for _ in range(cfg.episodes):
        prices = rng.choice(valid_sets)
        if len(prices) > cfg.episode_length:
            start = rng.randrange(0, len(prices) - cfg.episode_length)
            prices = prices[start : start + cfg.episode_length]
        env = StockEnv(prices)
        state = env.reset()
        done = False
        while not done:
            action = ensemble.act(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state

            if len(memory) >= cfg.batch_size:
                states, actions, rewards, next_states, dones = memory.sample(cfg.batch_size)
                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                q = _router_q_for_actions(ensemble, states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = _router_q_for_actions(ensemble, next_states_t).max(dim=1, keepdim=True).values
                    target = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q

                loss = F.mse_loss(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dim": 6, "router": router.state_dict()}, output_path)
    return output_path


def load_router(path: str, device: str = "cpu") -> SoftRouter:
    checkpoint = torch.load(path, map_location=device)
    router = SoftRouter(state_dim=checkpoint.get("state_dim", 6))
    router.load_state_dict(checkpoint["router"])
    router.eval()
    return router
