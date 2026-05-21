from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_dim: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_size: int = 10_000
    target_update: int = 25
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.asarray, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig | None = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim=self.config.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim=self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.memory = ReplayBuffer(self.config.replay_size)
        self.epsilon = self.config.epsilon_start
        self.steps = 0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.q_values(state)
        return int(torch.argmax(q_values).item())

    def q_values(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.q_net(state).squeeze(0)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.config.hidden_dim,
                "q_net": self.q_net.state_dict(),
            },
            path,
        )


def load_agent(path: str, config: DQNConfig | None = None, map_location: str = "cpu") -> DQNAgent:
    checkpoint = torch.load(path, map_location=map_location)
    cfg = config or DQNConfig(device=map_location)
    if "hidden_dim" in checkpoint and config is None:
        cfg.hidden_dim = int(checkpoint["hidden_dim"])
    agent = DQNAgent(checkpoint["state_dim"], checkpoint["action_dim"], config=cfg)
    agent.q_net.load_state_dict(checkpoint["q_net"])
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    agent.epsilon = 0.0
    agent.q_net.eval()
    return agent


def train_dqn_on_envs(agent: DQNAgent, env_factory, episodes: int) -> list[float]:
    episode_rewards: list[float] = []
    for _ in range(episodes):
        env = env_factory()
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
    return episode_rewards
