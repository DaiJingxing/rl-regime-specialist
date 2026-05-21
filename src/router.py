from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SoftRouter(nn.Module):
    def __init__(self, state_dim: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class EnsembleAgent:
    def __init__(self, bull_agent, bear_agent, sideways_agent, router: SoftRouter, device: str = "cpu") -> None:
        self.specialists = [bull_agent, bear_agent, sideways_agent]
        self.router = router
        self.device = torch.device(device)
        self.router.to(self.device)
        for specialist in self.specialists:
            specialist.q_net.to(self.device)
            specialist.q_net.eval()
            for param in specialist.q_net.parameters():
                param.requires_grad = False

    def get_q_values(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(state):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)

        q_values = torch.stack([agent.q_net(state_t).squeeze(0) for agent in self.specialists], dim=0)
        weights = self.router(state_t).squeeze(0)
        return torch.sum(weights.unsqueeze(1) * q_values, dim=0)

    def weights(self, state: np.ndarray | torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            return self.router(state_t).squeeze(0).cpu().numpy()

    def act_greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            return int(torch.argmax(self.get_q_values(state)).item())

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(0, 2))
        return self.act_greedy(state)

