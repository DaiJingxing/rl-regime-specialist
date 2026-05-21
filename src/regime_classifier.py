from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REGIME_NAMES = ["bull", "bear", "sideways"]


class RegimeClassifier(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def probabilities(self, x: np.ndarray | torch.Tensor, device: str = "cpu") -> np.ndarray:
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x.to(device))
            return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


@dataclass(frozen=True)
class RegimeTrainConfig:
    horizon: int = 60
    bull_return: float = 0.06
    bear_return: float = -0.06
    hidden_dim: int = 64
    epochs: int = 35
    lr: float = 1e-3
    batch_size: int = 256
    device: str = "cpu"


def _safe_return(prices: np.ndarray, t: int, lookback: int) -> float:
    if t - lookback < 0 or prices[t - lookback] <= 0:
        return 0.0
    return float(prices[t] / prices[t - lookback] - 1.0)


def market_features(prices: np.ndarray, t: int) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float32)
    if t < 200:
        raise ValueError("Need at least 200 lookback points for regime features.")
    price = float(prices[t])
    returns = np.diff(prices[: t + 1]) / np.maximum(prices[:t], 1e-12)
    vol20 = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
    vol60 = float(np.std(returns[-60:])) if len(returns) >= 60 else vol20
    ma20 = float(np.mean(prices[t - 19 : t + 1]))
    ma60 = float(np.mean(prices[t - 59 : t + 1]))
    ma200 = float(np.mean(prices[t - 199 : t + 1]))
    high20 = float(np.max(prices[t - 19 : t + 1]))
    high60 = float(np.max(prices[t - 59 : t + 1]))
    low20 = float(np.min(prices[t - 19 : t + 1]))
    rsi_window = returns[-14:] if len(returns) >= 14 else returns
    gains = rsi_window[rsi_window > 0].mean() if np.any(rsi_window > 0) else 0.0
    losses = abs(rsi_window[rsi_window < 0].mean()) if np.any(rsi_window < 0) else 1e-6
    rsi = 100.0 - 100.0 / (1.0 + gains / max(losses, 1e-6))
    return np.asarray(
        [
            _safe_return(prices, t, 1),
            _safe_return(prices, t, 5),
            _safe_return(prices, t, 20),
            _safe_return(prices, t, 60),
            price / ma20 - 1.0,
            price / ma60 - 1.0,
            price / ma200 - 1.0,
            vol20,
            vol60,
            price / high20 - 1.0,
            price / high60 - 1.0,
            (price - low20) / max(high20 - low20, 1e-6),
        ],
        dtype=np.float32,
    )


def make_regime_dataset(
    price_sets: Sequence[np.ndarray],
    config: RegimeTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for prices in price_sets:
        prices = np.asarray(prices, dtype=np.float32)
        if len(prices) <= 200 + config.horizon:
            continue
        for t in range(200, len(prices) - config.horizon):
            future_return = float(prices[t + config.horizon] / prices[t] - 1.0)
            if future_return >= config.bull_return:
                label = 0
            elif future_return <= config.bear_return:
                label = 1
            else:
                label = 2
            features.append(market_features(prices, t))
            labels.append(label)
    if not features:
        raise ValueError("No regime classifier samples could be generated.")
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def train_regime_classifier(
    train_price_sets: Sequence[np.ndarray],
    output_path: str,
    config: RegimeTrainConfig | None = None,
) -> str:
    cfg = config or RegimeTrainConfig()
    x, y = make_regime_dataset(train_price_sets, cfg)
    device = torch.device(cfg.device)
    model = RegimeClassifier(input_dim=x.shape[1], hidden_dim=cfg.hidden_dim).to(device)
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_dim": x.shape[1], "hidden_dim": cfg.hidden_dim, "state_dict": model.state_dict(), "config": cfg.__dict__}, output_path)
    return output_path


def load_regime_classifier(path: str, device: str = "cpu") -> RegimeClassifier:
    checkpoint = torch.load(path, map_location=device)
    model = RegimeClassifier(input_dim=int(checkpoint["input_dim"]), hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
