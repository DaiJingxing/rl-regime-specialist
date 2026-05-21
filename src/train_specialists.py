from __future__ import annotations

from pathlib import Path

from .dqn import DQNAgent, DQNConfig, train_dqn_on_envs
from .env import StockEnv
from .simulators import generate_gbm


REGIMES = {
    "bull": {"mu": 0.001, "sigma": 0.020},
    "bear": {"mu": -0.0005, "sigma": 0.020},
    "sideways": {"mu": 0.000, "sigma": 0.025},
}


def train_specialists(
    model_dir: str = "models",
    episodes: int = 300,
    n_steps: int = 120,
    seed: int = 7,
    device: str = "cpu",
    config: DQNConfig | None = None,
) -> dict[str, str]:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, str] = {}
    for offset, (name, params) in enumerate(REGIMES.items()):
        agent_config = config or DQNConfig(device=device)
        agent_config.device = device
        agent = DQNAgent(state_dim=6, action_dim=2, config=agent_config)

        def env_factory(name=name, params=params, offset=offset):
            path = generate_gbm(
                n_steps=n_steps,
                start_price=100.0,
                mu=params["mu"],
                sigma=params["sigma"],
                seed=seed + offset * 10_000 + agent.steps + len(agent.memory),
            )
            return StockEnv(path)

        train_dqn_on_envs(agent, env_factory, episodes)
        path = str(Path(model_dir) / f"{name}_agent.pt")
        agent.save(path)
        saved_paths[name] = path
    return saved_paths
