# Simulation-Pretrained Specialist Agents + Soft Router

This project uses a two-stage training framework.
First, three DQN specialist agents are pretrained in simulated bull, bear, and sideways market regimes.
Second, instead of manually labeling real market regimes, a soft router is trained on real historical data to dynamically combine the Q-values of the three specialists.
The final trading decision is made by a weighted ensemble of specialist policies.

本项目采用两阶段训练框架。
第一阶段，在模拟牛市、熊市和震荡市中分别预训练三个 DQN 专才 Agent。
第二阶段，不对真实市场进行硬标签划分，而是在真实历史数据上训练一个 Soft Router，让它动态分配三个专家的权重。
最终卖出决策由三个专家 Q-value 的加权融合得到。

## Structure

```text
data/
  real_prices.csv
models/
outputs/
src/
  data_loader.py
  simulators.py
  env.py
  dqn.py
  train_specialists.py
  router.py
  train_router.py
  backtest.py
  baselines.py
  metrics.py
  plots.py
  main.py
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train specialists on simulation:

```bash
python -m src.main train-specialists
```

Train the soft router on real data:

```bash
python -m src.main train-router --data data/real_prices.csv
```

Evaluate all major methods:

```bash
python -m src.main evaluate --data data/real_prices.csv
```

If no real CSV is available, add `--use-simulated-real` to run the full pipeline with a synthetic price series:

```bash
python -m src.main run-all --use-simulated-real
```

The real-data CSV should include a price column such as `Close` and optionally a date column such as `Date`.

