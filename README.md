# A Comparative Study of DRL Algorithms for Automated Stock Trading

Course project comparing Dueling Double DQN, PPO and Discrete SAC with Buy-and-Hold, SMA Crossover
and fixed-budget DCA on AMD (Technology) and OXY (Energy). The aim is to compare how the algorithms
behave under one trading environment, not to show that reinforcement learning beats other strategies.

This is coursework, not investment advice.

## Setup

- Actions: target stock weight of 0/25/50/75/100%, shared by all agents
- Reward: `100 × log` portfolio growth after costs
- Capital $10,000, fee 0.1%, execution at the next open, no shorting or leverage, fractional shares
- Train 2015–2021, Validation 2022–2023, Test 2024–2025, plus a frozen transfer run on 2026-01-02 to 2026-10-30
- 204,800 environment steps per run, 5 seeds per agent–stock pair
- Hyperparameters chosen on Validation only from a 3×3 grid of learning rate and gamma, 5 tuning seeds

## Test results, 2024–2025 (mean of 5 seeds)

| Method | AMD return | AMD Sharpe | AMD MDD | OXY return | OXY Sharpe | OXY MDD |
|---|---|---|---|---|---|---|
| Buy-and-Hold | +48.3% | 0.55 | −63.0% | −29.1% | −0.57 | −46.9% |
| SMA Crossover | +56.7% | 0.64 | −53.7% | −37.9% | −1.30 | −45.8% |
| DCA | +51.1% | 0.64 | −32.0% | −15.6% | −0.59 | −23.6% |
| Dueling DDQN | +28.9% | 0.33 | −51.0% | −22.7% | −0.76 | −34.9% |
| PPO | +8.9% | 0.20 | −62.3% | −22.8% | −1.12 | −33.4% |
| Discrete SAC | +37.5% | 0.43 | −45.7% | −28.4% | −1.15 | −38.0% |

Differences between agents are smaller than the spread between seeds, so these numbers do not rank
the algorithms. With two stocks and one test window they are case-study results. The 2024–2025 window
was analysed in an earlier version of this project, so it is not a pure holdout. When Sharpe is
negative, lower volatility makes it more negative, so read it together with return and drawdown.
`results/phase7/` holds per-seed results, bootstrap intervals, figures and exploratory diagnostics.

## Reproducing

Python 3.12: `pip install -r requirements.lock.txt`, then `pytest`.

Price data is not included. It comes from Yahoo Finance through `yfinance`:

```
python -c "from trading.data import download_data; download_data()"
python -m tuning.grid --workers 6      # tuning, frozen selection, final training
python tools/phase7_evaluate.py        # one-shot Test and 2026 evaluation
python tools/phase7_report.py          # tables and figures
```

`download_data` covers up to 2025-12-31; the 2026 run also needs the same columns in `data/raw_2026/`.
Yahoo revises adjusted prices over time, so new downloads may not match `data/MANIFEST.sha256`,
which records the files behind these results. Checkpoints are not included.
