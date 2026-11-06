import numpy as np
import pandas as pd
from .config import INITIAL_CAPITAL, TARGET_WEIGHTS


def evaluate(env, model=None, strategy=None):
    observation, _ = env.reset()
    records = []
    # Fixed total budget: 20 equal contributions from the ORIGINAL cash balance.
    # One installment every `interval` sessions (25 on the 502-day Test, C4); never inject capital.
    interval = (env.end - env.start + 1) // 20
    dca_ticks = {env.start + i * interval for i in range(20)}
    allocation = INITIAL_CAPITAL / 20
    while not env.finished:
        if model is not None:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, _, info = env.step(action)
        else:
            if strategy == "DCA":
                budget = min(allocation, env.cash) if env.tick in dca_ticks else 0.0
                observation, _, _, _, info = env.execute(
                    buy_budget=budget, action=2 if budget > 0 else 1)
            else:
                if strategy == "Buy-and-Hold":
                    target = 1.0
                elif strategy == "SMA-Crossover":
                    previous = env.frame.iloc[env.tick - 1]
                    target = 1.0 if previous["sma10"] > previous["sma30"] else 0.0
                else:
                    raise ValueError(f"Unknown strategy {strategy}")
                observation, _, _, _, info = env.execute(
                    target_weight=target, action=TARGET_WEIGHTS.index(target))
        records.append(info)
    return pd.DataFrame(records)


def metrics(history):
    values = np.r_[INITIAL_CAPITAL, history["value"].to_numpy()]
    returns = values[1:] / values[:-1] - 1
    # Previous day's Treasury rate is already attached by the causal ledger.
    rf = np.power(1 + history["risk_free_annual"].to_numpy(), 1 / 252) - 1
    excess = returns - rf
    deviation = excess.std(ddof=1)
    # A flat cash-only portfolio has no return variability. Do not fabricate a
    # huge finite Sharpe from tiny changes in the Treasury proxy itself.
    sharpe = (float(np.sqrt(252) * excess.mean() / deviation)
              if deviation > 1e-10 and returns.std(ddof=1) > 1e-10 else None)
    drawdown = values / np.maximum.accumulate(values) - 1
    return {"cumulative_return": values[-1] / INITIAL_CAPITAL - 1,
            "sharpe_ratio": sharpe, "max_drawdown": float(drawdown.min()),
            "number_of_trades": int(history["traded"].sum()),
            "turnover": float(history["notional"].sum() / history["value"].mean()),
            "total_fees": float(history["fee"].sum()), "final_value": float(values[-1]),
            "mean_position_ratio": float(history["position_ratio"].mean()),
            "exposure": float((history["shares"] > 0).mean()),
            "days": len(history), "start": str(history.iloc[0]["date"]),
            "end": str(history.iloc[-1]["date"])}
