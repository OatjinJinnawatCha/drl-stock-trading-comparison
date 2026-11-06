"""Same cash/share ledger for agents and baselines, with causal execution."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .config import INITIAL_CAPITAL, TRANSACTION_COST, SPLITS, TARGET_WEIGHTS, REBALANCE_BAND


class TradingEnv(gym.Env):
    """Observe close(t-1), execute open(t), value at close(t).

    Action i sets the target stock weight TARGET_WEIGHTS[i] (0/25/50/75/100%); moves smaller
    than REBALANCE_BAND are skipped unless the target is 0 or 100%.
    Fractional shares, no borrowing/shorting; cash earns zero. Adjusted OHLC is
    used as a total-return proxy, not a historical executable unadjusted ledger.
    """
    metadata = {"render_modes": []}

    def __init__(self, frame, scaled_features, split="train", initial_capital=INITIAL_CAPITAL,
                 transaction_cost=TRANSACTION_COST):
        super().__init__()
        self.frame = frame
        self.features = np.asarray(scaled_features, dtype=np.float32)
        self.opens = frame["Open"].to_numpy(dtype=float)
        self.closes = frame["Close"].to_numpy(dtype=float)
        self.dates = frame.index
        self.rf = frame["risk_free_annual"].to_numpy(dtype=float)
        start, end = SPLITS[split] if isinstance(split, str) else split
        positions = np.flatnonzero((frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end)))
        if len(positions) < 2 or positions[0] < 1:
            raise ValueError("Need split data and at least one preceding observation")
        self.start, self.end = int(positions[0]), int(positions[-1])
        self.initial_capital = float(initial_capital)
        self.cost = float(transaction_cost)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(TARGET_WEIGHTS))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = self.start
        self.cash, self.shares = self.initial_capital, 0.0
        self.value = self.initial_capital
        self.total_fees, self.total_notional, self.trades = 0.0, 0.0, 0
        self.finished = False
        return self._observation(), {}

    def _observation(self):
        stock_value = self.shares * self.closes[self.tick - 1]
        value = self.cash + stock_value
        return np.concatenate([self.features[self.tick - 1],
                               [self.cash / value, stock_value / value]]).astype(np.float32)

    def step(self, action):
        action = int(np.asarray(action).item())
        if not 0 <= action < len(TARGET_WEIGHTS):
            raise ValueError(f"action must be an index into TARGET_WEIGHTS {TARGET_WEIGHTS}")
        return self.execute(target_weight=TARGET_WEIGHTS[action], action=action)

    def _rebalance(self, price, target):
        """Move to `target` share of portfolio value at the open; fee = cost * gross trade value.

        Buy: spend = (wV - H) / (1 - c + wc); sell: gross = (H - wV) / (1 - wc).
        Both leave the post-trade weight exactly at `target`.
        """
        value = self.cash + self.shares * price
        holding = self.shares * price
        current = holding / value
        if abs(target - current) < 1e-9 or (0 < target < 1 and abs(target - current) < REBALANCE_BAND):
            return 0.0, 0.0
        c = self.cost
        if target > current:
            spend = min((target * value - holding) / (1 - c + target * c), self.cash)
            self.cash -= spend
            self.shares += spend * (1 - c) / price
            return spend, spend * c
        gross = min((holding - target * value) / (1 - target * c), holding)
        self.shares = self.shares - gross / price if target > 0 else 0.0
        self.cash += gross * (1 - c)
        return gross, gross * c

    def execute(self, *, sell_fraction=0.0, buy_budget=0.0, target_weight=None, action=1):
        """Shared execution. Agents and B&H/SMA use `target_weight`; DCA uses `buy_budget`
        (cash spent, fee included). Fee is `cost` times the gross trade value in every path."""
        if self.finished:
            raise RuntimeError("Episode finished; call reset")
        old_value = self.value
        price = self.opens[self.tick]
        notional, fee = 0.0, 0.0
        if target_weight is not None:
            notional, fee = self._rebalance(price, float(target_weight))
        elif sell_fraction > 0 and self.shares > 0:
            quantity = self.shares * min(float(sell_fraction), 1.0)
            notional = quantity * price
            fee = notional * self.cost
            self.shares -= quantity
            self.cash += notional - fee
        elif buy_budget > 0 and self.cash > 0:
            notional = min(float(buy_budget), self.cash)
            fee = notional * self.cost
            self.shares += (notional - fee) / price
            self.cash -= notional
        traded = notional > 1e-6
        self.trades += int(traded)
        self.total_fees += fee
        self.total_notional += notional
        self.value = self.cash + self.shares * self.closes[self.tick]
        daily_return = self.value / old_value - 1
        info = {"date": str(self.dates[self.tick].date()), "value": self.value,
                "daily_return": daily_return, "cash": self.cash, "shares": self.shares,
                "position_ratio": self.shares * self.closes[self.tick] / self.value,
                "notional": notional, "fee": fee, "traded": traded, "action": int(action),
                "risk_free_annual": self.rf[self.tick - 1]}
        # Exact log portfolio growth, including overnight gaps and transaction fees.
        reward = float(100 * np.log(self.value / old_value))
        self.tick += 1
        self.finished = self.tick > self.end
        obs = np.zeros(10, dtype=np.float32) if self.finished else self._observation()
        return obs, reward, self.finished, False, info
