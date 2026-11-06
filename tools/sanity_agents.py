"""Phase 4 sanity checks for the three agents (5 actions, 128 units, 10 inputs).

  cartpole : CartPole-v1 padded to 10 inputs; actions 0-4 map to left/right via a % 2.
             Pass = greedy mean return >= 475 over 10 episodes.
  toy      : real TradingEnv on a price that rises every day. Correct policy is to hold 100%.
             Pass = mean position ratio >= 0.9 and return >= 90% of Buy-and-Hold on the same env.

usage: python tools/sanity_agents.py {cartpole|toy} {DuelingDDQN|PPO|SAC-Discrete} STEPS [SEED] [EVAL_EVERY]
"""
import json
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading.agents import DuelingDDQN, SACDiscrete, set_seed
from trading.config import TARGET_WEIGHTS, HIDDEN, WARMUP_STEPS
from trading.environment import TradingEnv
from trading.evaluation import evaluate, metrics


class PaddedCartPole(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (10,), np.float32)
        self.action_space = gym.spaces.Discrete(len(TARGET_WEIGHTS))

    def _pad(self, obs):
        return np.r_[obs, np.zeros(6)].astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action) % 2)
        return self._pad(obs), reward, terminated, truncated, info


def toy_env(days=300, drift=0.004):
    index = pd.bdate_range("2023-01-02", periods=days)
    price = 100 * np.cumprod(np.full(days, 1 + drift))
    frame = pd.DataFrame({"Open": price, "Close": price, "Volume": 1000.0,
                          "risk_free_annual": 0.0, "sma10": 1.0, "sma30": 1.0}, index=index)
    return TradingEnv(frame, np.zeros((days, 8), np.float32), (index[1], index[-1]))


def make_env(task):
    return PaddedCartPole() if task == "cartpole" else toy_env()


def train(task, name, steps, seed, eval_every=0, trace=None):
    set_seed(seed)
    env = make_env(task)
    if name == "PPO":
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, learning_rate=3e-4,
                    n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01,
                    policy_kwargs={"net_arch": {"pi": [HIDDEN, HIDDEN], "vf": [HIDDEN, HIDDEN]},
                                   "activation_fn": torch.nn.ReLU})
        model.learn(total_timesteps=steps)
        return model
    model = DuelingDDQN(seed) if name == "DuelingDDQN" else SACDiscrete(seed)
    obs, _ = env.reset(seed=seed)
    for step in range(1, steps + 1):
        action = model.sample_action(obs, step, steps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Time-limit truncation is not a true terminal state: keep bootstrapping.
        model.replay.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if step >= WARMUP_STEPS and step % 4 == 0:
            model.update()
            model.update_target(step)
        if terminated or truncated:
            obs, _ = env.reset()
        if eval_every and step % eval_every == 0 and trace is not None and task == "cartpole":
            trace.append((step, check_cartpole(model, episodes=5)["mean_return"]))
    return model


def check_cartpole(model, episodes=10):
    env = PaddedCartPole()
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(model.predict(obs, deterministic=True)[0])
            total += reward
            done = terminated or truncated
        returns.append(total)
    return {"mean_return": float(np.mean(returns)), "returns": returns,
            "passed": bool(np.mean(returns) >= 475)}


def check_toy(model):
    history = evaluate(toy_env(), model=model)
    benchmark = evaluate(toy_env(), strategy="Buy-and-Hold")
    result, base = metrics(history), metrics(benchmark)
    return {"cumulative_return": result["cumulative_return"],
            "buy_hold_return": base["cumulative_return"],
            "mean_position_ratio": result["mean_position_ratio"],
            "actions_used": sorted(set(history["action"].tolist())),
            "passed": bool(result["mean_position_ratio"] >= 0.9
                           and result["cumulative_return"] >= 0.9 * base["cumulative_return"])}


if __name__ == "__main__":
    task, name, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    eval_every = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    trace = []
    model = train(task, name, steps, seed, eval_every, trace)
    result = {"trace": trace, "task": task, "algorithm": name, "steps": steps, "seed": seed,
              **(check_cartpole(model) if task == "cartpole" else check_toy(model))}
    print(json.dumps(result))
