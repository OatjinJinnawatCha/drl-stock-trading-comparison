"""Configurable versions with default numerical parity against frozen v6 agents."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from trading.agents import (DuelingDDQN, SACDiscrete, ReplayBuffer, set_seed,
                            double_dqn_target, DuelingNetwork, he_init)
from trading.config import TARGET_WEIGHTS, WARMUP_STEPS


def mlp(inputs, outputs, width):
    return he_init(nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width),
                                 nn.ReLU(), nn.Linear(width, outputs)))


class VariableDuelingNetwork(DuelingNetwork):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.shared = nn.Sequential(nn.Linear(10, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, len(TARGET_WEIGHTS))
        he_init(self)


class TunableDDQN(DuelingDDQN):
    def __init__(self, seed, params):
        set_seed(seed)
        self.params = params.copy()
        self.rng = np.random.default_rng(seed)
        self.online = VariableDuelingNetwork(params["hidden_size"])
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=params["learning_rate"])
        self.replay = ReplayBuffer(seed=seed)
        self.gamma = params["gamma"]

    def sample_action(self, obs, step, total):
        p = self.params
        epsilon = 1 - (1 - p["epsilon_final"]) * min(step / (total * p["exploration_fraction"]), 1)
        return int(self.rng.integers(len(TARGET_WEIGHTS))) if self.rng.random() < epsilon else self.predict(obs)[0]

    def update(self):
        states, actions, rewards, next_states, dones = self.replay.sample(self.params["batch_size"])
        target = double_dqn_target(self.online, self.target, next_states, rewards, dones, self.gamma)
        prediction = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self, step):
        if step % self.params["target_interval"] == 0:
            self.target.load_state_dict(self.online.state_dict())


class TunableSAC(SACDiscrete):
    def __init__(self, seed, params):
        set_seed(seed)
        self.params = params.copy()
        self.rng = np.random.default_rng(seed)
        width, rate = params["hidden_size"], params["learning_rate"]
        self.actor = mlp(10, len(TARGET_WEIGHTS), width)
        self.q1, self.q2 = mlp(10, len(TARGET_WEIGHTS), width), mlp(10, len(TARGET_WEIGHTS), width)
        self.target1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rate)
        self.critic_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=rate)
        self.log_alpha = torch.tensor(np.log(params["initial_alpha"]), dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=rate)
        self.target_entropy = params["entropy_fraction"] * np.log(len(TARGET_WEIGHTS))
        self.gamma, self.tau = params["gamma"], params["tau"]
        self.replay = ReplayBuffer(seed=seed)

    def update(self):
        states, actions, rewards, next_states, dones = self.replay.sample(self.params["batch_size"])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            probabilities, log_probabilities = self.distribution(next_states)
            q_min = torch.minimum(self.target1(next_states), self.target2(next_states))
            soft_value = (probabilities * (q_min - alpha * log_probabilities)).sum(1)
            target = rewards + self.gamma * (1 - dones) * soft_value
        q1 = self.q1(states).gather(1, actions[:, None]).squeeze(1)
        q2 = self.q2(states).gather(1, actions[:, None]).squeeze(1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_([*self.q1.parameters(), *self.q2.parameters()], 10)
        self.critic_optimizer.step()
        probabilities, log_probabilities = self.distribution(states)
        with torch.no_grad():
            q_min = torch.minimum(self.q1(states), self.q2(states))
        actor_loss = (probabilities * (alpha * log_probabilities - q_min)).sum(1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        entropy = -(probabilities * log_probabilities).sum(1).detach()
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10, 2)
        return float(critic_loss.item())


def create_model(algorithm, seed, params, env):
    if algorithm == "DuelingDDQN":
        return TunableDDQN(seed, params)
    if algorithm == "SAC-Discrete":
        return TunableSAC(seed, params)
    from stable_baselines3 import PPO
    set_seed(seed)
    p = params.copy()
    width = p.pop("hidden_size")
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
               policy_kwargs={"net_arch": {"pi": [width, width], "vf": [width, width]},
                              "activation_fn": nn.ReLU}, **p)
