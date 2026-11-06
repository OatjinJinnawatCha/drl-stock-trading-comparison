"""Predeclared search design. Ranges are experimental choices, not stock optima."""
import copy
import random

SEARCH_SEED = 631
SCREEN_SEEDS = (101, 202)
CONFIRM_SEEDS = (303, 404)
FINAL_SEEDS = (901, 902, 903, 904, 905)
SCREEN_STEPS = 24_576
FULL_STEPS = 204_800
DEFAULTS = {
    "DuelingDDQN": dict(learning_rate=3e-4, gamma=.99, batch_size=256, hidden_size=128,
                        target_interval=1000, exploration_fraction=.25, epsilon_final=.05),
    "PPO": dict(learning_rate=3e-4, gamma=.99, batch_size=256, hidden_size=128,
                n_steps=2048, n_epochs=10, clip_range=.2, ent_coef=.01, gae_lambda=.95),
    "SAC-Discrete": dict(learning_rate=3e-4, gamma=.99, batch_size=256, hidden_size=128,
                         tau=.005, initial_alpha=.1, entropy_fraction=.7),
}
COMMON = dict(learning_rate=[1e-4, 3e-4, 1e-3], gamma=[.95, .99, .995],
              batch_size=[128, 256], hidden_size=[64, 128])
SPACES = {
    "DuelingDDQN": {**COMMON, "target_interval": [500, 1000, 2000],
                    "exploration_fraction": [.15, .25, .5], "epsilon_final": [.02, .05, .1]},
    "PPO": {**COMMON, "n_steps": [512, 1024, 2048], "n_epochs": [5, 10],
            "clip_range": [.1, .2, .3], "ent_coef": [0., .01, .03], "gae_lambda": [.9, .95, .98]},
    "SAC-Discrete": {**COMMON, "tau": [.005, .01, .02],
                     "initial_alpha": [.01, .1, .3], "entropy_fraction": [.3, .7, .95]},
}


def candidates(count=8):
    """Default + seeded categorical random design with balanced marginals.

    Every listed value appears in the seven sampled conditions. Joint combinations
    are not exhaustive and this design cannot isolate causal parameter effects.
    """
    rng = random.Random(SEARCH_SEED)
    result = {}
    for algorithm, space in SPACES.items():
        conditions = [{"id": "default", "params": copy.deepcopy(DEFAULTS[algorithm])}]
        columns = {}
        for parameter, values in space.items():
            samples = (values * (1 + (count - 2) // len(values)))[:count - 1]
            rng.shuffle(samples)
            columns[parameter] = samples
        for i in range(count - 1):
            params = {key: values[i] for key, values in columns.items()}
            conditions.append({"id": f"trial_{i + 1:02d}", "params": params})
        assert len({str(c["params"]) for c in conditions}) == count
        result[algorithm] = conditions
    return result
