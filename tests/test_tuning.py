import copy
import numpy as np
import pytest
import torch
from trading.config import TARGET_WEIGHTS
from trading.agents import DuelingDDQN, SACDiscrete
from tuning.learners import TunableDDQN, TunableSAC
from tuning.runner import rank_candidates, Study
from tuning.space import candidates, DEFAULTS, SPACES


def test_predeclared_design_covers_all_levels_and_preserves_default():
    design = candidates()
    assert design == candidates()
    for algorithm, conditions in design.items():
        assert len(conditions) == 8
        assert conditions[0] == {"id": "default", "params": DEFAULTS[algorithm]}
        for key, levels in SPACES[algorithm].items():
            assert {c["params"][key] for c in conditions} == set(levels)
        if algorithm == "PPO":
            for c in conditions:
                p = c["params"]
                assert 24576 % p["n_steps"] == 0
                assert 8192 % p["n_steps"] == 0
                assert p["n_steps"] % p["batch_size"] == 0


def populate(models):
    rng = np.random.default_rng(72)
    for i in range(160):
        item = (rng.normal(size=10), i % len(TARGET_WEIGHTS), float(rng.normal()), rng.normal(size=10), i % 11 == 0)
        for model in models:
            model.replay.add(*item)


def assert_states_equal(first, second):
    a, b = first.state_dict(), second.state_dict()
    for key in a:
        if isinstance(a[key], dict):
            for name in a[key]:
                torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("old,new,name", [
    (DuelingDDQN, TunableDDQN, "DuelingDDQN"), (SACDiscrete, TunableSAC, "SAC-Discrete")])
def test_default_numerical_parity_with_v6_before_and_after_update(old, new, name):
    original = old(42)
    tunable = new(42, DEFAULTS[name])
    assert_states_equal(original, tunable)
    populate([original, tunable])
    assert original.update() == tunable.update()
    assert_states_equal(original, tunable)
    obs = np.linspace(-1, 1, 10).astype(np.float32)
    assert original.predict(obs)[0] == tunable.predict(obs)[0]


@pytest.mark.parametrize("model_cls,name", [(TunableDDQN, "DuelingDDQN"), (TunableSAC, "SAC-Discrete")])
def test_nondefault_width_rate_batch_gamma_are_actually_applied(model_cls, name):
    params = copy.deepcopy(DEFAULTS[name])
    params.update(hidden_size=128, learning_rate=1e-4, batch_size=128, gamma=.95)
    model = model_cls(2, params)
    populate([model])
    called = []
    original_sample = model.replay.sample
    def sample(size):
        called.append(size)
        return original_sample(size)
    model.replay.sample = sample
    assert np.isfinite(model.update())
    assert called == [128]
    assert model.gamma == .95
    if name == "DuelingDDQN":
        assert model.online.shared[0].out_features == 128
        assert model.optimizer.param_groups[0]["lr"] == 1e-4
    else:
        assert model.actor[0].out_features == 128
        assert model.actor_optimizer.param_groups[0]["lr"] == 1e-4


def row(candidate, ticker, seed, sharpe, test_return=0):
    return {"candidate": candidate, "ticker": ticker, "seed": seed, "params": {},
            "validation": {"sharpe_ratio": sharpe, "cumulative_return": .1, "max_drawdown": -.2},
            "test": {"cumulative_return": test_return}}


def test_selection_uses_validation_and_equal_stock_seed_cells_not_test():
    rows = [row(c, t, s, score, future) for c, score, future in
            [("a", .7, -100), ("b", .3, 100)] for t in ("AMD", "OXY") for s in (101, 202)]
    ranking = rank_candidates(rows, 4)
    assert ranking[0]["candidate"] == "a"
    assert ranking[0]["validation_sharpe_mean"] == .7


def test_missing_undefined_or_duplicate_cells_cannot_win():
    rows = [row("a", t, s, .5) for t in ("AMD", "OXY") for s in (101, 202)]
    rows += [row("undefined", t, s, None) for t in ("AMD", "OXY") for s in (101, 202)]
    rows += [row("duplicates", "AMD", 101, 100) for _ in range(4)]
    rows += [row("missing", "AMD", 101, 100)]
    ranking = rank_candidates(rows, 4)
    assert ranking[0]["candidate"] == "a"
    assert sum(r["eligible"] for r in ranking) == 1


def test_test_evaluation_is_blocked_until_selection_is_frozen(tmp_path):
    study = Study.__new__(Study)
    study.directory = tmp_path
    with pytest.raises(RuntimeError, match="selection is frozen"):
        study.env("AMD", "test")
