import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from stable_baselines3.common.env_checker import check_env
from trading.environment import TradingEnv
from trading.evaluation import evaluate, metrics
from trading.data import make_features, TrainScaler
from trading.agents import DuelingNetwork, double_dqn_target, SACDiscrete
from trading.config import MARKET_FEATURES


def fixture_frame(n=30):
    index = pd.bdate_range("2023-12-29", periods=n)
    frame = pd.DataFrame({"Open": 100., "Close": 100., "Volume": 1000.,
                          "risk_free_annual": 0.04, "sma10": 101., "sma30": 100.}, index=index)
    return frame


def make_env(frame=None, **kwargs):
    frame = fixture_frame() if frame is None else frame
    return TradingEnv(frame, np.zeros((len(frame), 8), np.float32), "test", **kwargs)


def test_buy_to_half_by_hand_calculation():
    # Hand calc, c=0.001, V=10000, Open=100, target 0.5:
    # spend = 5000 / (1 - 0.001 + 0.5*0.001) = 5000 / 0.9995 = 5002.50125
    # shares = spend * 0.999 / 100 = 49.974987; cash = 4997.49875; fee = spend * 0.001
    frame = fixture_frame()
    frame.iloc[1, frame.columns.get_loc("Close")] = 110.
    env = make_env(frame)
    env.reset()
    obs, reward, _, _, info = env.step(2)
    spend = 5000 / 0.9995
    assert info["notional"] == pytest.approx(spend)
    assert info["fee"] == pytest.approx(spend * 0.001)
    assert env.shares == pytest.approx(spend * 0.999 / 100)
    assert env.cash == pytest.approx(10000 - spend) and env.cash >= 0
    # Weight right after the trade, valued at the open, equals the target exactly.
    assert env.shares * 100 / (env.cash + env.shares * 100) == pytest.approx(0.5, abs=1e-12)
    assert env.value == pytest.approx(env.cash + env.shares * 110)
    assert reward == pytest.approx(100 * np.log(env.value / 10000))
    assert obs[-2:].sum() == pytest.approx(1)


@pytest.mark.parametrize("start_shares,target", [
    (0, 0.25), (0, 0.5), (0, 0.75), (0, 1.0),     # entries from cash
    (25, 0.5), (50, 0.75), (50, 1.0), (60, 1.0),  # step ups
    (100, 0.75), (75, 0.25), (60, 0.0), (100, 0.0),  # step downs, exit
])
def test_post_trade_weight_equals_target_and_cash_not_negative(start_shares, target):
    env = make_env()
    env.reset()
    env.shares, env.cash = float(start_shares), 10000. - 100. * start_shares
    _, _, _, _, info = env.execute(target_weight=target)
    value = env.cash + env.shares * 100
    assert env.shares * 100 / value == pytest.approx(target, abs=1e-12)
    assert env.cash >= -1e-9 and env.shares >= 0
    assert info["fee"] == pytest.approx(0.001 * info["notional"])
    assert value == pytest.approx(10000 - info["fee"])


def test_sell_by_hand_calculation():
    # Start 100 shares (all stock, V=10000), target 0.5: gross = 5000 / (1 - 0.0005) = 5002.50125
    env = make_env()
    env.reset()
    env.shares, env.cash = 100., 0.
    env.execute(target_weight=0.5)
    gross = 5000 / 0.9995
    assert env.shares == pytest.approx(100 - gross / 100)
    assert env.cash == pytest.approx(gross * 0.999)


@pytest.mark.parametrize("shares,target,trades", [
    (52, 0.5, False),    # |0.52-0.5| < band -> skipped
    (56, 0.5, True),     # 0.06 >= band -> trades
    (44, 0.5, True),
    (47, 0.5, False),
    (98, 1.0, True),     # 100% target is exempt from band
    (2, 0.0, True),      # 0% target is exempt from band
])
def test_rebalance_band_and_full_exemption(shares, target, trades):
    env = make_env()
    env.reset()
    env.shares, env.cash = float(shares), 10000. - 100. * shares
    _, _, _, _, info = env.execute(target_weight=target)
    assert info["traded"] == trades


def test_full_exit_leaves_zero_shares_and_holding_target_is_not_a_trade():
    env = make_env()
    env.reset()
    env.step(4)
    assert env.cash == pytest.approx(0, abs=1e-9)
    _, _, _, _, info = env.step(4)
    assert not info["traded"] and info["fee"] == 0
    _, _, _, _, info = env.step(0)
    assert env.shares == 0 and info["traded"]


def test_observation_precedes_execution_and_includes_overnight_gap():
    frame = fixture_frame()
    frame.iloc[2, frame.columns.get_loc("Open")] = 120.
    frame.iloc[2, frame.columns.get_loc("Close")] = 120.
    features = np.tile(np.arange(len(frame))[:, None], (1, 8)).astype(np.float32)
    env = TradingEnv(frame, features, "test", transaction_cost=0)
    obs, _ = env.reset()
    assert obs[0] == 0  # The prior close, not today's feature.
    env.execute(buy_budget=10000)
    _, reward, _, _, info = env.step(1)
    assert info["value"] == pytest.approx(12000)
    assert reward == pytest.approx(100 * np.log(1.2))


def test_noop_and_api_contract():
    env = make_env()
    check_env(env, warn=True)
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == 0 and not info["traded"] and info["fee"] == 0
    with pytest.raises(ValueError):
        env.step(5)


def test_buy_hold_includes_entry_fee_and_first_day_loss_in_drawdown():
    env = make_env()
    history = evaluate(env, strategy="Buy-and-Hold")
    result = metrics(history)
    assert result["cumulative_return"] == pytest.approx(0.999 - 1)
    assert result["max_drawdown"] == pytest.approx(0.999 - 1)
    assert result["number_of_trades"] == 1
    assert len(history) == len(env.frame) - 1


def test_fixed_budget_dca_has_no_external_deposits():
    env = make_env()
    history = evaluate(env, strategy="DCA")
    assert history["traded"].sum() == 20
    assert history.iloc[-1]["cash"] == pytest.approx(0)
    assert history["cash"].min() >= -1e-8
    assert history["fee"].sum() == pytest.approx(10000 * 0.001)
    assert history.iloc[-1]["value"] == pytest.approx(10000 * 0.999)


def test_terminal_state_and_reward_telescope():
    env = make_env()
    env.reset()
    rewards = []
    while not env.finished:
        obs, reward, _, _, _ = env.step(4)
        rewards.append(reward)
    assert np.all(obs == 0)
    assert sum(rewards) == pytest.approx(100 * np.log(env.value / 10000))
    with pytest.raises(RuntimeError):
        env.step(1)


def test_all_cash_sharpe_is_undefined_not_artificially_large():
    env = make_env()
    history = evaluate(env, model=type("Cash", (), {"predict": lambda self, obs, **kw: (0, None)})())
    assert metrics(history)["sharpe_ratio"] is None


def test_feature_pipeline_does_not_use_future_rows():
    dates = pd.bdate_range("2014-09-01", periods=200)
    prices = 100 + np.sin(np.arange(200) / 10) * 10
    stock = pd.DataFrame({"Open": prices, "Close": prices, "Volume": np.arange(200) + 1000}, index=dates)
    context = pd.DataFrame({"Close": prices}, index=dates)
    original = make_features(stock, context, context, context)
    changed = stock.copy()
    changed.loc[dates[150]:, ["Close", "Volume"]] *= 2
    altered = make_features(changed, context, context, context)
    pd.testing.assert_frame_equal(original.loc[:dates[149]], altered.loc[:dates[149]])


def test_scaler_fits_only_train_dates():
    dates = pd.to_datetime(["2015-01-02", "2021-12-31", "2022-01-03", "2024-01-02"])
    frame = pd.DataFrame(np.tile(np.array([1., 3., 50., 100.])[:, None], (1, 8)),
                         index=dates, columns=MARKET_FEATURES)
    scaler = TrainScaler().fit(frame)
    assert np.all(scaler.mean == 2)
    frame.iloc[2:] = 1e9
    other = TrainScaler().fit(frame)
    np.testing.assert_array_equal(scaler.mean, other.mean)
    np.testing.assert_array_equal(scaler.std, other.std)


class ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, states):
        return self.values.repeat(len(states), 1)


def test_double_dqn_selects_online_action_not_target_argmax():
    actual = double_dqn_target(ConstantQ([1, 3, 2]), ConstantQ([100, 10, 20]),
                              torch.zeros(2, 10), torch.tensor([1., 1.]),
                              torch.tensor([0., 1.]), 0.9)
    torch.testing.assert_close(actual, torch.tensor([10., 1.]))


def test_dueling_mean_q_equals_state_value():
    model = DuelingNetwork()
    states = torch.rand(4, 10)
    torch.testing.assert_close(model(states).mean(-1), model.value(model.shared(states)).squeeze(-1))


def test_discrete_sac_update_is_finite_and_changes_actor():
    model = SACDiscrete(42)
    rng = np.random.default_rng(42)
    for i in range(100):
        model.replay.add(rng.normal(size=10), i % 3, float(rng.normal()),
                         rng.normal(size=10), i % 7 == 0)
    before = [p.detach().clone() for p in model.actor.parameters()]
    assert np.isfinite(model.update())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))
    p, _ = model.distribution(torch.zeros(2, 10))
    torch.testing.assert_close(p.sum(1), torch.ones(2))
    assert torch.isfinite(model.log_alpha)


def test_reward_telescopes_under_random_actions_and_buy_hold_matches_direct_formula():
    rng = np.random.default_rng(0)
    frame = fixture_frame(60)
    frame["Open"] = 100 * np.exp(np.cumsum(rng.normal(0, .02, 60)))
    frame["Close"] = frame["Open"] * np.exp(rng.normal(0, .01, 60))
    env = make_env(frame)
    env.reset()
    total = 0.0
    while not env.finished:
        _, reward, _, _, info = env.step(int(rng.integers(5)))
        total += reward
        assert info["cash"] >= -1e-9 and info["shares"] >= -1e-12
    assert total == pytest.approx(100 * np.log(env.value / 10000))
    history = evaluate(make_env(frame), strategy="Buy-and-Hold")
    first = make_env(frame).start
    expected = 10000 * 0.999 / frame["Open"].iloc[first] * frame["Close"].iloc[-1]
    assert history.iloc[-1]["value"] == pytest.approx(expected)
    assert metrics(history)["number_of_trades"] == 1


def test_sma_crossover_switches_between_zero_and_full():
    frame = fixture_frame()
    frame.iloc[10:, frame.columns.get_loc("sma10")] = 99.   # sma10 < sma30 from tick 10 on
    history = evaluate(make_env(frame), strategy="SMA-Crossover")
    assert history["traded"].sum() == 2
    assert history.iloc[-1]["shares"] == 0


def test_dca_installments_are_25_sessions_apart_on_a_502_day_window():
    frame = fixture_frame(503)
    frame.index = pd.bdate_range("2023-12-29", periods=503)
    env = TradingEnv(frame, np.zeros((503, 8), np.float32), (frame.index[1], frame.index[-1]))
    history = evaluate(env, strategy="DCA")
    assert np.flatnonzero(history["traded"]).tolist() == [25 * i for i in range(20)]
    assert history.iloc[-1]["cash"] == pytest.approx(0, abs=1e-9)


def test_turnover_is_traded_value_over_mean_portfolio_value_and_exposure_counts_days():
    env = make_env()
    history = evaluate(env, strategy="Buy-and-Hold")
    result = metrics(history)
    assert result["turnover"] == pytest.approx(10000 / history["value"].mean())
    assert result["exposure"] == 1.0
