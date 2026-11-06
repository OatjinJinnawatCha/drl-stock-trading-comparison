import argparse
import copy
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from .agents import DuelingDDQN, SACDiscrete, set_seed
from .config import (RAW_DIR, ROOT, TICKERS, SEEDS, ALGORITHMS, FULL_STEPS, EVAL_INTERVAL,
                     SPLITS, STATE_FEATURES, INITIAL_CAPITAL, TRANSACTION_COST, HIDDEN,
                     WARMUP_STEPS, BATCH_SIZE, TARGET_WEIGHTS, REBALANCE_BAND)
from .data import download_data, prepare_data, TrainScaler, sha256
from .environment import TradingEnv
from .evaluation import evaluate, metrics


def fit_agent(name, seed, steps, env_factory, checkpoint_dir, params=None):
    """Train one run. `params` (tuning.space-style dict) overrides the default hyperparameters."""
    set_seed(seed)
    env = env_factory("train")
    validation_log, best_state, best_score, selected_step = [], None, -np.inf, None
    if params is not None:
        from tuning.learners import create_model
        model = create_model(name, seed, params, env)
        if name != "PPO":
            observation, _ = env.reset(seed=seed)
    elif name == "PPO":
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                    learning_rate=3e-4, n_steps=2048, batch_size=BATCH_SIZE, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs={"net_arch": {"pi": [HIDDEN, HIDDEN], "vf": [HIDDEN, HIDDEN]},
                                   "activation_fn": torch.nn.ReLU})
    else:
        model = DuelingDDQN(seed) if name == "DuelingDDQN" else SACDiscrete(seed)
        observation, _ = env.reset(seed=seed)
    start_time = time.monotonic()
    completed = 0
    while completed < steps:
        stop = min(completed + EVAL_INTERVAL, steps)
        if name == "PPO":
            model.learn(total_timesteps=stop - model.num_timesteps, reset_num_timesteps=False)
            completed = stop
        else:
            for step in range(completed + 1, stop + 1):
                action = model.sample_action(observation, step, steps)
                next_obs, reward, done, _, _ = env.step(action)
                model.replay.add(observation, action, reward, next_obs, done)
                observation = next_obs
                if step >= WARMUP_STEPS and step % 4 == 0:
                    loss = model.update()
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss in {name} seed {seed}")
                    model.update_target(step)
                if done:
                    observation, _ = env.reset()
            completed = stop
        validation = metrics(evaluate(env_factory("validation"), model=model))
        score = validation["sharpe_ratio"]
        if score is None or not np.isfinite(score):
            score = -1e10
        validation_log.append({"step": completed, **validation})
        if best_state is None or score > best_score:
            best_score, selected_step = score, completed
            if name == "PPO":
                best_state = copy.deepcopy(model.policy.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())
        print(f"  {name} seed={seed} step={completed}/{steps} "
              f"validation Sharpe={score:.3f} return={validation['cumulative_return']:.2%} "
              f"elapsed={time.monotonic() - start_time:.0f}s", flush=True)
    if name == "PPO":
        model.policy.load_state_dict(best_state)
        model.save(checkpoint_dir / f"{name}_{seed}.zip")
    else:
        model.load_state_dict(best_state)
        torch.save({"algorithm": name, "seed": seed, "state_dict": best_state},
                   checkpoint_dir / f"{name}_{seed}.pt")
    pd.DataFrame(validation_log).to_csv(checkpoint_dir / f"{name}_{seed}_validation.csv", index=False)
    return model, selected_step, time.monotonic() - start_time


def aggregate(run_dir):
    rows = []
    for path in sorted((run_dir / "runs").glob("*.json")):
        rows.append(json.loads(path.read_text()))
    results = pd.DataFrame(rows)
    results["sharpe_ratio"] = pd.to_numeric(results["sharpe_ratio"], errors="raise")
    results.to_csv(run_dir / "metrics_per_seed.csv", index=False)
    fields = ["cumulative_return", "sharpe_ratio", "max_drawdown", "number_of_trades",
              "turnover", "total_fees", "final_value", "mean_position_ratio"]
    grouped = results.groupby(["ticker", "method"])
    summary = grouped[fields].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["n_runs"] = grouped.size()
    summary = summary.reset_index()
    # Deterministic baselines have one observation; std deliberately remains NA.
    summary.to_csv(run_dir / "summary.csv", index=False)
    return results, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    steps = FULL_STEPS if args.mode == "full" else 1024
    seeds = SEEDS if args.mode == "full" else SEEDS[:1]
    if args.download or args.refresh_data:
        download_data(refresh=args.refresh_data)
    frames = prepare_data()
    run_dir = ROOT / "results" / f"concept2_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for part in ("runs", "histories", "checkpoints", "figures"):
        (run_dir / part).mkdir(exist_ok=True)
    packages = ["numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "yfinance"]
    config = {"mode": args.mode, "training_steps_per_run": steps, "seeds": list(seeds),
              "algorithms": list(ALGORITHMS), "tickers": list(TICKERS), "splits": SPLITS,
              "state_features": STATE_FEATURES, "initial_capital": INITIAL_CAPITAL,
              "transaction_cost": TRANSACTION_COST, "eval_interval": EVAL_INTERVAL,
              "actions": {str(i): f"target stock weight {w:.0%}" for i, w in enumerate(TARGET_WEIGHTS)},
              "rebalance_band": REBALANCE_BAND,
              "selection": "maximum validation Sharpe; earlier checkpoint wins ties; no retraining on validation",
              "source_pdf_sha256": sha256(ROOT / "project-concept-2.pdf"),
              "data_sha256": {p.name: sha256(p) for p in sorted((RAW_DIR).glob("*.csv"))},
              "code_sha256": {p.name: sha256(p) for p in sorted((ROOT / "trading").glob("*.py"))},
              "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in packages}}
    # Round-trip tuples to lists for comparison against saved JSON.
    config = json.loads(json.dumps(config))
    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"{run_dir} already exists; use --resume to verify and continue")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume refused: data, code, dependencies or configuration changed")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    for ticker in TICKERS:
        frame = frames[ticker]
        scaler = TrainScaler().fit(frame)
        scaler.save(run_dir / f"{ticker}_scaler.json")
        features = scaler.transform(frame)
        def env_factory(split):
            return TradingEnv(frame, features, split)
        checkpoint_dir = run_dir / "checkpoints" / ticker
        checkpoint_dir.mkdir(exist_ok=True)
        for strategy in ("Buy-and-Hold", "SMA-Crossover", "DCA"):
            key = f"{ticker}_{strategy}_baseline"
            path = run_dir / "runs" / f"{key}.json"
            if path.exists() and args.resume:
                continue
            history = evaluate(env_factory("test"), strategy=strategy)
            history.to_csv(run_dir / "histories" / f"{key}.csv", index=False)
            row = {"ticker": ticker, "method": strategy, "seed": None, **metrics(history)}
            path.write_text(json.dumps(row, indent=2, allow_nan=False))
        for name in ALGORITHMS:
            for seed in seeds:
                key = f"{ticker}_{name}_{seed}"
                path = run_dir / "runs" / f"{key}.json"
                if path.exists() and args.resume:
                    print(f"Skip verified complete run {key}", flush=True)
                    continue
                print(f"Training {key}", flush=True)
                model, selected, seconds = fit_agent(name, seed, steps, env_factory, checkpoint_dir)
                # Test is accessed only after checkpoint selection has finished.
                history = evaluate(env_factory("test"), model=model)
                history.to_csv(run_dir / "histories" / f"{key}.csv", index=False)
                row = {"ticker": ticker, "method": name, "seed": seed,
                       "selected_step": selected, "training_steps": steps,
                       "training_seconds": seconds, **metrics(history)}
                path.write_text(json.dumps(row, indent=2, allow_nan=False))
                sharpe_text = "NA (constant portfolio)" if row['sharpe_ratio'] is None else f"{row['sharpe_ratio']:.3f}"
                print(f"Finished {key}: test return={row['cumulative_return']:.2%}, "
                      f"Sharpe={sharpe_text}", flush=True)
                del model
        aggregate(run_dir)
    results, summary = aggregate(run_dir)
    expected = len(TICKERS) * (3 * len(seeds) + 3)
    assert len(results) == expected, f"Incomplete run: {len(results)} / {expected}"
    assert results["start"].nunique() == results["end"].nunique() == results["days"].nunique() == 1
    from .report import build_report
    build_report(run_dir)
    (run_dir / "COMPLETE.json").write_text(json.dumps({
        "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "rows": len(results),
        "agent_runs": len(TICKERS) * 3 * len(seeds),
        "total_training_transitions": len(TICKERS) * 3 * len(seeds) * steps,
    }, indent=2))
    print(summary[["ticker", "method", "cumulative_return_mean", "sharpe_ratio_mean",
                   "max_drawdown_mean", "number_of_trades_mean"]].to_string(index=False), flush=True)
    print(f"Complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
