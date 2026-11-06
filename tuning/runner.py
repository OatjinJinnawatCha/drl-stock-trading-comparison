"""Budgeted search with immutable validation selection before any new test run."""
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
from trading.config import RAW_DIR, ROOT, TICKERS, ALGORITHMS, SPLITS
from trading.data import TrainScaler, prepare_data, sha256
from trading.environment import TradingEnv
from trading.evaluation import evaluate, metrics
from .learners import create_model
from trading.config import WARMUP_STEPS
from .space import candidates, DEFAULTS, SCREEN_SEEDS, CONFIRM_SEEDS, FINAL_SEEDS, SCREEN_STEPS, FULL_STEPS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def rank_candidates(rows, expected):
    """Same condition must have a defined Sharpe in every stock/seed cell."""
    result = []
    for candidate in sorted({r["candidate"] for r in rows}):
        selected = [r for r in rows if r["candidate"] == candidate]
        scores = [r["validation"]["sharpe_ratio"] for r in selected]
        eligible = (len(selected) == expected
                    and len({(r["ticker"], r["seed"]) for r in selected}) == expected
                    and all(s is not None and np.isfinite(s) for s in scores))
        result.append({"candidate": candidate, "eligible": eligible, "n_runs": len(selected),
                       "validation_sharpe_mean": float(np.mean(scores)) if eligible else None,
                       "validation_sharpe_std": float(np.std(scores, ddof=1)) if eligible else None,
                       "validation_return_mean": float(np.mean([r["validation"]["cumulative_return"] for r in selected])),
                       "validation_drawdown_mean": float(np.mean([r["validation"]["max_drawdown"] for r in selected])),
                       "params": selected[0]["params"]})
    result.sort(key=lambda r: (not r["eligible"],
                              -(r["validation_sharpe_mean"] if r["eligible"] else -1e10),
                              -r["validation_return_mean"], -r["validation_drawdown_mean"], r["candidate"]))
    for i, row in enumerate(result):
        row["rank"] = i + 1
    return result


class Study:
    def __init__(self, mode="full"):
        self.mode = mode
        self.directory = ROOT / "results" / f"concept2_tuning_v7{'_smoke' if mode == 'smoke' else ''}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.conditions = candidates(2 if mode == "smoke" else 8)
        self.screen_seeds = (101,) if mode == "smoke" else SCREEN_SEEDS
        self.confirm_seeds = (303,) if mode == "smoke" else CONFIRM_SEEDS
        self.final_seeds = (901,) if mode == "smoke" else FINAL_SEEDS
        self.screen_steps = 8192 if mode == "smoke" else SCREEN_STEPS
        self.full_steps = 8192 if mode == "smoke" else FULL_STEPS
        self.frames = prepare_data()
        self.features = {}
        for ticker, frame in self.frames.items():
            scaler = TrainScaler().fit(frame)
            self.features[ticker] = scaler.transform(frame)
            scaler.save(self.directory / f"{ticker}_scaler.json")
        sources = [*sorted((ROOT / "trading").glob("*.py")), *sorted((ROOT / "tuning").glob("*.py"))]
        protocol = {"mode": mode, "candidates": self.conditions, "screen_steps": self.screen_steps,
                    "confirm_and_final_steps": self.full_steps, "screen_seeds": self.screen_seeds,
                    "confirm_seeds": self.confirm_seeds, "final_seeds": self.final_seeds,
                    "top_k": 1 if mode == "smoke" else 2, "splits": SPLITS,
                    "selection_objective": "mean validation Sharpe across both stocks and search seeds",
                    "tie_breakers": ["higher mean return", "higher mean max_drawdown", "candidate ID"],
                    "eligibility": "defined validation Sharpe in every stock/seed; undefined is not zero",
                    "checkpoint_selection": "highest validation Sharpe; earliest checkpoint wins ties",
                    "shared_condition": "one winning parameter set per algorithm across both stocks",
                    "prior_test_exposure": "2024-2025 test was reported in v6; this is an exploratory re-use, not a new untouched holdout",
                    "environment": "unchanged v6 ledger, features, reward, costs and action space",
                    "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                    "data_sha256": {p.name: sha256(p) for p in sorted((RAW_DIR).glob("*.csv"))},
                    "concept_pdf_sha256": sha256(ROOT / "project-concept-2.pdf"),
                    "python": platform.python_version(),
                    "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "stable-baselines3", "gymnasium"]}}
        self.protocol = json.loads(json.dumps(protocol))
        path = self.directory / "protocol.json"
        if path.exists():
            if json.loads(path.read_text()) != self.protocol:
                raise ValueError("Refusing resume: protocol, source code, packages or data changed")
        else:
            write_json(path, self.protocol)
        self.protocol_hash = sha256(path)

    def env(self, ticker, split):
        if split == "test" and not (self.directory / "selection_frozen.json").exists():
            raise RuntimeError("Test evaluation forbidden until validation selection is frozen")
        return TradingEnv(self.frames[ticker], self.features[ticker], split)

    def job(self, stage, algorithm, condition, ticker, seed, steps):
        job_dir = self.directory / stage / ticker / algorithm / condition["id"] / f"seed_{seed}"
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / "complete.json"
        if path.exists():
            result = json.loads(path.read_text())
            assert result["protocol_sha256"] == self.protocol_hash
            return result
        params = condition["params"]
        env = self.env(ticker, "train")
        model = create_model(algorithm, seed, params, env)
        best_state, best_score, selected_step = None, -np.inf, None
        logs = []
        interval = 8192 if self.mode == "smoke" else (8192 if stage == "screen" else 10240)
        if algorithm == "PPO":
            assert interval % params["n_steps"] == steps % params["n_steps"] == 0
        else:
            observation, _ = env.reset(seed=seed)
        print(f"START {stage} {ticker} {algorithm} {condition['id']} seed={seed} steps={steps}", flush=True)
        start = time.monotonic()
        completed = 0
        while completed < steps:
            stop = min(completed + interval, steps)
            if algorithm == "PPO":
                model.learn(total_timesteps=stop - completed, reset_num_timesteps=False)
            else:
                for step in range(completed + 1, stop + 1):
                    action = model.sample_action(observation, step, steps)
                    next_obs, reward, done, _, _ = env.step(action)
                    model.replay.add(observation, action, reward, next_obs, done)
                    observation = next_obs
                    if step >= WARMUP_STEPS and step % 4 == 0:
                        loss = model.update()
                        if not np.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss: {job_dir}")
                        model.update_target(step)
                    if done:
                        observation, _ = env.reset()
            completed = stop
            validation = metrics(evaluate(self.env(ticker, "validation"), model=model))
            score = validation["sharpe_ratio"]
            score = float(score) if score is not None and np.isfinite(score) else -1e10
            logs.append({"step": completed, **validation})
            if best_state is None or score > best_score:
                best_score, selected_step = score, completed
                best_state = copy.deepcopy(model.policy.state_dict() if algorithm == "PPO" else model.state_dict())
        if algorithm == "PPO":
            model.policy.load_state_dict(best_state)
            model.save(job_dir / "model.zip")
        else:
            model.load_state_dict(best_state)
            torch.save({"algorithm": algorithm, "params": params, "seed": seed, "state_dict": best_state}, job_dir / "model.pt")
        validation_history = evaluate(self.env(ticker, "validation"), model=model)
        validation_history.to_csv(job_dir / "validation_history.csv", index=False)
        pd.DataFrame(logs).to_csv(job_dir / "validation_checkpoints.csv", index=False)
        result = {"stage": stage, "ticker": ticker, "algorithm": algorithm,
                  "candidate": condition["id"], "params": params, "seed": seed,
                  "training_steps": steps, "selected_step": selected_step,
                  "validation": metrics(validation_history), "protocol_sha256": self.protocol_hash,
                  "training_seconds": time.monotonic() - start,
                  "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "job_path": str(job_dir.relative_to(self.directory))}
        if stage == "final":
            # All winners are already frozen, independently of these final seeds.
            test_history = evaluate(self.env(ticker, "test"), model=model)
            test_history.to_csv(job_dir / "test_history.csv", index=False)
            result["test"] = metrics(test_history)
            result["selection_sha256"] = sha256(self.directory / "selection_frozen.json")
        write_json(path, result)
        print(f"DONE {stage} {ticker} {algorithm} {condition['id']} seed={seed} "
              f"validation Sharpe={result['validation']['sharpe_ratio']} "
              f"elapsed={result['training_seconds']:.1f}s", flush=True)
        return result

    def phase(self, stage, selected=None):
        seeds = self.screen_seeds if stage == "screen" else self.confirm_seeds
        steps = self.screen_steps if stage == "screen" else self.full_steps
        ranking = {}
        for algorithm in ALGORITHMS:
            conditions = self.conditions[algorithm] if selected is None else selected[algorithm]
            rows = []
            for condition in conditions:
                for ticker in TICKERS:
                    for seed in seeds:
                        rows.append(self.job(stage, algorithm, condition, ticker, seed, steps))
            ranking[algorithm] = rank_candidates(rows, len(TICKERS) * len(seeds))
        write_json(self.directory / f"{stage}_ranking.json", ranking)
        flat = [{"algorithm": a, **{k: v for k, v in r.items() if k != "params"},
                 "params_json": json.dumps(r["params"], sort_keys=True)} for a, rr in ranking.items() for r in rr]
        pd.DataFrame(flat).to_csv(self.directory / f"{stage}_ranking.csv", index=False)
        return ranking

    def run_search(self):
        screen = self.phase("screen")
        finalists = {}
        for algorithm in ALGORITHMS:
            eligible = [r for r in screen[algorithm] if r["eligible"]]
            if not eligible and self.mode == "smoke":
                # Integration smoke jobs need not learn a trading policy. Pick
                # a fixed condition solely to exercise save/load/final plumbing.
                # The full study never relaxes the defined-Sharpe criterion.
                eligible = [next(r for r in screen[algorithm] if r["candidate"] == "default")]
            if not eligible:
                raise RuntimeError(f"No defined-Sharpe screen condition for {algorithm}")
            finalists[algorithm] = [{"id": r["candidate"], "params": r["params"]}
                                    for r in eligible[:self.protocol["top_k"]]]
        write_json(self.directory / "shortlist.json", finalists)
        confirm = self.phase("confirm", finalists)
        winners = {}
        for algorithm in ALGORITHMS:
            eligible = [r for r in confirm[algorithm] if r["eligible"]]
            if not eligible and self.mode == "smoke":
                eligible = confirm[algorithm][:1]
            if not eligible:
                raise RuntimeError(f"No defined-Sharpe confirmation condition for {algorithm}")
            best = eligible[0]
            winners[algorithm] = {"id": best["candidate"], "params": best["params"],
                                  "validation_sharpe_mean": best["validation_sharpe_mean"]}
        path = self.directory / "selection_frozen.json"
        evidence = {"winners": winners, "protocol_sha256": self.protocol_hash,
                    "screen_ranking_sha256": sha256(self.directory / "screen_ranking.json"),
                    "confirm_ranking_sha256": sha256(self.directory / "confirm_ranking.json")}
        if path.exists():
            frozen = json.loads(path.read_text())
            assert {k: frozen[k] for k in evidence} == evidence, "Frozen selection cannot change after test"
        else:
            write_json(path, {**evidence, "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        print("SELECTION FROZEN: " + json.dumps(winners), flush=True)
        return winners

    def run_final(self, winners):
        selection_hash = sha256(self.directory / "selection_frozen.json")
        rows = []
        for algorithm in ALGORITHMS:
            # If the default wins, reuse identical jobs rather than retrain them.
            conditions = {"default": {"id": "default", "params": DEFAULTS[algorithm]}}
            winner = winners[algorithm]
            conditions[winner["id"]] = {"id": winner["id"], "params": winner["params"]}
            for ticker in TICKERS:
                for condition in conditions.values():
                    for seed in self.final_seeds:
                        result = self.job("final", algorithm, condition, ticker, seed, self.full_steps)
                        assert result["selection_sha256"] == selection_hash
                        roles = []
                        if condition["id"] == "default":
                            roles.append("Default")
                        if condition["id"] == winner["id"]:
                            roles.append("Tuned")
                        for role in roles:
                            rows.append({"ticker": ticker, "algorithm": algorithm, "condition": role,
                                         "candidate": condition["id"], "seed": seed,
                                         "job_path": result["job_path"], **result["test"]})
        baseline_dir = self.directory / "baselines"
        baseline_dir.mkdir(exist_ok=True)
        for ticker in TICKERS:
            for strategy in ("Buy-and-Hold", "SMA-Crossover", "DCA"):
                history = evaluate(self.env(ticker, "test"), strategy=strategy)
                history.to_csv(baseline_dir / f"{ticker}_{strategy}.csv", index=False)
                rows.append({"ticker": ticker, "algorithm": strategy, "condition": "Baseline",
                             "candidate": None, "seed": None, "job_path": None, **metrics(history)})
        final = pd.DataFrame(rows)
        final.to_csv(self.directory / "test_per_seed.csv", index=False)
        assert len(final) == 2 * 3 * 2 * len(self.final_seeds) + 6
        assert final.start.nunique() == final.end.nunique() == final.days.nunique() == 1
        from .report import build_report
        build_report(self.directory)
        completed_jobs = [json.loads(p.read_text()) for stage in ("screen", "confirm", "final")
                          for p in (self.directory / stage).glob("**/complete.json")]
        write_json(self.directory / "COMPLETE.json", {
            "mode": self.mode, "training_jobs": len(completed_jobs),
            "total_training_transitions": sum(r["training_steps"] for r in completed_jobs),
            "test_rows_including_baselines": len(final), "protocol_sha256": self.protocol_hash,
            "selection_sha256": selection_hash,
            "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args(argv)
    study = Study(args.mode)
    winners = study.run_search()
    study.run_final(winners)
    print(f"COMPLETE {study.directory}", flush=True)


if __name__ == "__main__":
    main()
