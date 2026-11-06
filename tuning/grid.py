"""Phase 6: the grid of EXPERIMENT_PLAN.md section 4 (widened from 2x2 to 3x3, see DECISIONS G9). Train/Validation only, never Test.

  tune : 3 agents x 2 tickers x 9 configs x 5 tune seeds = 270 runs
  select: per (agent, ticker) best mean validation Sharpe, frozen to selection_frozen.json
  final: retrain each selected config with the 5 experiment seeds (30 runs), keep best-validation checkpoints
Test and the 2026 transfer are evaluated later (Phase 7), not here.
"""
import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from trading.config import ROOT, TICKERS, ALGORITHMS, SEEDS, FULL_STEPS
from trading.data import prepare_data, TrainScaler, sha256
from trading.environment import TradingEnv
from .runner import rank_candidates, write_json
from .space import DEFAULTS

GRID = {"learning_rate": (1e-4, 3e-4, 1e-3), "gamma": (0.95, 0.99, 0.995)}
TUNE_SEEDS = (101, 202, 303, 404, 505)  # disjoint from the experiment seeds in config.SEEDS


def configs(algorithm):
    result = []
    for lr, gamma in itertools.product(*GRID.values()):
        result.append({"id": f"lr{lr:.0e}_g{gamma}", "params": {**DEFAULTS[algorithm], "learning_rate": lr, "gamma": gamma}})
    return result


_STATE = {}


def _init(frames):
    _STATE["frames"] = frames


def run_job(job):
    from trading.experiment import fit_agent
    stage, ticker, algorithm, config, seed, steps, root = job
    directory = Path(root) / stage / ticker / algorithm / config["id"] / f"seed_{seed}"
    done = directory / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    frame = _STATE["frames"][ticker]
    scaler = TrainScaler().fit(frame)
    features = scaler.transform(frame)
    env_factory = lambda split: TradingEnv(frame, features, split)
    model, selected, seconds = fit_agent(algorithm, seed, steps, env_factory, directory, params=config["params"])
    log = pd.read_csv(directory / f"{algorithm}_{seed}_validation.csv")
    row = log[log.step == selected].iloc[0]
    sharpe = None if pd.isna(row.sharpe_ratio) else float(row.sharpe_ratio)
    result = {"stage": stage, "ticker": ticker, "algorithm": algorithm, "candidate": config["id"],
              "params": config["params"], "seed": seed, "training_steps": steps, "selected_step": int(selected),
              "validation": {"sharpe_ratio": sharpe, "cumulative_return": float(row.cumulative_return),
                             "max_drawdown": float(row.max_drawdown), "number_of_trades": int(row.number_of_trades)},
              "training_seconds": seconds}
    write_json(done, result)
    return result


def run_all(jobs, workers, frames):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(frames,)) as pool:
        out = []
        for result in pool.map(run_job, jobs):
            out.append(result)
            v = result["validation"]
            print(f"DONE {result['stage']} {result['ticker']} {result['algorithm']} {result['candidate']} "
                  f"seed={result['seed']} sharpe={v['sharpe_ratio']} ret={v['cumulative_return']:.2%} "
                  f"{result['training_seconds']:.0f}s ({len(out)}/{len(jobs)})", flush=True)
        return out


def select(rows, root):
    ranking, winners = {}, {}
    for algorithm in ALGORITHMS:
        for ticker in TICKERS:
            subset = [r for r in rows if r["algorithm"] == algorithm and r["ticker"] == ticker]
            ranked = rank_candidates(subset, len(TUNE_SEEDS))
            ranking[f"{ticker}_{algorithm}"] = ranked
            eligible = [r for r in ranked if r["eligible"]]
            if not eligible:
                raise RuntimeError(f"No config with defined Sharpe in every seed: {ticker} {algorithm}")
            winners[f"{ticker}_{algorithm}"] = {"id": eligible[0]["candidate"], "params": eligible[0]["params"],
                                               "validation_sharpe_mean": eligible[0]["validation_sharpe_mean"]}
    write_json(Path(root) / "tune_ranking.json", ranking)
    path = Path(root) / "selection_frozen.json"
    evidence = {"winners": winners, "tune_ranking_sha256": sha256(Path(root) / "tune_ranking.json")}
    if path.exists():
        assert json.loads(path.read_text())["winners"] == winners, "Frozen selection cannot change"
    else:
        write_json(path, {**evidence, "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    return winners


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "final", "all"], default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--out", default=str(ROOT / "results" / "phase6"))
    args = parser.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    frames = prepare_data()
    start = time.monotonic()
    if args.stage in ("tune", "all"):
        jobs = [("tune", t, a, c, s, args.steps, str(root)) for a in ALGORITHMS for t in TICKERS
                for c in configs(a) for s in TUNE_SEEDS]
        rows = run_all(jobs, args.workers, frames)
        winners = select(rows, root)
        print("SELECTION FROZEN " + json.dumps({k: v["id"] for k, v in winners.items()}), flush=True)
    if args.stage in ("final", "all"):
        winners = json.loads((root / "selection_frozen.json").read_text())["winners"]
        jobs = [("final", t, a, {"id": winners[f"{t}_{a}"]["id"], "params": winners[f"{t}_{a}"]["params"]}, s,
                 args.steps, str(root)) for a in ALGORITHMS for t in TICKERS for s in SEEDS]
        rows = run_all(jobs, args.workers, frames)
        write_json(root / "COMPLETE.json", {
            "final_runs": len(rows), "steps": args.steps, "seeds": list(SEEDS),
            "selection_sha256": sha256(root / "selection_frozen.json"), "test_evaluated": False,
            "elapsed_seconds": time.monotonic() - start,
            "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        print(f"COMPLETE {root}", flush=True)


if __name__ == "__main__":
    main()
