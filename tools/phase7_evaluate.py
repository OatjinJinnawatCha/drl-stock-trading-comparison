"""Phase 7: single evaluation of the frozen checkpoints on Test 2024-2025 and on 2026 (B10).

Gate, before anything touches Test: selection hash matches, all 30 checkpoints exist, and each
restored model reproduces its recorded Validation Sharpe. Refuses to run if results/phase7 exists.

usage: python tools/phase7_evaluate.py [--verify-only]
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading.config import ROOT, TICKERS, ALGORITHMS, SEEDS
from trading.data import make_features, prepare_data, sha256, TrainScaler
from trading.environment import TradingEnv
from trading.evaluation import evaluate, metrics
from tuning.learners import create_model

SOURCE = ROOT / "results" / "phase6"
OUT = ROOT / "results" / "phase7"
PERIODS = {"test": "test", "transfer_2026": ("2026-01-01", "2026-10-30")}
BASELINES = ("Buy-and-Hold", "SMA-Crossover", "DCA")


def frames_2026():
    def read(name):
        return pd.read_csv(ROOT / "data" / "raw_2026" / f"{name}.csv", index_col="date", parse_dates=True)
    context = [read(name) for name in ("SPY", "VIX", "IRX")]
    frames = {t: make_features(read(t), *context) for t in TICKERS}
    common = frames[TICKERS[0]].index.intersection(frames[TICKERS[1]].index)
    return {t: frames[t].loc[common] for t in TICKERS}


def load_model(record, directory, env):
    algorithm, seed = record["algorithm"], record["seed"]
    if algorithm == "PPO":
        return PPO.load(directory / f"PPO_{seed}.zip", device="cpu")
    model = create_model(algorithm, seed, record["params"], env)
    payload = torch.load(directory / f"{algorithm}_{seed}.pt", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model


def main():
    complete = json.loads((SOURCE / "COMPLETE.json").read_text())
    assert sha256(SOURCE / "selection_frozen.json") == complete["selection_sha256"], "Selection changed"
    assert complete["test_evaluated"] is False
    records = [(json.loads(p.read_text()), p.parent) for p in sorted((SOURCE / "final").glob("*/*/*/seed_*/complete.json"))]
    assert len(records) == len(TICKERS) * len(ALGORITHMS) * len(SEEDS) == 30
    main_frames = prepare_data()
    scalers = {t: TrainScaler().fit(main_frames[t]) for t in TICKERS}
    features = {t: scalers[t].transform(main_frames[t]) for t in TICKERS}

    models = []
    for record, directory in records:
        ticker = record["ticker"]
        env = TradingEnv(main_frames[ticker], features[ticker], "validation")
        model = load_model(record, directory, env)
        replay = metrics(evaluate(env, model=model))
        recorded = record["validation"]["sharpe_ratio"]
        assert (replay["sharpe_ratio"] is None and recorded is None) or abs(replay["sharpe_ratio"] - recorded) < 1e-9, \
            f"Replay mismatch {directory}: {replay['sharpe_ratio']} != {recorded}"
        models.append((record, model))
    print(f"GATE PASSED: {len(models)} checkpoints reproduce their recorded Validation Sharpe", flush=True)
    if "--verify-only" in sys.argv:
        return

    OUT.mkdir(parents=True, exist_ok=False)
    transfer_frames = frames_2026()
    rows = []
    for period, split in PERIODS.items():
        frames = main_frames if period == "test" else transfer_frames
        # Scalers always come from the original Train fit, never refit.
        scaled = {t: scalers[t].transform(frames[t]) for t in TICKERS}
        histories = OUT / period / "histories"
        histories.mkdir(parents=True)
        for ticker in TICKERS:
            for strategy in BASELINES:
                history = evaluate(TradingEnv(frames[ticker], scaled[ticker], split), strategy=strategy)
                history.to_csv(histories / f"{ticker}_{strategy}.csv", index=False)
                rows.append({"period": period, "ticker": ticker, "method": strategy, "kind": "baseline",
                             "seed": None, **metrics(history)})
        for record, model in models:
            ticker, algorithm, seed = record["ticker"], record["algorithm"], record["seed"]
            history = evaluate(TradingEnv(frames[ticker], scaled[ticker], split), model=model)
            history.to_csv(histories / f"{ticker}_{algorithm}_{seed}.csv", index=False)
            rows.append({"period": period, "ticker": ticker, "method": algorithm, "kind": "agent",
                         "seed": seed, "config": record["candidate"], **metrics(history)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "per_seed.csv", index=False)
    assert len(table) == 2 * (6 + 30)
    assert table.groupby("period")[["start", "end", "days"]].nunique().eq(1).all().all()
    (OUT / "COMPLETE.json").write_text(json.dumps({
        "selection_sha256": complete["selection_sha256"], "rows": len(table),
        "scaler": "fit on Train from data/, reused for both periods",
        "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2))
    print(f"COMPLETE {OUT}", flush=True)


if __name__ == "__main__":
    main()
