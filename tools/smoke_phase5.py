"""Phase 5 smoke test. Touches Train and Validation ONLY; never evaluates on Test.

usage: python tools/smoke_phase5.py TICKER ALGORITHM STEPS SEED OUT_DIR
Trains twice with the same seed to check determinism, then reports timing,
validation metrics and the distribution of target weights chosen on Validation.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading.config import TARGET_WEIGHTS
from trading.data import prepare_data, TrainScaler
from trading.environment import TradingEnv
from trading.evaluation import evaluate, metrics
from trading.experiment import fit_agent

ticker, name, steps, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
frame = prepare_data()[ticker]
scaler = TrainScaler().fit(frame)
features = scaler.transform(frame)
env_factory = lambda split: TradingEnv(frame, features, split)


def one_run(tag):
    directory = out / f"{ticker}_{name}_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    model, selected, seconds = fit_agent(name, seed, steps, env_factory, directory)
    history = evaluate(env_factory("validation"), model=model)
    log = pd.read_csv(directory / f"{name}_{seed}_validation.csv")
    return model, selected, seconds, history, log


_, selected, seconds, history, log = one_run("a")
_, _, seconds_b, history_b, log_b = one_run("b")
counts = history["action"].value_counts().reindex(range(len(TARGET_WEIGHTS)), fill_value=0)
result = {"ticker": ticker, "algorithm": name, "steps": steps, "seed": seed,
          "seconds_run_a": seconds, "seconds_run_b": seconds_b, "selected_step": selected,
          "deterministic": bool(log.equals(log_b) and history["value"].equals(history_b["value"])),
          "validation": metrics(history),
          "action_share": {str(TARGET_WEIGHTS[i]): float(counts[i] / counts.sum()) for i in range(len(counts))},
          "distinct_actions": int((counts > 0).sum())}
(out / f"{ticker}_{name}.json").write_text(json.dumps(result, indent=2, default=float))
print(json.dumps(result, default=float))
