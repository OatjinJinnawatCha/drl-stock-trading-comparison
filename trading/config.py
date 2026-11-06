from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("AMD", "OXY")
SEEDS = (42, 123, 2024, 7, 2025)
SPLITS = {
    "train": ("2015-01-01", "2021-12-31"),
    "validation": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}
MARKET_FEATURES = (
    "stock_return", "volume_change", "rsi", "macd_ratio", "sma10_ratio",
    "sma30_ratio", "market_return", "vix",
)
STATE_FEATURES = (*MARKET_FEATURES, "cash_ratio", "position_ratio")
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001
FULL_STEPS = 204_800  # 100 PPO rollouts of 2,048
EVAL_INTERVAL = 20_480  # 10 rollouts
TARGET_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
REBALANCE_BAND = 0.05
HIDDEN = 128
WARMUP_STEPS = 5_000
BATCH_SIZE = 256
ALGORITHMS = ("DuelingDDQN", "PPO", "SAC-Discrete")
RAW_DIR = ROOT / "data"  # AMD/OXY/SPY/VIX/IRX .csv; 2026 extension is in data/raw_2026
