"""Download actual Yahoo data; freeze snapshots and fit scaling on train only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW_DIR, MARKET_FEATURES, ROOT, SPLITS, TICKERS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_data(refresh=False):
    import time
    import yfinance as yf
    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ticker in (*TICKERS, "SPY", "^VIX", "^IRX"):
        path = raw_dir / f"{ticker.replace('^', '')}.csv"
        if path.exists() and not refresh:
            continue
        error = None
        for attempt in range(3):
            try:
                frame = yf.download(ticker, start="2014-09-01", end="2026-01-01",
                                    auto_adjust=True, progress=False, threads=False)
                if frame.empty:
                    raise RuntimeError(f"Yahoo returned no data for {ticker}")
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                frame = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.index.name = "date"
                frame.sort_index().to_csv(path)
                print(f"Downloaded {ticker}: {len(frame)} rows", flush=True)
                break
            except Exception as exc:
                error = exc
                time.sleep(attempt + 1)
        else:
            raise RuntimeError(f"Cannot fetch {ticker}; no synthetic fallback") from error
    manifest = {p.name: {"sha256": sha256(p), "rows": len(pd.read_csv(p))}
                for p in sorted(raw_dir.glob("*.csv"))}
    (ROOT / "data" / "source_manifest.json").write_text(json.dumps({
        "source": "Yahoo Finance via yfinance", "auto_adjust": True,
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "requested_start": "2014-09-01", "requested_end_exclusive": "2026-01-01",
        "files": manifest,
    }, indent=2))


def make_features(stock, spy, vix, irx):
    frame = stock.copy().sort_index()
    close = frame["Close"]
    frame["stock_return"] = close.pct_change(fill_method=None)
    frame["volume_change"] = frame["Volume"].pct_change(fill_method=None)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    frame.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    frame.loc[(loss == 0) & (gain == 0), "rsi"] = 50
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    frame["macd_ratio"] = macd / close
    for n in (10, 30):
        frame[f"sma{n}"] = close.rolling(n).mean()
        frame[f"sma{n}_ratio"] = close / frame[f"sma{n}"] - 1
    frame["market_return"] = spy["Close"].pct_change(fill_method=None).reindex(frame.index)
    frame["vix"] = vix["Close"].reindex(frame.index).ffill()
    frame["risk_free_annual"] = irx["Close"].reindex(frame.index).ffill() / 100
    # Past-only indicators, no backward fill and no target/LSTM predictions.
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    return frame


def prepare_data():
    raw_dir = RAW_DIR
    def read(name):
        return pd.read_csv(raw_dir / f"{name}.csv", index_col="date", parse_dates=True)
    context = [read(name) for name in ("SPY", "VIX", "IRX")]
    frames = {ticker: make_features(read(ticker), *context) for ticker in TICKERS}
    # Identical exchange dates for both stocks and all strategies.
    common = frames[TICKERS[0]].index.intersection(frames[TICKERS[1]].index)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        frames[ticker] = frames[ticker].loc[common].copy()
        frames[ticker].to_csv(out / f"{ticker}_features.csv")
    return frames


class TrainScaler:
    def fit(self, frame):
        x = frame.loc[SPLITS["train"][0]:SPLITS["train"][1], list(MARKET_FEATURES)]
        self.mean = x.mean().to_numpy(dtype=np.float64)
        self.std = x.std(ddof=0).to_numpy(dtype=np.float64, copy=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, frame):
        x = frame[list(MARKET_FEATURES)].to_numpy(dtype=np.float64)
        return np.clip((x - self.mean) / self.std, -5, 5).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"features": list(MARKET_FEATURES),
                                        "mean": self.mean.tolist(), "std": self.std.tolist(),
                                        "fit_split": SPLITS["train"]}, indent=2))
