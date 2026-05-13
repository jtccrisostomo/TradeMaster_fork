#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

CLOSE_CANDS = ["adjclose", "adjusted_close", "adjcp", "close", "close_price"]
DATE_CANDS = ["date", "Date", "trade_date"]


def load_split_dir(split_dir: Path, start: pd.Timestamp, end: pd.Timestamp, label: str) -> pd.Series:
    """
    Load train/valid/test CSVs from a backtest-ready directory and return a Series of close prices.
    Expected columns include 'date' and one of CLOSE_CANDS.
    """
    prices = []
    for split in ["train.csv", "valid.csv", "test.csv"]:
        p = split_dir / split
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        close_col = next((cols[c] for c in CLOSE_CANDS if c in cols), None)
        if close_col is None:
            continue
        date_col = next((c for c in DATE_CANDS if c in df.columns), None)
        if date_col is None:
            continue
        try:
            df["date"] = pd.to_datetime(df[date_col])
        except Exception:
            continue
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        s = pd.to_numeric(df[close_col], errors="coerce").dropna()
        s = s[s > 0]
        if not s.empty:
            prices.append(s)
    if not prices:
        raise RuntimeError(f"No valid prices found for {label} in {split_dir}")
    return pd.concat(prices, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Compute PSE percentile of a price and NYSE price at that percentile.")
    ap.add_argument(
        "--pse-dir",
        type=Path,
        default=Path("data/portfolio_management/pse_top30_2008_2025"),
        help="Directory containing PSE backtest splits (train/valid/test).",
    )
    ap.add_argument(
        "--nyse-dir",
        type=Path,
        default=Path("data/portfolio_management/nyse_ohlc_aligned30_2008_2025"),
        help="Directory containing NYSE backtest splits (train/valid/test).",
    )
    ap.add_argument("--start", default="2008-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    ap.add_argument("--value", type=float, default=20.0, help="PHP price to score in PSE")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # Resolve to absolute paths to avoid double-prefixed paths when invoked from repo root.
    def resolve_dir(path: Path) -> Path:
        p = path.expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"Data directory not found: {path}")
        return p

    args.pse_dir = resolve_dir(args.pse_dir)
    args.nyse_dir = resolve_dir(args.nyse_dir)

    pse_series = load_split_dir(args.pse_dir, start, end, "PSE")
    pct = percentileofscore(pse_series, args.value, kind="weak")

    nyse_series = load_split_dir(args.nyse_dir, start, end, "NYSE")
    # np.percentile uses 'interpolation' kw in older NumPy; 'method' in newer versions.
    try:
        nyse_price = np.percentile(nyse_series, pct, method="linear")
    except TypeError:
        nyse_price = np.percentile(nyse_series, pct, interpolation="linear")

    print(f"PSE observations: {len(pse_series):,}")
    print(f"broker_fee_min_percentile (value={args.value}): {pct:.2f}")
    print(f"NYSE observations: {len(nyse_series):,}")
    print(f"NYSE price at {pct:.2f} percentile: {nyse_price:.2f} USD")

if __name__ == "__main__":
    main()
