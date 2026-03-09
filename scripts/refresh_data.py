#!/usr/bin/env python3
"""Refresh local data snapshot (offline synthetic by default, live optional)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfm.data.fetch import (
    DEFAULT_SNAPSHOT_PATH,
    fetch_live_yfinance,
    generate_synthetic_market_data,
    save_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local market data snapshot")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--live", action="store_true", help="Fetch data from yfinance instead of synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data_cfg = cfg.get("data", {})
    snapshot_path = Path(data_cfg.get("snapshot_path", str(DEFAULT_SNAPSHOT_PATH)))

    if args.live:
        tickers = data_cfg.get("tickers", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
        start = data_cfg.get("start_date", "2018-01-01")
        end = data_cfg.get("end_date", "2025-12-31")
        data = fetch_live_yfinance(tickers=tickers, start=start, end=end)
        metadata = save_snapshot(data, snapshot_path)
        print(f"Live snapshot saved: {metadata}")
        return

    data = generate_synthetic_market_data(
        start=data_cfg.get("start_date", "2018-01-01"),
        end=data_cfg.get("end_date", "2024-12-31"),
        tickers=data_cfg.get("tickers"),
        seed=int(data_cfg.get("seed", 42)),
    )
    metadata = save_snapshot(data, snapshot_path)
    print(f"Synthetic snapshot saved: {metadata}")


if __name__ == "__main__":
    main()
