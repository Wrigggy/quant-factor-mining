from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def test_run_walkforward_writes_stability_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "runs"
    cfg_path = tmp_path / "stability_test.yaml"

    cfg = {
        "data": {
            "mode": "snapshot",
            "snapshot_path": str(tmp_path / "snapshot.parquet"),
            "start_date": "2020-01-01",
            "end_date": "2021-12-31",
            "seed": 42,
            "max_ffill_days": 3,
            "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
        },
        "factors": {
            "momentum_lookback": 126,
            "momentum_skip": 21,
            "mean_reversion_lookback": 21,
            "volatility_window": 42,
        },
        "research": {
            "train_size": 126,
            "test_size": 42,
            "holdout_size": 42,
            "top_n": 3,
            "rebalance_frequency": 21,
            "transaction_cost_bps": 10.0,
            "initial_capital": 1_000_000,
            "bootstrap_samples": 0,
            "bootstrap_ci_level": 0.95,
            "bootstrap_seed": 42,
            "holdout_gate": {
                "min_sharpe": -999.0,
                "min_excess_total_return": -999.0,
            },
            "nested_search": {
                "enabled": True,
                "selection_mode": "stability_first",
                "selection_metric": "mean_fold_sharpe",
                "include_holdout": True,
                "space": {
                    "momentum_lookback": [126, 252],
                    "mean_reversion_lookback": [10, 21],
                    "volatility_window": [42],
                },
            },
        },
        "run": {"output_root": str(output_root)},
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(
        ["python3", "scripts/run_walkforward.py", "--config", str(cfg_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dirs = sorted(output_root.glob("*"))
    assert run_dirs, "No run directory was created"
    latest = run_dirs[-1]

    scoreboard_path = latest / "stability_scoreboard.csv"
    summary_path = latest / "stability_summary.json"
    assert scoreboard_path.exists()
    assert summary_path.exists()

    scoreboard = pd.read_csv(scoreboard_path)
    assert not scoreboard.empty

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gate_status"] in {"PASS", "FAIL"}
