#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Parameter mapping for job array 71989
# Tasks 1-30 map to (initial_amount, max_daily_turnover) combinations
INITIAL_AMOUNTS = [100000, 250000, 500000, 1000000, 2000000]
MAX_DAILY_TURNOVERS = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]

def get_task_params(task_id):
    """Get parameters for a given task ID (1-30)"""
    idx = task_id - 1
    initial_idx = idx // 6
    turnover_idx = idx % 6
    return INITIAL_AMOUNTS[initial_idx], MAX_DAILY_TURNOVERS[turnover_idx]

def format_amount(amount):
    """Format amount as k or M"""
    if amount >= 1000000:
        return f"{amount/1000000:.1f}M"
    else:
        return f"{amount/1000:.0f}k"

# Find all completed backtests
workdir = Path("TradeMaster/workdir")
backtests = [d for d in workdir.glob("backtests_eiie_2026020[78]*") if (d / "summary.json").exists()]

# Create a mapping from (initial_amount, max_daily_turnover) to backtest directory
backtest_map = {}
for backtest_dir in backtests:
    try:
        # Read manifest to get parameters
        with open(backtest_dir / "manifest.json") as f:
            manifest = json.load(f)

        init_amount = manifest.get("initial_amount", 0)
        max_turnover = manifest.get("max_daily_turnover", 0)

        # Use tuple as key
        key = (init_amount, max_turnover)
        backtest_map[key] = backtest_dir
    except Exception as e:
        print(f"Error reading manifest for {backtest_dir}: {e}")

print("Job#\tRuntime\tTurnover\tInit\tFinal Value\tTotal Return\tSharpe\tVolatility\tMax Drawdown")

# Iterate through all 30 tasks in order
for task_id in range(1, 31):
    init_amount, max_turnover = get_task_params(task_id)
    key = (init_amount, max_turnover)

    if key in backtest_map:
        backtest_dir = backtest_map[key]
        try:
            # Read summary
            with open(backtest_dir / "summary.json") as f:
                summary = json.load(f)

            # Read all_runs_metrics.csv to get final value
            metrics_file = backtest_dir / "all_runs_metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                if not df.empty:
                    final_value = df['final_value'].iloc[0]
                else:
                    final_value = init_amount
            else:
                final_value = init_amount

            # Extract metrics
            total_return = summary.get("total_return", {})
            sharpe = summary.get("sharpe", {})
            volatility = summary.get("volatility_pct_daily", {})
            max_dd = summary.get("max_drawdown_pct", {})

            # Get runtime from directory timestamp
            dir_name = backtest_dir.name
            timestamp_str = dir_name.split("_")[-1]
            try:
                timestamp = datetime.strptime(timestamp_str, "%H%M%S")
                runtime = f"{timestamp.hour:02d}:{timestamp.minute:02d}"
            except:
                runtime = "N/A"

            # Format output
            print(f"{task_id}\t{runtime}\t{max_turnover}\t{format_amount(init_amount)}\t{final_value:.0f}\t"
                  f"{total_return.get('mean', 0):.1f}\t{sharpe.get('mean', 0):.3f}\t"
                  f"{volatility.get('mean', 0):.2f}%\t{max_dd.get('mean', 0):.2f}%")
        except Exception as e:
            print(f"Error processing {backtest_dir}: {e}")
    else:
        # Task not yet completed
        print(f"{task_id}\tRUNNING\t{max_turnover}\t{format_amount(init_amount)}\t-\t-\t-\t-\t-")
