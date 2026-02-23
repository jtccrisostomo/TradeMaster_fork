# Slurm Job Sweep Status

## Job Array Configuration

- **Job ID**: 71989
- **Array Tasks**: 1-30 (30 total combinations)
- **Max Concurrent Jobs**: 5
- **Status**: Running

## Parameter Sweep

### Initial Amounts (5 values)
- 100k (100,000)
- 250k (250,000)
- 500k (500,000)
- 1M (1,000,000)
- 2M (2,000,000)

### Max Daily Turnover (6 values)
- 0.01 (1%)
- 0.1 (10%)
- 0.5 (50%)
- 1.0 (100%)
- 1.5 (150%)
- 2.0 (200%)

### Total Combinations: 30 (5 × 6)

## Task Mapping

Each array task corresponds to a specific parameter combination:

| Task ID | Initial Amount | Max Daily Turnover |
|---------|----------------|-------------------|
| 1-6     | 100k           | 0.01, 0.1, 0.5, 1.0, 1.5, 2.0 |
| 7-12    | 250k           | 0.01, 0.1, 0.5, 1.0, 1.5, 2.0 |
| 13-18   | 500k           | 0.01, 0.1, 0.5, 1.0, 1.5, 2.0 |
| 19-24   | 1M             | 0.01, 0.1, 0.5, 1.0, 1.5, 2.0 |
| 25-30   | 2M             | 0.01, 0.1, 0.5, 1.0, 1.5, 2.0 |

## Monitoring Commands

```bash
# Check job status
squeue -u laperia

# View output logs
tail -f TradeMaster/slurm_logs/tm-eiie-sweep-71989_*.out

# View error logs
tail -f TradeMaster/slurm_logs/tm-eiie-sweep-71989_*.err

# Cancel all jobs
scancel 71989

# Cancel specific task
scancel 71989_1
```

## Output Location

Results will be saved to:
```
TradeMaster/workdir/backtests_eiie_<timestamp>/
├── runs/
│   └── run_001/
│       ├── equity_curve.csv
│       ├── equity_curve.png
│       ├── metrics.json
│       ├── test_fee_details.csv
│       ├── daily_turnover.json
│       └── daily_trades.json
├── all_runs_metrics.csv
├── summary.json
└── total_return_hist.png
```

## Key Features

1. **Automatic Job Management**: Slurm automatically starts the next job when one finishes, maintaining exactly 5 concurrent jobs
2. **Task Limit**: The `%5` in `--array=1-30%5` ensures no more than 5 jobs run simultaneously
3. **Comprehensive Logging**: Each task has its own output and error log files
4. **Graceful Error Handling**: Failed tasks are logged but don't stop the entire sweep
