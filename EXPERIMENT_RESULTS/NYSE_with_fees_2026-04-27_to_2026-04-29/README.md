# NYSE With Fees (Top30)

## Experiment ID
`NYSE_with_fees_2026-04-27_to_2026-04-29`

## Scope
- Market: NYSE
- Dataset: `data/portfolio_management/nyse_ohlc_aligned30_2008_2025`
- Fee setup: enabled (`fee_model` from `nyse_eiie_fees.py`)
- Sweep grid:
  - `init`: `1e4, 1e5, 1e6, 1e7`
  - `turnover`: `0.001, 0.01, 0.05, 0.1`
- Jobs: `378–393`

## Folder Contents (GitHub)
- `summary_runs.csv`
- `summary_metrics.md`
- `ARTIFACTS_DRIVE.md`
- `config/`
- `scripts/`

## Notes
- Heavy artifacts (`work_dir`, full `slurm_logs`) are stored on Google Drive.
- Use `ARTIFACTS_DRIVE.md` for links and integrity notes.
