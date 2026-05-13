# PSE With Fees (With Board Lot)

## Experiment ID
`PSE_with_fees_with_boardlot_2026-05-06_to_2026-05-08`

## Scope
- Market: PSE
- Dataset: `data/portfolio_management/pse_top30_2008_2025`
- Fee setup: enabled (`fee_model="pse"`)
- Board lot: ON
- Sweep grid:
  - `init`: `1e4, 1e5, 1e6, 1e7`
  - `turnover`: `0.001, 0.01, 0.05, 0.1`
- Jobs: `684–699`

## Folder Contents (GitHub)
- `summary_runs.csv`
- `summary_metrics.md`
- `ARTIFACTS_DRIVE.md`
- `config/`
- `scripts/`

## Notes
- Heavy artifacts (`work_dir`, full `slurm_logs`) are stored on Google Drive.
- Use `ARTIFACTS_DRIVE.md` for links and integrity notes.
