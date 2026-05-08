# PSE Without Fees (Top30)

## Experiment ID

`PSE_without_fees_2026-04-13_to_2026-04-15`

## Scope

- Market: PSE
- Dataset: `data/portfolio_management/pse_top30_2008_2025`
- Fee setup: no-fee config (`fee_model=None`)
- Sweep grid:
  - `init`: `1e4, 1e5, 1e6, 1e7`
  - `turnover`: `0.001, 0.01, 0.05, 0.1`
- Jobs: `74136–74151` (PSE subset from the full NYSE+PSE sweep)

## Folder Contents (GitHub)

- `summary_runs.csv`
- `summary_metrics.md`
- `ARTIFACTS_DRIVE.md`
- `config/`
- `scripts/`
