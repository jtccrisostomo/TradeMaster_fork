# EXPERIMENT_RESULTS Structure

This directory stores curated experiment artifacts for reproducibility and reporting.

## Naming Convention

Use one subfolder per experiment, named as:

`<MARKET>_<setup>_<date-or-date-range>`

Examples:
- `NYSE_without_fees_2026-04-13_to_2026-04-15`
- `PSE_fees_boardlot_on_2026-05-06_to_2026-05-08`

## Standard Contents per Experiment Folder

Each experiment folder should contain:

- `slurm_logs/`  
  SLURM `.out` and `.err` logs for all jobs in the experiment.

- `work_dir/`  
  Backtest/training run outputs copied from `workdir/...` (one run folder per job).

- `config/`  
  Exact config file(s) used in the run.

- `scripts/`  
  Submission/run scripts used (e.g., `sbatch_backtest_eiie.sh`).

- `README.md`  
  Experiment-specific documentation (job mapping, date window, setup details).

## Notes

- Keep original source outputs untouched; only copy into this archive.
- Do not mix different experiments in one folder.
- Prefer immutable snapshots for traceability.
