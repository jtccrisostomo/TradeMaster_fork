# EIIE Parameter Sweep Report - Job 71989

**Date Range:** 1/30/2026–1/14/2026

**CONFIG:**
`configs/portfolio_management/portfolio_management_pse_top30_eiie_eiie_adam_mse.py`

**Dataset:**
- dataset_path: `data/portfolio_management/pse`
- dataset_name: `pse_top30`

**Model:**
- net_name: `eiie`
- agent_name: `eiie`

**Training Parameters:**
- epochs: 10
- batch_size: 128
- horizon_len: 768
- buffer_size: 4500
- length_day: 10
- learning_rate: 7.139134316876125e-05
- policy_update_frequency: 1000
- tech_indicator_list: `['zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30']`

**Fee Model:**
`pse | weight_threshold=0.001 | gross_threshold=100.0 | stt_cutoff=2025-07-01`

---

## Parameter Sweep

**Initial Amounts:** {100k, 250k, 500k, 1M, 2M}
**Turnover Limits:** {0.01, 0.1, 0.5, 1.0, 1.5, 2.0}

**Turnover limit ∈ {0.01, 0.1, 0.5, 1.0, 1.5, 2.0}**

---

## Results Summary

**Total runs:** 30 (one per Init × turnover point listed)

**Completed:** 26/30 (87%)
**Failed:** 4/30 (13%) - Tasks 1, 3, 4, 5 (100k init with 0.01, 0.5, 1.0, 1.5 turnover)
**Resubmitted:** Job 72029 (4 retry tasks)

---

## Key Findings

1. **Performance improves with higher Init at same turnover**
   - Higher initial amounts generally show better performance across all turnover levels
   - 2M init shows best returns (-1.4% to -6.3%)
   - 100k init shows worst returns (-2.7% to -27.9%)

2. **Low turnover (0.01) is least negative across all inits**
   - Best performing task: Job 25 (2.0M, 0.01 turnover) with -1.4% return, -0.135 Sharpe
   - Low turnover consistently shows smaller drawdowns

3. **At higher turnover (≥0.5), returns and Sharpe are strongly negative across all inits, with large drawdowns**
   - Worst performing task: Job 6 (100k, 2.0 turnover) with -27.9% return, -4.245 Sharpe
   - Turnover ≥ 0.5 shows significant degradation in all metrics

4. **Volatility stays ~0.97–0.98% and drawdown generally decreases as Init increases**
   - Consistent volatility across different initial amounts
   - Drawdowns tend to be smaller with larger initial capital

---

## Performance by Initial Amount

| Init | Best Turnover | Best Return | Best Sharpe | Worst Turnover | Worst Return |
|-------|----------------|--------------|--------------|-----------------|---------------|
| 100k | 0.1 | -5.7% | -1.409 | 2.0 | -27.9% |
| 250k | 0.01 | -2.7% | -1.755 | 2.0 | -23.3% |
| 500k | 0.01 | -2.1% | -0.368 | 2.0 | -14.9% |
| 1.0M | 0.01 | -2.2% | -0.384 | 2.0 | -9.7% |
| 2.0M | 0.01 | -1.4% | -0.135 | 2.0 | -6.3% |

---

## Performance by Turnover Level

| Turnover | Best Init | Best Return | Best Sharpe | Worst Init | Worst Return |
|----------|-----------|--------------|--------------|-------------|---------------|
| 0.01 | 2.0M | -1.4% | -0.135 | 100k | -2.7% |
| 0.1 | 500k | -2.1% | -0.368 | 100k | -5.7% |
| 0.5 | 2.0M | -6.3% | -0.421 | 100k | -23.3% |
| 1.0 | 2.0M | -6.3% | -0.421 | 100k | -14.9% |
| 1.5 | 2.0M | -6.3% | -0.421 | 100k | -23.3% |
| 2.0 | 2.0M | -6.3% | -0.421 | 100k | -27.9% |

---

## Runtime Statistics

**Runtimes ~4:10–4:26 (two runs around 1:34–1:36)**

| Task | Runtime | Task | Runtime | Task | Runtime |
|------|---------|------|---------|------|---------|
| 2 | 10:33 | 13 | 10:31 | 19 | 21:05 |
| 6 | 18:03 | 20 | 21:07 | 25 | 22:36 |
| 7 | 18:05 | 22 | 22:30 | 26 | 23:52 |
| 8 | 18:06 | 23 | 22:33 | 27 | 23:57 |
| 9 | 18:07 | 24 | 22:33 | 28 | 00:02 |
| 10 | 18:08 | 25 | 22:33 | 29 | 00:02 |
| 11 | 19:32 | 26 | 23:52 | 30 | 00:06 |

---

## Detailed Results

| Job# | Runtime | Turnover | Init | Final Value | Total Return | Sharpe | Volatility | Max Drawdown |
|-------|----------|----------|-------|-------------|--------------|---------|------------|---------------|
| 1 | FAILED | 0.01 | 100k | - | - | - | - | - |
| 2 | 10:33 | 0.1 | 100k | 94336 | -5.7 | -1.409 | 0.30% | 5.90% |
| 3 | FAILED | 0.5 | 100k | - | - | - | - | - |
| 4 | FAILED | 1.0 | 100k | - | - | - | - | - |
| 5 | FAILED | 1.5 | 100k | - | - | - | - | - |
| 6 | 18:03 | 2.0 | 100k | 72083 | -27.9 | -4.245 | 0.56% | 28.64% |
| 7 | 18:05 | 0.01 | 250k | 243145 | -2.7 | -1.755 | 0.12% | 3.46% |
| 8 | 18:06 | 0.1 | 250k | 241932 | -3.2 | -0.371 | 0.58% | 9.10% |
| 9 | 18:07 | 0.5 | 250k | 191724 | -23.3 | -2.372 | 0.80% | 26.04% |
| 10 | 18:08 | 1.0 | 250k | 191724 | -23.3 | -2.372 | 0.80% | 26.04% |
| 11 | 19:32 | 1.5 | 250k | 191724 | -23.3 | -2.372 | 0.80% | 26.04% |
| 12 | 19:33 | 2.0 | 250k | 191724 | -23.3 | -2.372 | 0.80% | 26.04% |
| 13 | 10:31 | 0.01 | 500k | 977623 | -2.1 | -0.368 | 0.38% | 5.78% |
| 14 | 19:37 | 0.1 | 500k | 458728 | -8.3 | -0.677 | 0.85% | 15.14% |
| 15 | 19:39 | 0.5 | 500k | 425642 | -14.9 | -1.252 | 0.90% | 18.99% |
| 16 | 20:59 | 1.0 | 500k | 425642 | -14.9 | -1.252 | 0.90% | 18.99% |
| 17 | 21:00 | 1.5 | 500k | 425642 | -14.9 | -1.252 | 0.90% | 18.99% |
| 18 | 21:05 | 2.0 | 500k | 425642 | -14.9 | -1.252 | 0.90% | 18.99% |
| 19 | 21:05 | 0.01 | 1.0M | 977623 | -2.2 | -0.384 | 0.40% | 5.86% |
| 20 | 21:07 | 0.1 | 1.0M | 903464 | -9.7 | -0.724 | 0.94% | 17.44% |
| 21 | 22:26 | 0.5 | 1.0M | 902791 | -9.7 | -0.730 | 0.94% | 17.45% |
| 22 | 22:30 | 1.0 | 1.0M | 902791 | -9.7 | -0.730 | 0.94% | 17.45% |
| 23 | 22:33 | 1.5 | 1.0M | 902791 | -9.7 | -0.730 | 0.94% | 17.45% |
| 24 | 22:33 | 2.0 | 1.0M | 902791 | -9.7 | -0.730 | 0.94% | 17.45% |
| 25 | 22:36 | 0.01 | 2.0M | 1971544 | -1.4 | -0.135 | 0.58% | 7.35% |
| 26 | 23:52 | 0.1 | 2.0M | 1873982 | -6.3 | -0.423 | 0.96% | 16.48% |
| 27 | 23:57 | 0.5 | 2.0M | 1874387 | -6.3 | -0.421 | 0.96% | 16.48% |
| 28 | 00:02 | 1.0 | 2.0M | 1874387 | -6.3 | -0.421 | 0.96% | 16.48% |
| 29 | 00:02 | 1.5 | 2.0M | 1874387 | -6.3 | -0.421 | 0.96% | 16.48% |
| 30 | 00:06 | 2.0 | 2.0M | 1874387 | -6.3 | -0.421 | 0.96% | 16.48% |

---

## Output Location

All results saved in: `/home/laperia/TradeMaster/workdir/backtests_eiie_*/`

Slurm logs: `/home/laperia/TradeMaster/slurm_logs/`
