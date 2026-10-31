# GreekFire research audit

> Historical preparation record. Superseded for current methodology by the [October 30 audit](../docs/AUDIT_2026-10-30.md). Numbers and unresolved-code descriptions below refer to the earlier implementation.

## 1. What changed

The existing Julia strategy remains a walk-forward RF/XGBoost stock ranker with an optional (disabled) short overlay. The training selector now checks the actual end date of each stock's forward label. Portfolio accounting carries both gross and cost-adjusted net long/linear returns. Validation exports aligned benchmark prices and explicit cost/turnover fields. A fixed, serial sensitivity runner and a generated report provide reproducible comparisons without selecting a winning historical configuration.

## 2. Problems discovered and evidence

- Training used a 10-calendar-day label cutoff although targets use the close five stock sessions later. Holiday gaps or missing stock sessions can extend beyond that buffer. The regression test includes a long gap and observes rows with a feature date before train end but a target after it.
- The recorded `test_end` was one session before the target's fifth close. Validation ETF exports also used close-to-close returns or a fabricated zero for missing prices. Both disagreed with the strategy's next-open to fifth-close execution.
- A price series with no open column (or an unadjustable raw close) could substitute a close or an invalid adjusted open. Those observations cannot substantiate executable next-open returns.
- Long and linear portfolio returns omitted explicit trading costs. The existing short return caps a loss at 7% ex post; it is not a simulated intraday stop. The short overlay remains disabled and is not claimed as tradable.
- Previous 2020-2026 analysis had already inspected/tuned the supposed holdout. Neither the older large returns nor the new backtest are a genuine untouched test. The saved momentum-overlay data predates later delisting remediation and is not comparable to current reruns.
- The point-in-time membership CSV is filtered at test dates, but 208 names are blacklisted globally, including historical constituents. Some delisted or bankrupt issues have corrupted post-delisting prices; historical exclusions and missing delisting returns can bias results in either direction. No defensible blanket blacklist removal is possible without independently verified price/outcome data.
- Cross-sectional percentile features are ranked over all cached ticker rows available on each date before the test-step point-in-time membership filter. This uses same-date information rather than future prices, but its reference population is not strictly the PIT S&P 500 universe; changing that definition would change model inputs and require a fresh full rerun.
- `TICKER_REMAP` includes merger/acquisition mappings such as ATVI -> MSFT and CELG -> BMY, not only ticker renames. Original ATVI/CELG price CSVs are absent locally, so those historical identities cannot be recovered by merely undoing the remap. Replacing predecessor returns with a successor's series is not a reliable PIT security history; independent predecessor prices and corporate-action identifiers are required before correction.
- An unparseable membership `end_date` is currently warned about and treated as still active by the loader. No such row was independently certified in this audit; a validated vendor membership file is required before interpreting the PIT claim as complete.
- **Material unresolved outcome-availability bias:** `features.jl` stops five rows before a ticker's final recorded price and includes `target` in its finite-row filter. Therefore a test-date stock whose future exit price is absent can disappear before ranking or portfolio selection. This makes eligibility depend on future data availability and can inflate returns, particularly near delistings. The 2020-2026 and lifetime performance figures remain conditional simulations, not investable performance estimates. A defensible fix requires generating signal rows without requiring future targets, then independently sourcing/handling missing delisting payoffs; assigning zero or a synthetic close would fabricate data. No such repair is claimed here.

## 3. Fixes and explicit assumptions

- `target_end_date` comes from the actual stock row used as the label exit. Training requires `target_end_date <= train_end` and filters membership at test time. `target_end_date` is excluded from predictors.
- Benchmarks enter at the next open and exit on the same fifth close; absent or mismatched ETF bars are missing, not zero. Comparisons require common observed dates.
- Missing/unadjustable opens become unavailable values and cannot generate valid trade labels.
- Default net returns deduct 10 bps transaction costs plus 5 bps slippage per unit of one-way portfolio turnover, including the first entry. Costs are fixed assumptions, not estimated fills. Gross fields remain in exports; shorts/SPY overlay is disallowed for net accounting because it lacks an executable cost/borrow model.
- Full validation starts directly at the requested historical test date but still uses prior observations for rolling training. The model seeds are fixed. One-thread execution also limits XGBoost to one training thread.
- Historical validation refuses missing model steps rather than silently shortening the reported period. The older full-main backtest still includes only valid observations; its 2000-2009 sparse years need a separate coverage audit before interpreting any lifetime CAGR.

## 4. New experiments

`robustness_matrix.py` predefines five one-at-a-time historical runs: equal weight/252-trading-session training/10 longs/50-50 RF-XGBoost (reference), 126-session training, five longs, rank-inverse-vol weighting, and 75-25 model blending. It is dry-run by default and runs serially on one thread when `--run` is specified. Only the full reference and 126-session runs completed. The five-long run was stopped at about 3% after a half-hour with a multi-hour ETA to honor the laptop compute constraint; its partial CSV was not archived or used. Rank-weighted and 75-25 full-period runs were not started. The report recalculates 0/15/30 bps cost sensitivities on the **same** stock picks; it does not pretend costs change the signals. Rebalance interval is tied to the five-session target. Short count and ex-post stop-loss cannot be defensibly swept with the disabled overlay.

## 5. Key results

- Complete default main backtest, 2009-04-08 through 2026-04-08 (one long, rank/inverse-vol weighting): 856 valid steps; base long-only **-3.38% net CAGR, 0.14 Sharpe (5% risk-free assumption), 96.65% maximum drawdown**. Linear reference: 3.85% net CAGR, 88.20% maximum drawdown. Main run completed with no per-step errors found in terminal output. The default main configuration is not the ten-long equal-weight validation reference.
- Complete 2020-01-07 through 2026-04-08 validation, 315 steps: 7.47% net CAGR, 46.42% maximum drawdown, 0.23 Sharpe with 5% risk-free rate. On aligned dates with rf=0, GreekFire net Sharpe 0.40 versus SPY 0.73; GreekFire net CAGR 7.47% versus SPY 12.56% without ETF costs.
- Gross-to-net cost drag is material: 12.93% gross CAGR, 7.47% at 15 bps, 2.26% at 30 bps. Mean one-way step turnover including first entry is 65.7%. All 315 SPY windows aligned in the full report.
- Reference configuration rerun reproduced exactly: `cagr=0.07465093446409599`, `sharpe=0.2335267201602762`, `max_dd=0.46420781982280396` in both full validation and archived reference runs.
- A 126-session training window over the same 315 dates produced **-2.65% net CAGR, -0.10 Sharpe at 5% risk-free rate, and 47.81% maximum drawdown**, versus +7.47%, 0.23 and 46.42% for the 252-session reference. This one-at-a-time result shows material training-window sensitivity, not evidence that either configuration will generalize.
- Four bounded 12-step runs (Jan-Mar 2020 stress) were all negative; **not** a full-period robustness result. The full five-run matrix was not completed; generated reporting includes only date-matched completed archives and names unfinished variants.

## 6. Negative or weak findings

Mean cross-sectional IC was -0.0055 over the full main run and -0.0086 over the 2020-2026 validation. The 315-step observed average net minus SPY per step is -0.029%; a descriptive 12-step moving-block bootstrap 95% interval is [-0.328%, 0.254%]. This interval is **not** a multiple-testing-adjusted alpha test. The project has no demonstrated investable edge. Past parameter selection on the same dates disqualifies a claim of independent statistical significance.

## 7. Remaining limitations

Data files are local and excluded from Git. The unresolved future-target availability filter above, 208-name blacklist, historical membership accuracy, adjusted-open reconstruction, corporate actions, delisting payoffs, new-listing coverage, borrow availability, market impact, spread changes, liquidity, and real entry/exit execution are not independently validated. The main backtest's early sparse steps are omitted; its 856-step history is not a complete 2000-2026 simulation. ETF returns in the report exclude ETF transaction costs; portfolio costs are a simple turnover approximation. The 2020-2026 period was already used for research decisions and cannot be made genuinely out of sample after the fact. The top-N sector map is incomplete and groups unknown tickers under a common fallback. No paper-portfolio or live performance is asserted.

## 8. Files changed for this audit

`features.jl`, `backtest.jl`, `data_loader.jl`, `model.jl`, `metrics.jl`, `main.jl`, `validation.jl`, `test_research_validity.jl`, `robustness_matrix.py`, `research_report.py`, `README.md`, `.gitignore`, and this audit. Previous worktree modifications and historical experiment records were retained, not reverted.

## 9. Reproduce

Julia 1.12 with `Project.toml`/`Manifest.toml`, Python 3, cached historical `data/` price and membership CSVs, and Python `matplotlib` for charts are required. Missing source data cannot be reconstructed from this repository alone. Commands from the project root:

```text
julia --project=. test_research_validity.jl
julia --project=. test_delisting_fix.jl
python research_report.py --self-test
python robustness_matrix.py --self-test
python robustness_matrix.py
julia --threads=1 --project=. main.jl validate
python research_report.py
python robustness_matrix.py --run
python research_report.py --steps octave/robustness/reference_validation_holdout_A1_steps.csv
julia --threads=1 --project=. main.jl
```

`python robustness_matrix.py --run --max-steps 12` is a bounded integration check only, not a full result. The full main backtest takes substantially longer. The generated report and figures reside under `octave/`; source market data and these outputs are not part of the versioned research code.