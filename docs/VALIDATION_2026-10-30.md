# Validation record, October 30, 2026

## Scope and status

This record describes the current audit, not the earlier September reruns. Existing uncommitted work was retained. No changes were pushed. Source market inputs were not modified or distributed.

The original PATH entries for Julia and Python were inaccessible Windows app aliases. Checks used the installed Julia 1.12.4 executable and the desktop's bundled Python 3.12 runtime. Python temporary directories had to be redirected into the workspace because the sandbox's default temporary directory denied writes. That rerun passed; the initial permission failure is not a product-test success.

## Executed checks

- `test/runtests.jl` covers synthetic membership intervals, repeated membership, strict invalid end dates, security identity, features, mature-label filtering, model blending, benchmark alignment, gross/net metrics, ties, causal feature invariance, missing sessions and incomplete outcome rejection. It also fits RF/XGBoost/linear models and checks portfolio accounting against direct price arithmetic.
- `examples/synthetic_pipeline.jl`: 8 assertions passed. Signal rows survive absent future labels. This small example does not fit models.
- Full synthetic integration: RF, XGBoost and linear fitting, two walk-forward windows and 29 portfolio/diagnostic assertions passed before the added CSV-loading test. Final suite status is recorded below after rerunning the final source.
- Python recovery tests: 14 tests passed after using a writable temporary directory.
- `research_report.py --self-test`: arithmetic and deterministic bootstrap checks passed.
- `robustness_matrix.py --self-test`: archive-completeness checks passed.
- `risk_overlay_study.py --self-test`: legacy timing and baseline arithmetic checks passed. This does not validate the legacy market series.
- Repository checks cover Python parsing, local Markdown links, required archive artifacts, oversized files and selected high-confidence credential patterns. This is not a forensic scan of Git history.

CI is configured to install the pinned Julia environment and run public tests without market data. A local pass does not claim a remote GitHub Actions run.

## Historical evaluation

Executed `main.jl validate --max-steps=1`. It stopped with `Failed loading ABS: No date column found.` This previously would have been silently skipped. No corrected historical performance was produced or substituted with old exports.

The separate structural inventory reports 1,246 membership intervals, 1,194 distinct symbols, 529 missing symbol files and one malformed header (ABS). Near-500 constituent counts on sampled dates cannot prove historical completeness. Membership provenance, security identity, dividends, delisting payouts and execution assumptions remain unverified. The strict loader now refuses missing coverage as well as malformed files.

Full model-versus-momentum comparisons, corrected historical gross/net results, period breakdowns and sensitivity runs remain blocked by these inputs. Past dates have been repeatedly studied, so even a complete rerun would be exploratory. No new cumulative market-return, drawdown, rolling-Sharpe, IC, decile or feature-importance figure is presented as validated evidence. Existing September figures are preserved with a superseded-methodology notice.

## Measured performance work

The pre-edit feature benchmark on 3,572 deterministic price observations had median 0.1731195 seconds across five warm repetitions. An intermediate revised feature version measured 0.1474424 seconds. Its feature definitions changed, so this is **not** a controlled optimization comparison or a claimed speedup.

For a controlled rank-only benchmark, 30,000 rows across 300 dates use identical tied-rank/clamp mathematics. `tools/benchmark_ranks.jl` asserts equal outputs and times seven warm repetitions on this machine:

| Implementation | Median seconds |
| --- | ---: |
| Repeated whole-table date masks | 0.0019260 |
| SubDataFrame column replacement, rejected approach | 0.0323438 |
| Contiguous sorted date ranges, retained | 0.0008174 |

The retained operation took about 42% of the mask version's time on this fixture. This is a small preprocessing benchmark; it does not establish total training speed, memory improvement or scaling on market data. Model fitting remains a major cost. No claimed profitability influenced the changes.

## Remaining technical concerns

The coarse sector map groups unknown stocks together. Impurity importance can favor correlated predictors and is not inferential evidence. End-of-window drawdowns miss intraperiod losses. The next-open model assumes executable adjusted opens and coherent corporate-action factors; no liquidity/impact model is fitted. Entry and exit fees are deducted from initial-window NAV as a simplified notional fee calculation. Cash earns zero and SPY is an uncosted matched-window comparator. The dormant HMM needs separate numerical repair and tests before use.

Admissions readiness is strongest as a transparent software/research project. The README quickly states the question and implementation and makes the lack of validated historical results visible. A professor can inspect regressions for timing and eligibility errors. Independent input validation and a new frozen evaluation are the most valuable next improvements. The project should not be presented as a proven trading strategy or as an audit performed entirely without assistance.

## Final execution status

- Final `test/runtests.jl`: **112 assertions passed across 13 test sets**, exit code 0.
- Standalone `examples/synthetic_backtest.jl`: **3 CSV-loading assertions and 29 full-pipeline assertions passed**, exit code 0. It wrote `results/synthetic/steps.csv`; the logged model/backtest portion took 31.7983116 seconds including first-call compilation in that process. This is not a throughput benchmark.
- The standalone example initially exposed a missing `Statistics` import that the enclosing test suite had supplied. The import was fixed and the standalone command rerun successfully.
- `examples/synthetic_pipeline.jl`: 8 assertions passed after the signal-eligibility changes.
- Final repository checker passed over 56 publication files, including archived source and the repaired graph-ignore file.
- `git diff --check` reported no whitespace errors; Git emitted line-ending conversion notices.


- Julia `Pkg.resolve()` completed successfully using the installed cache after sandbox-restricted native-artifact checks were rerun with expanded access. No packages were added or removed. Julia confirmed `Manifest current: true`; only the manifest project hash changed.
- Final Python rerun: all 14 recovery tests and all three script self-tests passed.
- The final standalone exporter rerun passed (33 assertions including CSV loading and the observed SPY equity assertion). It generated all diagnostic CSVs under `results/synthetic/diagnostics/`, with no writes to historical outputs. The model/backtest portion in this additional run took 25.8335149 seconds; no speedup is inferred from first-call timings.
- Required `graphify update .` completed after the final source edit: 1,334 nodes, 1,680 edges and 113 communities. Graph outputs remain ignored.
