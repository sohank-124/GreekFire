# Research notes

The experiment log is an exploratory record. Its entries are preserved, including failed runs and obsolete terminology. The current working log contains more than the approximately 77 runs mentioned in the project brief. Blank notes and absent code/data hashes limit reconstruction of individual experiments. Interpretations below are separated from recorded observations.

## Regime switching and the HMM

**Hypothesis inferred from the implementation:** volatility states might help switch between long, short and cash exposure. `hmm.jl` implements a two-state Gaussian model; `regime.jl` labels the higher-volatility state as bear and smooths state probabilities before switching. Volatility states do not themselves predict the sign of returns.

Run 66 records a 126-step HMM training window, five longs and five shorts, 10.12% CAGR and 24.3% drawdown. Run 67 records HMM disabled, changed feature processing, 20.15% CAGR and 55.91% drawdown. These are unvalidated simulated summaries, not evidence that disabling HMM caused either change. Several settings and the data interpretation changed together.

The next iterations left HMM disabled. Source inspection also found that posterior windows reuse observations while feeding back a previous posterior, which can count evidence repeatedly. Baum-Welch transition updates mix emissions updated during the same iteration with earlier forward/backward probabilities. The implementation accepts arbitrary state counts while internal routines hardcode two. These issues need dedicated numerical tests before reuse. The short-loss cap is an ex-post return clip, not an executed stop order. Main no longer loads either HMM module. The defensible lesson is to test a regime detector separately before attributing portfolio changes to it.

## High returns with weak ranking

**Hypothesis inferred from the model and portfolio design:** RF/XGBoost rankings might identify stocks with higher subsequent five-session returns. Run 77 records ten longs, 84.2% CAGR, Sharpe 0.66, 44.92% drawdown, mean Pearson IC 0.0335 and mean rank IC -0.0049. Run 78 records 52.93% CAGR and the same reported rank IC. Its notes do not identify the intervention, so the cause of the change cannot be reconstructed. Run 79 records 166.26% CAGR alongside 96.41% drawdown.

The summaries are not algebraically impossible. Compounding, outliers, volatility and a selected basket can differ greatly from whole-universe rank correlation. But a large CAGR with near-zero rank IC is a reason to examine individual trades and data. Summary statistics cannot establish which effect caused Run 77's return.

The subsequent audit found acquisition-to-buyer remaps, future-outcome eligibility filtering and misleading metric labels in the code. Earlier audit notes also report suspect post-removal price series. Without that run's frozen inputs and holdings we cannot assign numerical responsibility to any one bug. Corrected-v2 preserves security identity, keeps signals independent of future targets, uses a common market calendar and refuses missing outcomes. The lesson is that portfolio performance and predictive ranking require separate checks; neither substitutes for input validation.

## Costs, baseline comparisons and training-window sensitivity

**Hypothesis represented by the existing fixed sensitivity runner:** results should persist under reasonable training-window changes and trading frictions. The September 30 archive reports a ten-long reference at 12.93% gross CAGR versus 7.47% after its old 15-bps turnover approximation. Its 126-session training variant reports -2.65% net CAGR versus +7.47% for 252 sessions. Aligned SPY reports 12.56% CAGR without ETF costs. These figures are historical, unvalidated and predate corrected-v2. This audit did not regenerate them.

The archive says only two full sensitivity variants finished; it does not justify choosing the better training window. Its old turnover model charges changes between target weights even though the engine exits at the close and re-enters at the next open. Corrected-v2 charges both legs, including repeated names, uses a zero risk-free default consistently and shares a metric implementation between main and validation.

The next evaluation must run the same corrected pipeline for ensemble and momentum, preserve configurations separately and compare the same dates. Five predefined sensitivities remain available, but malformed/missing market inputs block a defensible full rerun. Synthetic integration checks establish accounting behavior, not robustness of a market prediction. The lesson is to freeze comparison rules before inspecting outcomes and report incomplete experiments explicitly.

## Application presentation

A short admissions review should make the question, engineering pipeline and honest status easy to find. The README now leads with those points and separates historical observations from executed software checks. A technical reviewer can inspect the timing convention, synthetic assertions and coverage failure directly. Strong evidence is the modular Julia pipeline and regression testing of subtle evaluation errors. Remaining weaknesses include missing input lineage, a coarse sector map, no untouched test period and the need for independent numerical review of the archived HMM.

Suggested Common App description (135 characters):

Built a Julia stock-research engine with RF/XGBoost models, threaded features and walk-forward backtests; documented evaluation limits.
