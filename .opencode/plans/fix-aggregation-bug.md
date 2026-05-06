# Plan: Fix GreekFire Backtest Aggregation Bug

## Diagnosis

### Current State (VERIFIED from CSV data)
- 1190 result entries (one per step) — count is correct
- Mean actual_alpha = 0.006172 (0.617% per 5-day step)
- Win rate = 60.6%, equity_curve[end] = ~1513x, CAGR = 36%
- SPY benchmark = 4.9x, CAGR = 7%
- **36% CAGR is unrealistic → bug confirmed**

### Root Cause: overlapping forward returns inflate alpha

The `target` in features.jl is the 5-day FORWARD excess return from each day. The test window spans 5 calendar days (~3-5 trading days). When backtest.jl groups by ticker and **averages `target` across all days in the test window**, it's averaging overlapping 5-day forward returns:

- Day 1 target: return from day 1 to day 6
- Day 2 target: return from day 2 to day 7
- Day 3 target: return from day 3 to day 8

These overlap. When compounded step-by-step, consecutive steps also overlap (step i's day-3 target covers returns through step i+1's day-1). This overlapping compounding inflates the measured alpha.

### The fix: use first-day-only targets for actual returns

For each top stock, take the `target` from the **first trading day** in the test window only. This gives a clean, non-overlapping 5-day forward return that aligns perfectly with step boundaries (day d → day d+5 = exactly one step).

Predictions can still be averaged across days (that's fine for ranking). But actual returns must be non-overlapping.

## Changes

### 1. backtest.jl — Replace lines 92-129

**Before:**
```julia
test_predictions = DataFrame(
    ticker = combined_test.ticker,
    pred_ensemble = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
    actual = combined_test.target
)
stock_ranks = groupby(test_predictions, :ticker)
period_scores = combine(stock_ranks, :pred_ensemble => mean => :avg_pred, :actual => mean => :avg_actual)
dropmissing!(period_scores, :avg_pred)
filter!(row -> !isnan(row.avg_pred), period_scores)
if nrow(period_scores) == 0
    results[i] = nothing
    next!(p)
    continue
end
sort!(period_scores, :avg_pred, rev=true)
top_n = min(TOP_N_STOCKS, nrow(period_scores))
best_stocks = period_scores[1:top_n, :]
actual_portfolio_return = mean(skipmissing(best_stocks.avg_actual))
if isnan(actual_portfolio_return)
    results[i] = nothing
    next!(p)
    continue
end
predicted_portfolio_return = mean(best_stocks.avg_pred)
results[i] = (
    date = step.test_start,
    tickers = best_stocks.ticker,
    predicted_alpha = predicted_portfolio_return,
    actual_alpha = actual_portfolio_return
)
```

**After:**
```julia
test_predictions = DataFrame(
    ticker = combined_test.ticker,
    date = combined_test.date,
    pred_ensemble = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
    actual = combined_test.target
)

stock_ranks = groupby(test_predictions, :ticker)
period_scores = combine(stock_ranks, :pred_ensemble => mean => :avg_pred)

dropmissing!(period_scores, :avg_pred)
filter!(row -> !isnan(row.avg_pred), period_scores)

if nrow(period_scores) == 0
    results[i] = nothing
    next!(p)
    continue
end

sort!(period_scores, :avg_pred, rev=true)
top_n = min(TOP_N_STOCKS, nrow(period_scores))
best_stocks = period_scores[1:top_n, :]

first_date = minimum(combined_test.date)
first_day_df = combined_test[combined_test.date .== first_date, :]
actual_vals = Float64[]
for t in best_stocks.ticker
    rows = first_day_df[first_day_df.ticker .== t, :]
    if nrow(rows) > 0
        val = rows.target[1]
        if isfinite(val)
            push!(actual_vals, val)
        end
    end
end

if isempty(actual_vals)
    results[i] = nothing
    next!(p)
    continue
end

actual_portfolio_return = mean(actual_vals)

if isnan(actual_portfolio_return)
    results[i] = nothing
    next!(p)
    continue
end

predicted_portfolio_return = mean(best_stocks.avg_pred)

results[i] = (
    date = step.test_start,
    tickers = best_stocks.ticker,
    predicted_alpha = predicted_portfolio_return,
    actual_alpha = actual_portfolio_return
)
```

### 2. metrics.jl — Verify steps_per_year

`steps_per_year = 252.0 / 5.0 = 50.4` — already correct at line 28. No change needed.

### 3. Remove debug prints from metrics.jl

Lines 31-34 have debug println statements that should be removed:
```julia
println("equity end: ", equity_curve[end])
println("num alphas: ", length(alphas))
println("years: ", years)
println("steps_per_year: ", steps_per_year)
```

### Expected Impact

| Metric | Before | After (expected) |
|--------|--------|-------------------|
| Mean alpha/step | 0.617% | 0.1-0.3% |
| equity_curve[end] | 1513x | 2-10x |
| CAGR | 36% | 3-10% |
| Win rate | 60.6% | ~55-60% (similar) |
