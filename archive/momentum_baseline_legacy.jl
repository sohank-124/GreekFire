# ARCHIVAL SOURCE: relative includes reflect its original root location. Do not execute.
# momentum_baseline.jl
# =============================================================================
# Pure 20-day momentum baseline on the frozen GreekFire holdout.
#
# This script replicates the walk-forward pipeline from `backtest.jl` EXACTLY
# (same feature engineering, same PIT universe, same long-basket construction
#  with sector cap, same weight scheme) but REPLACES the ML ensemble (RF+XGB)
# with a simple cross-sectional momentum sort based on `ret_20`.
#
# The point of the baseline is to provide a no-machine-learning reference for
# measuring how much of the A1/F1 holdout performance is attributable to the
# ML model vs. a vanilla "buy recent winners, sector-cap, equal-weight" rule.
#
# Pipeline (1:1 with backtest.jl):
#   1. Load all data (DataLoader.load_all_data + load_ticker_data("SPY")).
#   2. Compute Features.calculate_features for every ticker in parallel.
#   3. Stack features into master_df sorted by [date, ticker].
#   4. Cross-sectional percentile rank of ret_20 -> rank_ret_20.
#   5. Save the raw ret_20 into `raw_ret_20` BEFORE standardization
#      (so we can compute Pearson IC vs. target cleanly).
#   6. Cross-sectional percentile rank of every other numeric feature
#      (skips rank_ret_20 and raw_ret_20, just like backtest.jl skips ret_20).
#   7. Build the walk-forward step schedule (TRAIN_WINDOW_DAYS=252, STEP_DAYS=5).
#   8. Pre-cache point-in-time S&P 500 universe per step.
#   9. Walk-forward loop:
#        - filter test rows to PIT active set
#        - sort by momentum signal (rank_ret_20) descending
#        - sector-capped top-10 long basket (sector limit 2, 3 for "Unknown")
#        - equal-weight long portfolio (1/N, matches A1 exactly)
#        - record per-step tuple compatible with Validation.compute_holdout_metrics
#   10. Filter to holdout (>= 2020-01-01) with Validation.filter_holdout.
#   11. Compute full institutional metrics via Validation.compute_holdout_metrics.
#   12. Export CSVs into octave/.
#   13. Recompute A1/F1 metrics from their holdout step CSVs and print a
#       3-column Momentum vs A1 vs F1 comparison.
#   14. Export side-by-side summary and per-step comparison CSVs.
#
# Usage:
#   julia --project=. momentum_baseline.jl
# =============================================================================

# ---------------------------------------------------------------------------
# Module includes (matching main.jl's top-level includes).
# ---------------------------------------------------------------------------
include("data_loader.jl")
include("features.jl")
include("model.jl")
include("backtest.jl")
include("validation.jl")

using DataFrames, Dates, Statistics, CSV, Random, StatsBase, Logging
using Base.Threads  # required for Threads.@threads at top level

# ---------------------------------------------------------------------------
# Frozen constants (mirror backtest.jl / validation.jl).
# ---------------------------------------------------------------------------
const HOLDOUT_START       = Date(2020, 1, 1)
const STEPS_PER_YEAR      = 252.0 / 5.0
const TOP_N_STOCKS        = 10
const TRAIN_WINDOW_DAYS   = 252
const STEP_DAYS           = 5

# ---------------------------------------------------------------------------
# Small CSV-field parsers (A1/F1 step CSVs store lists as ";"-delimited strings).
# ---------------------------------------------------------------------------
_split_csv_field(s::AbstractString) = isempty(s) ? String[] : String.(split(s, ';'))
_split_csv_field(::Missing)         = String[]
_split_weights(s::AbstractString)  = isempty(s) ? Float64[] : [parse(Float64, x) for x in split(s, ';')]
_split_weights(::Missing)          = Float64[]

# ---------------------------------------------------------------------------
# Reconstruct a `compute_holdout_metrics`-compatible result vector from a step
# DataFrame produced by either this script or validation.jl. This lets us
# recompute A1/F1 metrics from their existing CSVs and reuse the exact same
# formulas in `Validation.compute_holdout_metrics`.
# ---------------------------------------------------------------------------
function _results_from_steps_df(steps_df::DataFrame)
    results = Vector{Any}(undef, nrow(steps_df))
    for (i, row) in enumerate(eachrow(steps_df))
        tickers_long = _split_csv_field(row.long_tickers)
        weights      = _split_weights(row.weights)

        # Convert Union{Missing,Float64} to Float64 (or NaN) so downstream
        # isfinite / std / cor calls don't choke on missings.
        safe_f(x) = (x === missing || !isfinite(x)) ? NaN : Float64(x)

        results[i] = (
            date                 = row.date,
            actual_alpha_long    = safe_f(row.actual_alpha_long),
            chosen_alpha         = safe_f(row.chosen_alpha),
            ic                   = safe_f(row.ic),
            rank_ic              = safe_f(row.rank_ic),
            decile_spread        = safe_f(row.decile_spread),
            best_decile_return   = safe_f(row.best_decile),
            worst_decile_return  = safe_f(row.worst_decile),
            n_universe           = Int(row.n_universe),
            tickers_long         = tickers_long,
            inv_vol_weights      = weights,
        )
    end
    return results
end

# ---------------------------------------------------------------------------
# Recompute the full institutional metrics tuple from a step DataFrame,
# exactly as Validation.compute_holdout_metrics would compute it on raw
# results. We use the same underlying function to guarantee apples-to-apples
# comparison.
# ---------------------------------------------------------------------------
function _recompute_metrics_from_steps_df(steps_df::DataFrame, config_label::AbstractString)
    results = _results_from_steps_df(steps_df)
    m = compute_holdout_metrics(results; signal=:actual_alpha_long)
    if m === nothing
        return nothing
    end
    return merge(m, (config = String(config_label),))
end

# ---------------------------------------------------------------------------
# Long-basket construction (sector-capped, equal weight).
# Identical to backtest.jl lines 343-366. Lives at top level so we can call
# it without re-instantiating the Backtest module's internal helpers.
# ---------------------------------------------------------------------------
function _build_long_basket(period_scores::DataFrame)
    long_indices       = Int[]
    long_sector_counts = Dict{String, Int}()
    for (idx, row) in enumerate(eachrow(period_scores))
        length(long_indices) >= TOP_N_STOCKS && break
        sector = get(Backtest.SECTOR_MAP, row.ticker, "Unknown")
        count  = get(long_sector_counts, sector, 0)
        sector_limit = sector == "Unknown" ? 3 : 2
        if count < sector_limit
            push!(long_indices, idx)
            long_sector_counts[sector] = count + 1
        end
    end
    # Fallback: fill remaining slots if sector caps prevented reaching TOP_N_STOCKS
    if length(long_indices) < TOP_N_STOCKS
        existing = Set(period_scores.ticker[long_indices])
        for (idx, row) in enumerate(eachrow(period_scores))
            row.ticker in existing && continue
            push!(long_indices, idx)
            push!(existing, row.ticker)
            length(long_indices) >= TOP_N_STOCKS && break
        end
    end
    return long_indices
end

# ---------------------------------------------------------------------------
# SPY per-step return (open[t+1] -> close[t+FORECAST_HORIZON]).
# Identical to backtest.jl lines 542-557. Hand-rolled here so we don't have
# to call into the full Backtest.run_walk_forward pipeline.
# ---------------------------------------------------------------------------
function _spy_return_for_step(spy_data::DataFrame, test_start::Date)
    spy_exec_idx = searchsortedfirst(spy_data.date, test_start)
    if spy_exec_idx <= nrow(spy_data) && spy_data.date[spy_exec_idx] == test_start
        spy_exec_idx += 1
        spy_exit_idx  = spy_exec_idx + Features.FORECAST_HORIZON - 1
        if spy_exec_idx <= nrow(spy_data) && spy_exit_idx <= nrow(spy_data)
            entry_px = spy_data.open[spy_exec_idx]
            exit_px  = spy_data.close[spy_exit_idx]
            return (!ismissing(entry_px) && !ismissing(exit_px) &&
                    isfinite(entry_px) && isfinite(exit_px) && entry_px > 0.0) ?
                Float64(exit_px) / Float64(entry_px) - 1.0 : 0.0
        end
    end
    return 0.0
end

# ---------------------------------------------------------------------------
# Cross-sectional diagnostics: Pearson IC, Spearman rank IC, decile table.
# Operates on the *test* period_scores (one rebalance date), exactly like
# backtest.jl lines 299-331.
# ---------------------------------------------------------------------------
function _cross_sectional_diagnostics(period_scores::DataFrame)
    mom_vals    = Float64.(period_scores.raw_ret_20)
    actual_vals = Float64.(period_scores.avg_actual)

    ic      = NaN
    rank_ic = NaN
    if length(mom_vals) >= 5 && std(mom_vals) > 1e-12 && std(actual_vals) > 1e-12
        ic      = cor(mom_vals, actual_vals)
        rank_ic = cor(tiedrank(mom_vals), tiedrank(actual_vals))
    end

    sorted_ps = sort(period_scores, :mom_signal, rev = true)
    n_dec     = nrow(sorted_ps)
    decs      = Float64[]
    for d in 1:10
        lo = max(1, (d - 1) * n_dec Ã· 10 + 1)
        hi = min(n_dec, d * n_dec Ã· 10)
        if lo <= hi
            push!(decs, mean(sorted_ps[lo:hi, :avg_actual]))
        else
            push!(decs, NaN)
        end
    end
    return (ic = ic, rank_ic = rank_ic,
            decs = decs,
            best_decile_return = decs[1],
            worst_decile_return = decs[end],
            decile_spread = decs[1] - decs[end])
end

# ---------------------------------------------------------------------------
# Build a single momentum-baseline step result tuple. Schema matches the
# tuple Backtest.run_walk_forward produces for the ML runs, so we can pass
# the result list straight into Validation.filter_holdout +
# Validation.compute_holdout_metrics.
# ---------------------------------------------------------------------------
function _build_step_result(step, period_scores::DataFrame, long_indices::Vector{Int},
                              spy_data::DataFrame)
    long_candidates   = period_scores[long_indices, :]
    n_long            = length(long_indices)
    weights           = fill(1.0 / n_long, n_long)

    actual_vals_long  = Float64.(skipmissing(long_candidates.avg_actual))
    actual_long       = isempty(actual_vals_long) ? 0.0 : mean(actual_vals_long)
    if !isfinite(actual_long)
        actual_long = 0.0
    end

    diags = _cross_sectional_diagnostics(period_scores)

    spy_ret = _spy_return_for_step(spy_data, step.test_start)

    return (
        date                          = step.test_start,
        train_start                   = step.train_start,
        train_end                     = step.train_end,
        test_start                    = step.test_start,
        test_end                      = step.test_end,
        tickers_long                  = long_candidates.ticker,
        tickers_high_pred             = long_candidates.ticker,
        tickers_short                 = String[],
        predicted_alpha               = 0.0,
        actual_alpha_long             = actual_long,
        actual_alpha_high_pred        = actual_long,
        actual_alpha_short            = 0.0,
        chosen_alpha                  = actual_long,
        regime_signal                 = :bull,
        actual_equal_weight_long      = actual_long,
        actual_equal_weight_short     = 0.0,
        best_decile_return            = diags.best_decile_return,
        worst_decile_return           = diags.worst_decile_return,
        decile_spread                 = diags.decile_spread,
        period_scores_copy            = copy(period_scores),
        long_indices_copy             = copy(long_indices),
        short_indices_copy            = Int[],
        inv_vol_weights               = weights,
        actual_long_equal_weight_long = actual_long,
        ic                            = diags.ic,
        rank_ic                       = diags.rank_ic,
        spy_ret                       = spy_ret,
        long_preds_copy               = Float64.(long_candidates.mom_signal),
        short_preds_copy              = Float64[],
        n_universe                    = nrow(period_scores),
    )
end

# ---------------------------------------------------------------------------
# Export per-step + annual + equity CSVs for the momentum baseline.
# Schema intentionally matches `validation_holdout_A1_steps.csv` so all three
# configs can be diffed in any tool.
# ---------------------------------------------------------------------------
function export_csvs(metrics, holdout_results)
    out_dir = joinpath(pwd(), "octave")
    mkpath(out_dir)

    # ---- per-step CSV ----------------------------------------------------
    step_rows = NamedTuple[]
    for r in holdout_results
        r === nothing && continue
        push!(step_rows, (
            date             = r.date,
            actual_alpha_long = r.actual_alpha_long,
            chosen_alpha     = r.chosen_alpha,
            ic               = r.ic,
            rank_ic          = r.rank_ic,
            decile_spread    = r.decile_spread,
            best_decile      = r.best_decile_return,
            worst_decile     = r.worst_decile_return,
            n_universe       = r.n_universe,
            n_long           = length(r.tickers_long),
            long_tickers     = join(string.(r.tickers_long), ";"),
            weights          = join([round(w; digits=6) for w in r.inv_vol_weights], ";"),
        ))
    end
    steps_path = joinpath(out_dir, "momentum_baseline_steps.csv")
    CSV.write(steps_path, DataFrame(step_rows))
    println("[momentum] wrote $steps_path")

    # ---- annual returns CSV ---------------------------------------------
    annual_path = joinpath(out_dir, "momentum_baseline_annual.csv")
    annual_df = DataFrame(
        year          = sort(collect(keys(metrics.annual_returns))),
        annual_return = [metrics.annual_returns[y] for y in sort(collect(keys(metrics.annual_returns)))],
    )
    CSV.write(annual_path, annual_df)
    println("[momentum] wrote $annual_path")

    # ---- equity curve CSV -----------------------------------------------
    equity_path = joinpath(out_dir, "momentum_baseline_equity.csv")
    equity_df = DataFrame(
        step_index = 0:(length(metrics.equity_curve) - 1),
        date       = vcat([metrics.dates[1]], metrics.dates),
        equity     = metrics.equity_curve,
    )
    CSV.write(equity_path, equity_df)
    println("[momentum] wrote $equity_path")
end

# ---------------------------------------------------------------------------
# Per-step comparison CSV: inner-join momentum / A1 / F1 on date, then emit
# the actual_alpha_long values side-by-side so any downstream tool (or
# eyeball) can compare returns path-by-path.
# ---------------------------------------------------------------------------
function _per_step_comparison_df(mom_steps::DataFrame, a1_steps::DataFrame, f1_steps::DataFrame)
    sel_cols(c) = select(c, :date, :actual_alpha_long => Symbol(c, "_alpha"))
    m1 = rename(select(mom_steps, :date, :actual_alpha_long), :actual_alpha_long => :momentum_alpha)
    a1 = rename(select(a1_steps,  :date, :actual_alpha_long), :actual_alpha_long => :A1_alpha)
    f1 = rename(select(f1_steps,  :date, :actual_alpha_long), :actual_alpha_long => :F1_alpha)

    joined = innerjoin(m1, a1, on = :date)
    joined = innerjoin(joined, f1, on = :date)
    sort!(joined, :date)

    # Cumulative equity (starting at 1.0) for each strategy at each step.
    n = nrow(joined)
    eq_mom = cumprod(1.0 .+ Float64.(joined.momentum_alpha))
    eq_a1  = cumprod(1.0 .+ Float64.(joined.A1_alpha))
    eq_f1  = cumprod(1.0 .+ Float64.(joined.F1_alpha))
    joined[!, :momentum_equity] = eq_mom
    joined[!, :A1_equity]       = eq_a1
    joined[!, :F1_equity]       = eq_f1
    joined[!, :momentum_vs_A1]  = Float64.(joined.momentum_alpha) .- Float64.(joined.A1_alpha)
    joined[!, :momentum_vs_F1]  = Float64.(joined.momentum_alpha) .- Float64.(joined.F1_alpha)
    return joined
end

# ---------------------------------------------------------------------------
# Side-by-side 3-config summary CSV. Mirrors `validation_holdout_summary.csv`
# so it can be diffed in any spreadsheet.
# ---------------------------------------------------------------------------
function _summary_df(mom::NamedTuple, a1::NamedTuple, f1::NamedTuple)
    cfgs  = [mom, a1, f1]
    names_ = [mom.config, a1.config, f1.config]
    all_years = sort(union([collect(keys(m.annual_returns)) for m in cfgs]...))

    rows = NamedTuple[]
    for (i, m) in enumerate(cfgs)
        push!(rows, (
            config            = String(names_[i]),
            n_steps           = Int(m.n_steps),
            date_start        = m.date_start,
            date_end          = m.date_end,
            total_return      = Float64(m.total_return),
            cagr              = Float64(m.cagr),
            sharpe            = Float64(m.sharpe),
            max_dd            = Float64(m.max_dd),
            calmar            = Float64(m.calmar),
            win_rate          = Float64(m.win_rate),
            avg_excess        = Float64(m.avg_excess),
            mean_turnover     = Float64(m.mean_turnover),
            turnover_max      = Float64(m.turnover_max),
            trade_count       = Int(m.trade_count),
            mean_ic           = Float64(m.mean_ic),
            mean_rank_ic      = Float64(m.mean_rank_ic),
            mean_dec_spread   = Float64(m.mean_dec_spread),
            mean_best_decile  = Float64(m.mean_best_decile),
            mean_worst_decile = Float64(m.mean_worst_decile),
        ))
    end
    df = DataFrame(rows)
    for y in all_years
        col = Symbol(string("annual_", y))
        df[!, col] = [get(m.annual_returns, y, NaN) for m in cfgs]
    end
    return df
end

# ---------------------------------------------------------------------------
# Pretty-print a 3-column comparison table. Custom-ordered (Momentum, A1, F1)
# so the baseline is shown first, then the two ML configs.
# ---------------------------------------------------------------------------
function print_comparison(mom::NamedTuple, a1::NamedTuple, f1::NamedTuple)
    order  = [mom, a1, f1]
    labels = [mom.config, a1.config, f1.config]

    println()
    println(repeat("=", 110))
    println("  HOLDOUT COMPARISON  -  Momentum (20d, sector-cap, equal-weight)  vs  A1 (ML equal)  vs  F1 (ML rank+inv-vol)")
    println("  Holdout window: $(HOLDOUT_START)  ->  latest available data")
    println("  Same walk-forward pipeline + PIT universe + sector cap as the ML configs.  Only the SIGNAL differs.")
    println(repeat("=", 110))

    if any(m -> m === nothing, order)
        @warn "One or more configs produced no metrics.  Table may be incomplete."
    end

    # ---- core metrics block ---------------------------------------------
    println()
    print(rpad("Metric", 32))
    for lbl in labels
        print("  ", rpad(lbl, 24))
    end
    println()
    print(rpad(repeat("-", 30), 32))
    for _ in labels
        print("  ", rpad(repeat("-", 22), 24))
    end
    println()

    rows = [
        ("Steps (n)"            , m -> m === nothing ? "n/a" : string(m.n_steps)),
        ("Date range"           , m -> m === nothing ? "n/a" : "$(m.date_start) .. $(m.date_end)"),
        ("Total return"         , m -> m === nothing ? "n/a" : _pctv(m.total_return)),
        ("CAGR"                 , m -> m === nothing ? "n/a" : _pctv(m.cagr)),
        ("Sharpe (rf=5%)"       , m -> m === nothing ? "n/a" : _fnum(m.sharpe)),
        ("Max Drawdown"         , m -> m === nothing ? "n/a" : _pctv(m.max_dd)),
        ("Calmar"               , m -> m === nothing ? "n/a" : _fnum(m.calmar)),
        ("Win rate"             , m -> m === nothing ? "n/a" : _pctv(m.win_rate)),
        ("Avg excess / step"    , m -> m === nothing ? "n/a" : _pctv(m.avg_excess)),
        ("Mean IC"              , m -> m === nothing ? "n/a" : _fnum(m.mean_ic, 4)),
        ("Mean Rank IC"         , m -> m === nothing ? "n/a" : _fnum(m.mean_rank_ic, 4)),
        ("Decile spread (mean)" , m -> m === nothing ? "n/a" : _pctv(m.mean_dec_spread)),
        ("Best decile (mean)"   , m -> m === nothing ? "n/a" : _pctv(m.mean_best_decile)),
        ("Worst decile (mean)"  , m -> m === nothing ? "n/a" : _pctv(m.mean_worst_decile)),
        ("Mean turnover / step" , m -> m === nothing ? "n/a" : _fnum(m.mean_turnover * 100) * "%"),
        ("Turnover p25"         , m -> m === nothing ? "n/a" : _fnum(m.turnover_p25 * 100)  * "%"),
        ("Turnover max"         , m -> m === nothing ? "n/a" : _fnum(m.turnover_max * 100)  * "%"),
        ("Trade (rebalance) ct" , m -> m === nothing ? "n/a" : string(m.trade_count)),
    ]
    for (label, f) in rows
        print(rpad(label, 32))
        for m in order
            print("  ", rpad(f(m), 24))
        end
        println()
    end

    # ---- annual returns block -------------------------------------------
    println()
    println("  Annual returns (compounded within calendar year, holdout only):")
    valid_cfgs = [m for m in order if m !== nothing]
    if !isempty(valid_cfgs)
        all_years = sort(union([collect(keys(m.annual_returns)) for m in valid_cfgs]...))
        print(rpad("Year", 32))
        for lbl in labels
            print("  ", rpad(lbl, 24))
        end
        println()
        print(rpad(repeat("-", 30), 32))
        for _ in labels
            print("  ", rpad(repeat("-", 22), 24))
        end
        println()
        for y in all_years
            print(rpad(string(y), 32))
            for m in order
                v = (m === nothing) ? NaN : get(m.annual_returns, y, NaN)
                print("  ", rpad(_pctv(v), 24))
            end
            println()
        end
    end
    println()
    println(repeat("=", 110))
    println()
end

# ---------------------------------------------------------------------------
# Generate the two cross-config CSVs.
# ---------------------------------------------------------------------------
function generate_comparison_csvs(mom::NamedTuple, a1::NamedTuple, f1::NamedTuple,
                                   mom_steps_df::DataFrame,
                                   a1_steps_df::DataFrame,
                                   f1_steps_df::DataFrame)
    out_dir = joinpath(pwd(), "octave")
    mkpath(out_dir)

    # ---- summary side-by-side -------------------------------------------
    summary_df = _summary_df(mom, a1, f1)
    summary_path = joinpath(out_dir, "momentum_baseline_summary.csv")
    CSV.write(summary_path, summary_df)
    println("[momentum] wrote $summary_path  (cols = $(length(names(summary_df))))")

    # ---- per-step return comparison -------------------------------------
    per_step_df = _per_step_comparison_df(mom_steps_df, a1_steps_df, f1_steps_df)
    per_step_path = joinpath(out_dir, "momentum_vs_ml_comparison.csv")
    CSV.write(per_step_path, per_step_df)
    println("[momentum] wrote $per_step_path  (rows = $(nrow(per_step_df)))")
end

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
function main()
    Random.seed!(42)
    disable_logging(Logging.Info)
    println("[momentum] Info-level logging suppressed.")
    println("[momentum] Running on $(Threads.nthreads()) threads")
    println("[momentum] HOLDOUT_START = $HOLDOUT_START")
    println("[momentum] TOP_N_STOCKS = $TOP_N_STOCKS  |  TRAIN_WINDOW_DAYS = $TRAIN_WINDOW_DAYS  |  STEP_DAYS = $STEP_DAYS")
    println("=" ^ 80)

    # -------------------------------------------------------------------
    # 1. Load data (identical to validation.jl)
    # -------------------------------------------------------------------
    all_stock_data = DataLoader.load_all_data()
    spy_data       = DataLoader.load_ticker_data("SPY")
    if spy_data === nothing
        error("Failed to load SPY benchmark data. Cannot proceed.")
    end
    println("[momentum] data loaded: $(length(all_stock_data)) tickers")

    # -------------------------------------------------------------------
    # 2. Compute features for all tickers (identical to backtest.jl L109-120)
    # -------------------------------------------------------------------
    println("[momentum] Pre-computing feature matrices...")
    tickers = sort(collect(keys(all_stock_data)))
    all_features_vec = Vector{Pair{String, DataFrame}}(undef, length(tickers))
    Threads.@threads for i in 1:length(tickers)
        ticker = tickers[i]
        all_features_vec[i] = ticker => Features.calculate_features(all_stock_data[ticker], spy_data)
    end

    # -------------------------------------------------------------------
    # 3. Build master_df (identical to backtest.jl L122-135)
    # -------------------------------------------------------------------
    println("[momentum] Building master DataFrame...")
    all_dfs = DataFrame[]
    for (ticker, df) in all_features_vec
        if nrow(df) > 0
            df_copy = copy(df)
            df_copy.ticker .= ticker
            push!(all_dfs, df_copy)
        end
    end
    if isempty(all_dfs)
        error("No feature rows were generated. Check the feature pipeline and data loader.")
    end
    master_df = vcat(all_dfs...)
    sort!(master_df, [:date, :ticker])
    all_dates = sort(unique(master_df.date))
    println("[momentum] master_df rows = $(nrow(master_df))  |  trading dates = $(length(all_dates))")

    # -------------------------------------------------------------------
    # 4. Cross-sectional rank of ret_20 -> rank_ret_20 (backtest.jl L138-144)
    # -------------------------------------------------------------------
    println("[momentum] Computing cross-sectional rank_ret_20...")
    master_df[!, :rank_ret_20] = fill(0.5, nrow(master_df))
    for d in all_dates
        mask = master_df.date .== d
        if any(mask)
            master_df[mask, :rank_ret_20] = Backtest.percentile_rank(master_df[mask, :ret_20])
        end
    end

    # -------------------------------------------------------------------
    # 5. Save raw ret_20 BEFORE standardization (for Pearson IC).
    # -------------------------------------------------------------------
    master_df[!, :raw_ret_20] = copy(master_df.ret_20)

    # -------------------------------------------------------------------
    # 6. Cross-sectional percentile rank of every numeric feature,
    #    except: date, target, ticker, raw_ret_20 (saved raw).
    #    The loop ALSO skips rank_ret_20 (matches backtest.jl behavior).
    # -------------------------------------------------------------------
    println("[momentum] Standardizing features cross-sectionally (percentile rank + clamp)...")
    feature_cols = String[]
    for col in names(master_df)
        if (eltype(master_df[!, col]) <: Number) && !(col in Set(["date", "target", "ticker", "raw_ret_20"]))
            push!(feature_cols, String(col))
        end
    end

    for sub_df in groupby(master_df, :date)
        n = nrow(sub_df)
        if n == 1
            for col in feature_cols
                col == "rank_ret_20" && continue
                sub_df[:, Symbol(col)] = fill(0.5, n)
            end
        else
            for col in feature_cols
                col == "rank_ret_20" && continue
                sub_df[:, Symbol(col)] = Backtest.percentile_rank(sub_df[:, Symbol(col)])
            end
        end
    end

    # -------------------------------------------------------------------
    # 7. Build walk-forward step schedule (identical to backtest.jl L176-194)
    # -------------------------------------------------------------------
    println("[momentum] Building walk-forward step schedule...")
    trading_dates = all_dates
    start_date    = DataLoader.START_DATE
    steps = []
    curr_idx = searchsortedfirst(trading_dates, start_date + Day(TRAIN_WINDOW_DAYS))
    while curr_idx + STEP_DAYS + 1 <= length(trading_dates)
        train_end   = trading_dates[curr_idx]
        test_start  = trading_dates[curr_idx + 1]
        test_end    = trading_dates[min(curr_idx + STEP_DAYS, length(trading_dates))]
        train_start = trading_dates[max(1, curr_idx - TRAIN_WINDOW_DAYS)]
        push!(steps, (
            train_start = train_start,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end,
        ))
        curr_idx += STEP_DAYS
    end
    println("[momentum] total walk-forward steps: $(length(steps))")

    # -------------------------------------------------------------------
    # 8. Pre-cache PIT universe (identical to backtest.jl L226-234)
    # -------------------------------------------------------------------
    println("[momentum] Pre-caching S&P 500 point-in-time universe...")
    pit_cache = Dict{Date, Set{String}}()
    for step in steps
        date = step.test_start
        if !haskey(pit_cache, date)
            active = DataLoader.get_sp500_constituents(date)
            pit_cache[date] = Set(DataLoader.resolve_ticker(t) for t in active)
        end
    end
    println("[momentum] PIT cache built for $(length(pit_cache)) distinct rebalance dates")

    # -------------------------------------------------------------------
    # 9. Walk-forward loop â€” NO ML, just momentum sort.
    # -------------------------------------------------------------------
    println("[momentum] Running walk-forward momentum loop...")
    results = Vector{Any}(undef, length(steps))
    n_skipped = 0
    for i in 1:length(steps)
        step       = steps[i]
        active_set = pit_cache[step.test_start]

        # Test rows on this rebalance date, filtered to the PIT active set.
        combined_test = master_df[
            (master_df.date .== step.test_start) .&
            (master_df.ticker .âˆˆ Ref(active_set)), :]

        if nrow(combined_test) == 0
            results[i] = nothing
            n_skipped += 1
            continue
        end

        # Build the per-step frame. The "momentum signal" is rank_ret_20
        # (cross-sectional percentile rank of 20-day return). raw_ret_20
        # is preserved for Pearson IC computation.
        period_scores = DataFrame(
            ticker      = combined_test.ticker,
            mom_signal  = combined_test.rank_ret_20,
            raw_ret_20  = combined_test.raw_ret_20,
            avg_actual  = combined_test.target,
            idio_vol_20 = combined_test.idio_vol_20,
        )

        # Drop NaN/missing defensively.
        valid_mask = .!isnan.(period_scores.mom_signal) .& .!isnan.(period_scores.avg_actual) .& .!isnan.(period_scores.raw_ret_20)
        period_scores = period_scores[valid_mask, :]

        if nrow(period_scores) == 0
            results[i] = nothing
            n_skipped += 1
            continue
        end

        # Sort by momentum signal descending â€” top 10 become the long basket.
        sort!(period_scores, :mom_signal, rev = true)
        long_indices   = _build_long_basket(period_scores)
        results[i]     = _build_step_result(step, period_scores, long_indices, spy_data)

        if i % 100 == 0
            println("  step $i / $(length(steps)) complete")
        end
    end
    println("[momentum] walk-forward loop done.  results slots = $(length(results))  |  skipped = $n_skipped")

    # -------------------------------------------------------------------
    # 10. Filter to holdout (>= 2020-01-01) using Validation.filter_holdout.
    # -------------------------------------------------------------------
    holdout = filter_holdout(results)
    println("[momentum] holdout steps: $(length(holdout))  ($(holdout[1].date) .. $(holdout[end].date))")

    # -------------------------------------------------------------------
    # 11. Compute full institutional metrics.
    # -------------------------------------------------------------------
    m = compute_holdout_metrics(holdout; signal=:actual_alpha_long)
    if m === nothing
        error("Momentum holdout produced no finite-alpha steps.  Aborting.")
    end
    m = merge(m, (config = "MOMENTUM",))
    println("[momentum] metrics computed: $(m.n_steps) steps, CAGR = $(round(m.cagr * 100; digits=2))%, Sharpe = $(round(m.sharpe; digits=2))")

    # -------------------------------------------------------------------
    # 12. Export the per-config CSVs (steps / annual / equity).
    # -------------------------------------------------------------------
    export_csvs(m, holdout)

    # -------------------------------------------------------------------
    # 13. Load A1/F1 holdout step CSVs and recompute their metrics from
    #     scratch using the SAME formulas. This guarantees apples-to-apples.
    # -------------------------------------------------------------------
    a1_steps_path = "octave/validation_holdout_A1_steps.csv"
    f1_steps_path = "octave/validation_holdout_F1_steps.csv"
    if !isfile(a1_steps_path) || !isfile(f1_steps_path)
        @warn "A1/F1 holdout step CSVs not found at $a1_steps_path / $f1_steps_path.  " *
              "Comparison table will be partial.  Re-run validation.jl to populate them."
        a1 = nothing
        f1 = nothing
        a1_steps_df = DataFrame()
        f1_steps_df = DataFrame()
    else
        a1_steps_df = CSV.read(a1_steps_path, DataFrame)
        f1_steps_df = CSV.read(f1_steps_path, DataFrame)
        a1 = _recompute_metrics_from_steps_df(a1_steps_df, "A1")
        f1 = _recompute_metrics_from_steps_df(f1_steps_df, "F1")
    end

    # -------------------------------------------------------------------
    # 14. Print the 3-column comparison.
    # -------------------------------------------------------------------
    if a1 !== nothing && f1 !== nothing
        print_comparison(m, a1, f1)
    else
        # Only have the momentum metrics â€” print a single-column summary.
        print_comparison(m, m, m)  # ugly but functional: shows the line twice
        @warn "Could not load A1/F1 CSVs â€” table shown for momentum only."
    end

    # -------------------------------------------------------------------
    # 15. Export the two cross-config CSVs.
    # -------------------------------------------------------------------
    if a1 !== nothing && f1 !== nothing
        # Use a1/f1 metrics + a1_steps/f1_steps as DataFrames for the per-step join.
        generate_comparison_csvs(m, a1, f1,
                                 CSV.read("octave/momentum_baseline_steps.csv", DataFrame),
                                 a1_steps_df, f1_steps_df)
    end

    println("\nDone! CSVs written to octave/")
    return m
end

# Entrypoint: run when invoked as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
