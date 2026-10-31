# validation.jl
# =============================================================================
# Historical validation harness for the GreekFire walk-forward engine.
#
# Historical reporting boundary: default 2020-01-01 through latest data.
# This period has been inspected and tuned in prior research and is not an
# untouched out-of-sample evaluation. Models continue rolling prior-only training.
#
# The runner evaluates one chosen fixed configuration at a time using the
# same walk-forward model/portfolio logic and reports only the selected dates.
#
# Anti-leakage: the first training end precedes its signal date; the engine
# admits a training row only if its own target_end_date <= train_end.
# =============================================================================

using DataFrames, Dates, Statistics, CSV, Random, Logging

# ---------------------------------------------------------------------------
# Frozen holdout boundary.
# ---------------------------------------------------------------------------
const HOLDOUT_START = Date(2020, 1, 1)

# Steps-per-year used by the rest of the codebase (252 trading days / 5-day step).
const STEPS_PER_YEAR = 252.0 / 5.0

# Annualization helper.
_trading_years(n_steps::Int) = n_steps / STEPS_PER_YEAR

# ---------------------------------------------------------------------------
# filtering & leakage guards
# ---------------------------------------------------------------------------
"""
    filter_holdout(results; holdout_start=HOLDOUT_START)

Return only the walk-forward steps whose `result.date >= holdout_start`.
Also asserts that no selected step precedes the reporting boundary and its
training end precedes its signal date. Later rolling training may legitimately
use labels observed within the selected historical period.
"""
function filter_holdout(results; holdout_start::Date=HOLDOUT_START)
    holdout = [r for r in results if r !== nothing && !ismissing(r) && r.date >= holdout_start]
    if isempty(holdout)
        return holdout
    end

    # Anti-leakage assertion #1: every holdout result must be on/after the boundary.
    min_date = minimum(r.date for r in holdout)
    @assert min_date >= holdout_start begin
        "LEAKAGE: earliest holdout result date $(min_date) precedes HOLDOUT_START $(holdout_start)"
    end

    # Rolling walk-forward training continues within the holdout. The engine
    # filters individual labels by target_end_date <= train_end, not by a
    # calendar-day approximation or a globally frozen training dataset.
    all(r -> !hasproperty(r, :train_end) || r.train_end < r.date, holdout) || error("Training overlaps signal")
    first_holdout = holdout[1]
    if hasproperty(first_holdout, :train_end) && hasproperty(first_holdout, :train_start)
        te = first_holdout.train_end
        ts = first_holdout.train_start
        println("[LEAKAGE GUARD] first holdout step date         = $(first_holdout.date)")
        println("[LEAKAGE GUARD] first holdout train_end         = $te")
        println("[LEAKAGE GUARD] first holdout train_start       = $ts")
        @assert te < first_holdout.date "Train end must precede the first holdout signal"
    else
        # Fallback when the result tuple does not carry training window fields.
        println("[LEAKAGE GUARD] first holdout step date = $(first_holdout.date)  (>= $holdout_start : OK)")
        println("[LEAKAGE GUARD] result tuple has no train_end field; relying on test_start >= $holdout_start invariants.")
    end

    return holdout
end

# ---------------------------------------------------------------------------
# core metric computation (holdout-only)
# ---------------------------------------------------------------------------
"""
    compute_holdout_metrics(holdout_results; signal=:actual_alpha_long)

Compute the full institutional metric set over a holdout-only slice of
walk-forward results.  Returns `nothing` if the slice is empty.

`signal` selects which per-step return field is used to build the equity
curve.  The default `:actual_alpha_long` is the *weighted* long-basket return,
which is exactly what differs between A1 (equal weight) and F1 (rank+inv-vol).
"""
function compute_holdout_metrics(holdout_results; signal::Symbol=:actual_alpha_long)
    assert_complete_validation(holdout_results)
    valid = collect(holdout_results)
    all(r -> hasproperty(r, signal) && isfinite(getproperty(r, signal)), valid) ||
        error("Invalid return in validation")
    if isempty(valid)
        return nothing
    end

    alphas = Float64[getproperty(r, signal) for r in valid]
    dates  = Date[r.date for r in valid]
    n      = length(valid)

    benchmark = all(r -> hasproperty(r, :spy_ret) && isfinite(r.spy_ret), valid) ?
        [r.spy_ret for r in valid] : nothing
    core = Metrics.return_metrics(alphas; benchmark)
    equity_curve = core.equity_curve
    total_return = equity_curve[end] - 1
    cagr, sharpe, max_dd = core.cagr, core.sharpe, core.max_dd
    calmar = max_dd > 0 ? cagr / max_dd : NaN
    win_rate, avg_excess = core.win_rate, core.avg_excess

    # ----- annual returns --------------------------------------------------
    year_of   = year.(dates)
    years_set = sort(unique(year_of))
    annual_returns = Dict{Int, Float64}()
    for y in years_set
        idxs  = findall(==(y), year_of)
        sub_a = alphas[idxs]
        annual_returns[y] = prod(1.0 .+ sub_a) - 1.0
    end

    # Executed buy + sell notional relative to each window's initial NAV.
    turnovers = Float64[r.long_turnover for r in valid]
    mean_turnover = mean(turnovers)
    turnover_25 = quantile(turnovers, 0.25)

    # ----- cross-sectional diagnostics -------------------------------------
    ics          = Float64[getproperty(r, :ic)           for r in valid if hasproperty(r, :ic) && isfinite(getproperty(r, :ic))]
    rics         = Float64[getproperty(r, :rank_ic)      for r in valid if hasproperty(r, :rank_ic) && isfinite(getproperty(r, :rank_ic))]
    dec_spreads  = Float64[getproperty(r, :decile_spread) for r in valid if hasproperty(r, :decile_spread) && isfinite(getproperty(r, :decile_spread))]
    best_decs    = Float64[getproperty(r, :best_decile_return)  for r in valid if hasproperty(r, :best_decile_return)  && isfinite(getproperty(r, :best_decile_return))]
    worst_decs   = Float64[getproperty(r, :worst_decile_return) for r in valid if hasproperty(r, :worst_decile_return) && isfinite(getproperty(r, :worst_decile_return))]

    mean_ic            = isempty(ics)         ? NaN : mean(ics)
    mean_rank_ic       = isempty(rics)        ? NaN : mean(rics)
    mean_dec_spread    = isempty(dec_spreads) ? NaN : mean(dec_spreads)
    mean_best_decile   = isempty(best_decs)   ? NaN : mean(best_decs)
    mean_worst_decile  = isempty(worst_decs)  ? NaN : mean(worst_decs)

    return (
        config            = nothing,             # filled by caller
        n_steps           = n,
        date_start        = dates[1],
        date_end          = dates[end],
        total_return      = total_return,
        cagr              = cagr,
        sharpe            = sharpe,
        max_dd            = max_dd,
        calmar            = calmar,
        win_rate          = win_rate,
        avg_excess        = avg_excess,
        mean_return       = core.mean_return,
        annual_returns    = annual_returns,
        mean_turnover     = mean_turnover,
        turnover_p25      = turnover_25,
        turnover_max      = isempty(turnovers) ? NaN : maximum(turnovers),
        turnover_per_step = turnovers,
        trade_count       = n,
        mean_ic           = mean_ic,
        mean_rank_ic      = mean_rank_ic,
        mean_dec_spread   = mean_dec_spread,
        mean_best_decile  = mean_best_decile,
        mean_worst_decile = mean_worst_decile,
        equity_curve      = equity_curve,
        dates             = dates,
        alphas            = alphas,
    )
end

# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
function _pctv(v, d::Int=2)
    (v isa Real) || return string(v)
    isnan(v) && return "NaN"
    isinf(v) && return (v > 0 ? "+Inf" : "-Inf")
    return string(round(v * 100; digits=d)) * "%"
end
function _fnum(x, d::Int=2)
    isnan(x) && return "NaN"
    isinf(x) && return (x > 0 ? "+Inf" : "-Inf")
    return string(round(Float64(x); digits=d))
end

# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------
"""
    print_comparison_table(metrics::Dict{String, Any}; holdout_start::Date=HOLDOUT_START)

Pretty-print a side-by-side comparison of A1 vs F1 holdout metrics.
"""
function print_comparison_table(metrics::Dict{String, Any}; holdout_start::Date=HOLDOUT_START)
    println()
    println(repeat("=", 96))
    println("  HISTORICAL WALK-FORWARD VALIDATION  -  SELECTED PERIOD ONLY")
    println("  Holdout window: $(holdout_start)  ->  latest available data")
    println("  Model retrains on prior observations. This period has been studied before.")
    println(repeat("=", 96))

    order = sort(collect(keys(metrics)))
    n_cfg = length(order)
    if n_cfg == 0
        println("  (no valid configs to compare)")
        println(repeat("=", 96))
        println()
        return
    end

    # Header
    println()
    print(rpad("Metric", 30))
    for cfg in order
        print("  ", rpad(cfg, 24))
    end
    println()
    print(rpad(repeat("-", 28), 30))
    for cfg in order
        print("  ", rpad(repeat("-", 22), 24))
    end
    println()

    rows = [
        ("Steps (n)"               , m -> string(m.n_steps)),
        ("Date range"              , m -> "$(m.date_start) .. $(m.date_end)"),
        ("Total return"            , m -> _pctv(m.total_return)),
        ("CAGR"                    , m -> _pctv(m.cagr)),
        ("Sharpe (rf=5%)"          , m -> _fnum(m.sharpe)),
        ("Max Drawdown"            , m -> _pctv(m.max_dd)),
        ("Calmar"                  , m -> _fnum(m.calmar)),
        ("Win rate"                , m -> _pctv(m.win_rate)),
        ("Avg excess / step"       , m -> _pctv(m.avg_excess)),
        ("Mean IC"                 , m -> _fnum(m.mean_ic, 4)),
        ("Mean Rank IC"            , m -> _fnum(m.mean_rank_ic, 4)),
        ("Decile spread (mean)"    , m -> _pctv(m.mean_dec_spread)),
        ("Best decile (mean)"      , m -> _pctv(m.mean_best_decile)),
        ("Worst decile (mean)"     , m -> _pctv(m.mean_worst_decile)),
        ("Mean turnover / step"    , m -> _fnum(m.mean_turnover * 100) * "%"),
        ("Turnover p25"            , m -> _fnum(m.turnover_p25 * 100)  * "%"),
        ("Turnover max"            , m -> _fnum(m.turnover_max * 100)  * "%"),
        ("Trade (rebalance) count" , m -> string(m.trade_count)),
    ]
    for (label, f) in rows
        print(rpad(label, 30))
        for cfg in order
            print("  ", rpad(f(metrics[cfg]), 24))
        end
        println()
    end

    # Annual returns block
    println()
    println("  Annual returns (compounded within calendar year, holdout only):")
    all_years = sort(union([collect(keys(m.annual_returns)) for m in values(metrics)]...))
    print(rpad("Year", 30))
    for cfg in order
        print("  ", rpad(cfg, 24))
    end
    println()
    print(rpad(repeat("-", 28), 30))
    for cfg in order
        print("  ", rpad(repeat("-", 22), 24))
    end
    println()
    for y in all_years
        print(rpad(string(y), 30))
        for cfg in order
            v = get(metrics[cfg].annual_returns, y, NaN)
            print("  ", rpad(_pctv(v), 24))
        end
        println()
    end
    println()
    println(repeat("=", 96))
    println()
end

# ---------------------------------------------------------------------------
# CSV exports (per config + shared summary)
# ---------------------------------------------------------------------------

"""
    _execution_return(price_df, signal_date, exit_date)

Use the strategy's signal-close, next-open entry and fifth-close exit convention.
Missing or mismatched prices return `missing`, never an invented zero return.
"""
function _execution_return(price_df::Union{DataFrame,Nothing},
                           signal_date::Date, exit_date::Date)
    price_df === nothing && return missing
    s_idx = searchsortedfirst(price_df.date, signal_date)
    e_idx = s_idx + Features.FORECAST_HORIZON
    (s_idx > nrow(price_df) || e_idx > nrow(price_df)) && return missing
    (price_df.date[s_idx] != signal_date || price_df.date[e_idx] != exit_date) && return missing
    s_px = price_df.open[s_idx + 1]
    e_px = price_df.close[e_idx]
    (ismissing(s_px) || ismissing(e_px)) && return missing
    (isfinite(s_px) && isfinite(e_px) && s_px > 0.0) || return missing
    return e_px / s_px - 1.0
end

"""
    _step_turnover(prev, curr) -> Float64

One-way portfolio turnover between two consecutive rebalance steps,
defined as `0.5 * sum_i |w_curr[i] - w_prev[i]|` over the union of names.
Mirrors the `compute_holdout_metrics` turnover logic verbatim so the per-step
column matches the aggregate `mean_turnover` statistic.
"""
function _step_turnover(prev, curr)
    prev_tickers = collect(prev.tickers_long)
    curr_tickers = collect(curr.tickers_long)
    prev_w = hasproperty(prev, :inv_vol_weights) ?
        collect(prev.inv_vol_weights) :
        fill(1.0 / max(1, length(prev_tickers)), max(1, length(prev_tickers)))
    curr_w = hasproperty(curr, :inv_vol_weights) ?
        collect(curr.inv_vol_weights) :
        fill(1.0 / max(1, length(curr_tickers)), max(1, length(curr_tickers)))
    prev_map = Dict{String,Float64}()
    for (t, w) in zip(prev_tickers, prev_w)
        prev_map[string(t)] = Float64(w)
    end
    agg = 0.0
    for t in union(Set(prev_tickers), Set(curr_tickers))
        ts = string(t)
        p_w = get(prev_map, ts, 0.0)
        c_i = findfirst(==(t), curr_tickers)
        c_w = (c_i === nothing) ? 0.0 : Float64(curr_w[c_i])
        agg += abs(c_w - p_w)
    end
    return agg * 0.5
end

function _holdout_step_rows(holdout_results;
                             spy_data::Union{DataFrame,Nothing}=nothing,
                             qqq_data::Union{DataFrame,Nothing}=nothing,
                             soxx_data::Union{DataFrame,Nothing}=nothing)
    rows = []
    prev = nothing
    for r in holdout_results
        if r === nothing || ismissing(r)
            continue
        end
        # Pull start/end dates from the result tuple (backtest.jl writes both).
        ts = hasproperty(r, :test_start) ? r.test_start : r.date
        te = hasproperty(r, :test_end)   ? r.test_end   : r.date
        spy_return  = _execution_return(spy_data,  ts, te)
        qqq_return  = _execution_return(qqq_data,  ts, te)
        soxx_return = _execution_return(soxx_data, ts, te)
        turnover = hasproperty(r, :long_turnover) ? r.long_turnover :
               (prev === nothing ? 1.0 : _step_turnover(prev, r))

        push!(rows, (
            methodology     = r.methodology,
            date            = r.date,
            actual_alpha_long = r.actual_alpha_long,
            net_alpha_long  = hasproperty(r, :net_alpha_long) ? r.net_alpha_long : missing,
            long_cost       = hasproperty(r, :long_cost) ? r.long_cost : missing,
            chosen_alpha    = r.chosen_alpha,
            ic              = r.ic,
            rank_ic         = r.rank_ic,
            decile_spread   = r.decile_spread,
            best_decile     = r.best_decile_return,
            worst_decile    = r.worst_decile_return,
            n_universe      = r.n_universe,
            n_long          = length(r.tickers_long),
            long_tickers    = join(string.(r.tickers_long), ";"),
            weights         = join([round(w; digits=6) for w in r.inv_vol_weights], ";"),
            spy_return      = spy_return,
            qqq_return      = qqq_return,
            soxx_return     = soxx_return,
            turnover        = turnover,
        ))
        prev = r
    end
    return rows
end

function export_holdout_csvs(config_id::AbstractString, holdout_results, metrics;
                              spy_data::Union{DataFrame,Nothing}=nothing,
                              qqq_data::Union{DataFrame,Nothing}=nothing,
                              soxx_data::Union{DataFrame,Nothing}=nothing)
    out_dir = joinpath(pwd(), "results", "corrected")
    mkpath(out_dir)

    steps_path    = joinpath(out_dir, "validation_holdout_$(config_id)_steps.csv")
    annual_path   = joinpath(out_dir, "validation_holdout_$(config_id)_annual.csv")
    equity_path   = joinpath(out_dir, "validation_holdout_$(config_id)_equity.csv")

    CSV.write(steps_path, DataFrame(_holdout_step_rows(holdout_results;
                                                         spy_data=spy_data,
                                                         qqq_data=qqq_data,
                                                         soxx_data=soxx_data)))
    println("[validation] wrote $steps_path")

    annual_df = DataFrame(
        year = sort(collect(keys(metrics.annual_returns))),
        annual_return = [metrics.annual_returns[y] for y in sort(collect(keys(metrics.annual_returns)))],
    )
    CSV.write(annual_path, annual_df)
    println("[validation] wrote $annual_path")

    equity_df = DataFrame(
        step_index = 0:(length(metrics.equity_curve) - 1),
        date = vcat([metrics.dates[1]], metrics.dates),
        equity = metrics.equity_curve,
    )
    CSV.write(equity_path, equity_df)
    println("[validation] wrote $equity_path")
end

function export_summary_csv(metrics_by_config::Dict{String, Any})
    out_dir = joinpath(pwd(), "results", "corrected")
    mkpath(out_dir)
    path = joinpath(out_dir, "validation_holdout_summary.csv")

    all_years = sort(union([collect(keys(m.annual_returns)) for m in values(metrics_by_config)]...))
    # Build a vector of homogeneous NamedTuples and construct the DataFrame from
    # that — avoids the DataFrames.jl `push!(df, row)` issue that requires the
    # row to have the SAME number of fields as the existing df columns.
    rows = NamedTuple[]
    for cfg in sort(collect(keys(metrics_by_config)))
        m = metrics_by_config[cfg]
        push!(rows, (
            config            = String(cfg),
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
            mean_return       = Float64(m.mean_return),
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
    # Add annual return columns (one per year that appears in any config).
    for y in all_years
        col = Symbol(string("annual_", y))
        df[!, col] = [get(metrics_by_config[cfg].annual_returns, y, NaN) for cfg in df.config]
    end
    CSV.write(path, df)
    println("[validation] wrote $path  (cols = $(length(names(df))))")
end

# ---------------------------------------------------------------------------
# top-level validation runner
# ---------------------------------------------------------------------------
"""
    run_validation(; top_n=10, universe="all", exclude_features=Set{String}(),
                   inv_vol_exp=0.25, exclude_tickers=Set{String}(),
                   holdout_start=HOLDOUT_START, train_window=:fixed, max_steps=nothing)

Run one historical walk-forward validation configuration:
  1. Set RNG seed (42).
    2. Load the cached stock and benchmark data once.
    3. Run the chosen model blend, training window and portfolio weighting.
    4. Evaluate only steps on/after holdout_start with prior-only training labels.
    5. Compute net metrics and export per-step and summary CSVs to octave/.

CLI kwargs (Phase 1 - experiment support):
  - top_n            (Int)                 : # of long positions per rebalance.
  - universe         (String)              : "all" | "tech" | "semis"
  - exclude_features (Set{String})         : feature columns to drop.
  - inv_vol_exp      (Float64)             : inv-vol exponent passed to backtest.
  - exclude_tickers  (Set{String})         : tickers to drop from universe.
    - holdout_start    (Date)                : historical reporting start; default = HOLDOUT_START.
  - train_window     (Symbol|Int)          : :fixed (252d), :expanding, or Int days.
  - max_steps        (Int|nothing)         : if set, limits number of walk-forward
                                              steps (smoke testing). Falls back to
                                              ENV GREEKFIRE_MAX_STEPS if not set.
    - weighting_method (String)              : equal (default), shifted, rank, or inverse-vol variants.
    - transaction_cost_bps, slippage_bps    : nonnegative cost assumptions.
    - ensemble_rf_weight (Float64)           : RF share of RF/XGBoost blend, default 0.5.
"""
function assert_complete_validation(results)
    isempty(results) && error("No validation steps")
    missing_steps = count(result -> result === nothing, results)
    missing_steps == 0 || error("Validation has $missing_steps missing steps; refusing to report selective-period performance")
    return nothing
end

function run_validation(;
                          top_n::Int=10,
                          universe::String="all",
                          exclude_features::Set{String}=Set{String}(),
                          inv_vol_exp::Float64=0.25,
                          exclude_tickers::Set{String}=Set{String}(),
                          holdout_start::Date=HOLDOUT_START,
                          train_window::Union{Int,Symbol}=:fixed,
                          max_steps=nothing,
                          weighting_method::String="equal",
                          transaction_cost_bps::Float64=10.0,
                          slippage_bps::Float64=5.0,
                          ensemble_rf_weight::Float64=0.5,
                          prediction_mode::Symbol=:ensemble)
    Random.seed!(42)
    println("=" ^ 96)
    println("  GreekFire — HISTORICAL WALK-FORWARD VALIDATION")
    println("  Holdout boundary: $(holdout_start)")
    println("  Weighting: $(weighting_method), inv_vol_exponent=$(inv_vol_exp)")
    println("  RF/XGBoost model blend: $(ensemble_rf_weight)/$(1 - ensemble_rf_weight)")
    println("  Assumed one-way costs: $(transaction_cost_bps) bps transaction + $(slippage_bps) bps slippage")
    println("  Universe : $(universe)   (exclude_tickers = $(isempty(exclude_tickers) ? "(none)" : collect(exclude_tickers)))")
    println("  Top N    : $(top_n)   train_window=$(repr(train_window))   exclude_features=$(isempty(exclude_features) ? "(none)" : collect(exclude_features))")
    println("  Threads  : $(Threads.nthreads())")
    println("=" ^ 96)

    all_stock_data = DataLoader.load_all_data(universe=universe, exclude_tickers=exclude_tickers)
    spy_data = DataLoader.load_ticker_data("SPY")
    if spy_data === nothing
        error("Failed to load SPY benchmark data. Cannot proceed with validation.")
    end
    # Benchmark ETF data for additional columns in the steps CSV.
    # Loaded once and shared across all configs.
    qqq_data  = DataLoader.load_ticker_data("QQQ")
    soxx_data = DataLoader.load_ticker_data("SOXX")
    if qqq_data === nothing
        @warn "[validation] QQQ benchmark data not found in data/ — qqq_return will be missing."
    end
    if soxx_data === nothing
        @warn "[validation] SOXX benchmark data not found in data/ — soxx_return will be missing."
    end
    println("[validation] data loaded: $(length(all_stock_data)) tickers  (universe=$(universe))")
    println("[validation] benchmark data: SPY=$(spy_data === nothing ? "MISSING" : "OK")  " *
            "QQQ=$(qqq_data === nothing ? "MISSING" : "OK")  " *
            "SOXX=$(soxx_data === nothing ? "MISSING" : "OK")")

    # Optional smoke-test knob: CLI kwarg takes precedence; otherwise fall back
    # to ENV. Leave unset for the real institutional run.
    if max_steps === nothing
        max_steps_env = get(ENV, "GREEKFIRE_MAX_STEPS", nothing)
        if max_steps_env !== nothing
            max_steps = parse(Int, max_steps_env)
        end
    end
    if max_steps !== nothing
        println("[validation] SMOKE TEST: max_steps = $max_steps")
    end

    configs = [
        (
            id = "A1",
            weighting_method   = weighting_method,
            inv_vol_exponent   = inv_vol_exp,   # value is irrelevant for "equal"; keep matching const default
            max_position_cap   = 0.0,
        ),
    ]

    metrics_by_config = Dict{String, Any}()

    for (i, cfg) in enumerate(configs)
        println()
        println("=" ^ 96)
        println("  CONFIG $(cfg.id)  ($i / $(length(configs)))")
        println("    weighting_method  = $(repr(cfg.weighting_method))")
        println("    inv_vol_exponent  = $(cfg.inv_vol_exponent)")
        println("    max_position_cap  = $(cfg.max_position_cap)")
        println("    top_n             = $(top_n)")
        println("    train_window      = $(repr(train_window))")
        println("    exclude_features  = $(isempty(exclude_features) ? "(none)" : collect(exclude_features))")
        println("=" ^ 96)

        Random.seed!(42)  # re-seed before EACH config so the two runs are directly comparable
        t_run = @elapsed results, importances, ic_hist, rank_ic_hist, decile_hist =
            Backtest.run_walk_forward(
                all_stock_data, spy_data;
                weighting_method  = cfg.weighting_method,
                inv_vol_exponent  = cfg.inv_vol_exponent,
                max_position_cap  = cfg.max_position_cap,
                max_steps         = max_steps,
                top_n             = top_n,
                exclude_features  = exclude_features,
                train_window      = train_window,
                first_test_date   = holdout_start,
                transaction_cost_bps = transaction_cost_bps,
                slippage_bps      = slippage_bps,
                ensemble_rf_weight = ensemble_rf_weight,
                prediction_mode = prediction_mode,
            )
        println("[validation] $(cfg.id) walk-forward wall time: $(round(t_run; digits=1))s")

        # Apply the same post-processing main.jl applies: regime disabled,
        # chosen_alpha = net_alpha_long for every step.
        for k in 1:length(results)
            r = results[k]
            if r !== nothing
                results[k] = merge(
                    r,
                    (chosen_alpha  = r.net_alpha_long,
                     regime_signal = :bull),
                )
            end
        end

        assert_complete_validation(results)
        n_valid = length(results)
        valid_results = [r for r in results if r !== nothing]
        if !isempty(valid_results)
            println("[validation] $(cfg.id) valid steps: $(n_valid)  date range: $(valid_results[1].date) .. $(valid_results[end].date)")
        else
            println("[validation] $(cfg.id) produced NO valid results — skipping.")
            continue
        end

        # ---- filter & guard ------------------------------------------------
        holdout = filter_holdout(results; holdout_start=holdout_start)
        if isempty(holdout)
            println("[validation] $(cfg.id) produced NO holdout steps — skipping metrics.")
            continue
        end
        println("[validation] $(cfg.id) holdout steps: $(length(holdout)) ($(holdout[1].date) .. $(holdout[end].date))")

        # ---- metrics -------------------------------------------------------
        m = compute_holdout_metrics(holdout; signal=:net_alpha_long)
        if m === nothing
            println("[validation] $(cfg.id) no finite-alpha holdout steps — skipping.")
            continue
        end
        m = merge(m, (config = cfg.id,))
        metrics_by_config[cfg.id] = m

        export_holdout_csvs(cfg.id, holdout, m; spy_data=spy_data, qqq_data=qqq_data, soxx_data=soxx_data)
    end

    if isempty(metrics_by_config)
        println()
        @warn "No valid configurations were produced"
    end

    if !isempty(metrics_by_config)
        print_comparison_table(metrics_by_config; holdout_start=holdout_start)
        export_summary_csv(metrics_by_config)
    else
        println("[validation] no configs produced holdout metrics.")
    end

    return metrics_by_config
end
