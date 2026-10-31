module Backtest

using DataFrames, Dates, Statistics, StatsBase, ProgressMeter, Random
using Base.Threads

include("model.jl")
include("features.jl")
include("data_loader.jl")

const TRAIN_WINDOW_DAYS = 252
const STEP_DAYS = 5
const TOP_N_STOCKS = 1
const BOTTOM_N_STOCKS = 5
const STOP_LOSS_THRESHOLD = 0.07

# Inverse-vol weighting exponent. 0.0 = equal weight, 0.25/0.50 = partial, 1.0 = pure inverse-vol.
const INVERSE_VOL_EXPONENT = 0.50

# Portfolio weighting method: "equal", "shifted", "shifted_inv_vol", "rank", "rank_inv_vol"
const WEIGHTING_METHOD = "rank_inv_vol"

# Inverse-vol exponent (used when method includes "inv_vol")
const INV_VOL_EXPONENT = 0.25

# Maximum position cap (0.0 = no cap, 0.20 = 20%, 0.15 = 15%)
const MAX_POSITION_CAP = 0.0

# Keep the old short/SPY overlay available, but off by default.
const ENABLE_SHORT_OVERLAY = false
const CONFIDENCE_THRESHOLD = 0.01

const NON_FEATURE_COLS = Set(["date", "target", "target_end_date", "ticker", "raw_idio_vol_20"])

function one_way_turnover(previous_tickers, previous_weights, current_tickers, current_weights)
    previous = Dict(string(ticker) => weight for (ticker, weight) in zip(previous_tickers, previous_weights))
    current = Dict(string(ticker) => weight for (ticker, weight) in zip(current_tickers, current_weights))
    stock_change = sum(abs(get(current, ticker, 0.0) - get(previous, ticker, 0.0))
                       for ticker in union(keys(previous), keys(current)))
    cash_change = abs(sum(current_weights) - sum(previous_weights))
    return (stock_change + cash_change) / 2
end

function training_rows(master_df::DataFrame, step, active_set)
    return master_df[
        (master_df.date .>= step.train_start) .&
        isfinite.(master_df.target) .&
        (master_df.target_end_date .<= step.train_end) .&
        (master_df.ticker .∈ Ref(active_set)), :]
end

function ensemble_predictions(rf_predictions, xgb_predictions, rf_weight::Float64)
    0.0 <= rf_weight <= 1.0 || throw(ArgumentError("RF ensemble weight must be between 0 and 1"))
    return xgb_predictions === nothing ? rf_predictions :
           rf_weight .* rf_predictions .+ (1 - rf_weight) .* xgb_predictions
end

const SECTOR_MAP = Dict(
    "AAPL" => "Tech",     "MSFT" => "Tech",     "GOOGL" => "Tech",
    "AMZN" => "Tech",     "NVDA" => "Tech",     "META" => "Tech",
    "AMD"  => "Tech",     "INTC" => "Tech",     "ADBE" => "Tech",
    "CSCO" => "Tech",     "TXN" => "Tech",      "QCOM" => "Tech",
    "AVGO" => "Tech",     "ACN" => "Tech",      "IBM" => "Tech",
    "HPQ"  => "Tech",     "ORCL" => "Tech",
    "JPM"  => "Finance",  "BAC" => "Finance",   "GS" => "Finance",
    "AXP"  => "Finance",  "BLK" => "Finance",   "SCHW" => "Finance",
    "USB"  => "Finance",  "PNC" => "Finance",   "COF" => "Finance",
    "MET"  => "Finance",  "AIG" => "Finance",   "TRV" => "Finance",
    "PGR"  => "Finance",  "MS" => "Finance",    "BK" => "Finance",
    "STT"  => "Finance",  "CI" => "Finance",    "UNH" => "Finance",
    "HUM"  => "Finance",  "AFL" => "Finance",   "PRU" => "Finance",
    "ALL"  => "Finance",  "BEN" => "Finance",   "TROW" => "Finance",
    "IVZ"  => "Finance",  "NTRS" => "Finance",  "DFS" => "Finance",
    "SYF"  => "Finance",  "CINF" => "Finance",  "L" => "Finance",
    "BRK-B"=> "Finance",  "V" => "Finance",     "MA" => "Finance",
    "C"    => "Finance",
    "XOM"  => "Energy",   "CVX" => "Energy",    "COP" => "Energy",
    "EOG"  => "Energy",   "SLB" => "Energy",    "HAL" => "Energy",
    "NOV"  => "Energy",   "OXY" => "Energy",    "MRO" => "Energy",
    "DVN"  => "Energy",   "CHK" => "Energy",    "RRC" => "Energy",
    "SWN"  => "Energy",   "EQT" => "Energy",    "AR" => "Energy",
    "PXD"  => "Energy",
    "JNJ"  => "Health",   "PFE" => "Health",    "MRK" => "Health",
    "AMGN" => "Health",   "BMY" => "Health",    "ABT" => "Health",
    "MDT"  => "Health",   "TMO" => "Health",    "DHR" => "Health",
    "DIS"  => "Media",    "CMCSA" => "Media",
    "KO"   => "Consumer", "WMT" => "Consumer",  "PG" => "Consumer",
    "PEP"  => "Consumer", "MCD" => "Consumer",  "NKE" => "Consumer",
    "TGT"  => "Consumer", "HD" => "Consumer",   "EMR" => "Industrial",
    "HON"  => "Industrial","BA" => "Industrial","MMM" => "Industrial",
    "CAT"  => "Industrial","GE" => "Industrial", "LMT" => "Industrial",
    "GD"   => "Industrial","NOC" => "Industrial","UPS" => "Industrial",
    "FDX"  => "Industrial","CMI" => "Industrial","DE" => "Industrial",
    "F"    => "Auto",
    "NEE"  => "Utilities",
    "T"    => "Telecom",  "VZ" => "Telecom",
    "LUV"  => "Airlines", "DAL" => "Airlines",  "UAL" => "Airlines",
)

function percentile_rank(values::AbstractVector)
    n = length(values)
    out = fill(0.5, n)

    finite_idx = Int[]
    for i in eachindex(values)
        v = values[i]
        if v isa Real && isfinite(v)
            push!(finite_idx, i)
        end
    end

    if isempty(finite_idx)
        return out
    elseif length(finite_idx) == 1
        out[finite_idx[1]] = 0.5
        return out
    end

    finite_vals = Float64.(values[finite_idx])
    ranks = tiedrank(finite_vals)
    pct = clamp.((ranks .- 1.0) ./ (length(finite_vals) - 1.0), 0.02, 0.98)
    out[finite_idx] = pct
    return out
end

# master_df is sorted by date. Rank contiguous ranges directly, avoiding a
# full-table boolean scan and repeated SubDataFrame column replacement.
function rank_columns!(df, columns)
    lo = 1
    while lo <= nrow(df)
        hi = searchsortedlast(df.date, df.date[lo])
        for col in columns
            df[!, col][lo:hi] = percentile_rank(view(df[!, col], lo:hi))
        end
        lo = hi + 1
    end
    return df
end



function run_walk_forward(all_stock_data, spy_data;
                           max_steps=nothing,
                           weighting_method::String=WEIGHTING_METHOD,
                           inv_vol_exponent::Float64=INV_VOL_EXPONENT,
                           max_position_cap::Float64=MAX_POSITION_CAP,
                           top_n::Int=TOP_N_STOCKS,
                           exclude_features::Set{String}=Set{String}(),
                           train_window::Union{Int,Symbol}=:fixed,
                           first_test_date::Union{Date,Nothing}=nothing,
                           transaction_cost_bps::Float64=10.0,
                           slippage_bps::Float64=5.0,
                           ensemble_rf_weight::Float64=0.5,
                           membership_provider=DataLoader.get_sp500_constituents,
                           prediction_mode::Symbol=:ensemble,
                           verbose::Bool=false)
    prediction_mode in (:ensemble, :momentum) || throw(ArgumentError("Unknown prediction mode"))
    (isfinite(transaction_cost_bps) && transaction_cost_bps >= 0 &&
     isfinite(slippage_bps) && slippage_bps >= 0) ||
        throw(ArgumentError("Transaction costs and slippage must be finite and nonnegative"))
    top_n > 0 || throw(ArgumentError("top_n must be positive"))
    0 <= max_position_cap <= 1 || throw(ArgumentError("position cap must be in [0,1]"))
    train_window isa Int && train_window <= 0 && throw(ArgumentError("training window must be positive"))
    max_steps !== nothing && max_steps <= 0 && throw(ArgumentError("max_steps must be positive"))
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    0.0 <= ensemble_rf_weight <= 1.0 ||
        throw(ArgumentError("RF ensemble weight must be between 0 and 1"))
    ENABLE_SHORT_OVERLAY && cost_rate > 0 &&
        throw(ArgumentError("Short/SPY overlay has no executable cost model; disable it for net reporting"))
    @info "Pre-computing feature matrices..."
    tickers = sort(collect(keys(all_stock_data)))
    all_features_vec = Vector{Pair{String, DataFrame}}(undef, length(tickers))

    @threads for i in 1:length(tickers)
        ticker = tickers[i]
        all_features_vec[i] = ticker => Features.calculate_features(all_stock_data[ticker], spy_data)
    end

    @info "Building master DataFrame..."
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
    # Risk sizing must use measured volatility, not its percentile rank.
    master_df.raw_idio_vol_20 = copy(master_df.idio_vol_20)
    sort!(master_df, [:date, :ticker])
    memberships = Dict(d => Set(membership_provider(d)) for d in unique(master_df.date))
    filter!(row -> row.ticker in memberships[row.date], master_df)

    all_dates = sort(unique(master_df.date))

    @info "Computing cross-sectional rank_ret_20..."
    master_df[!, :rank_ret_20] = fill(0.5, nrow(master_df))
    master_df.rank_ret_20 = copy(master_df.ret_20)
    rank_columns!(master_df, [:rank_ret_20])

    feature_cols = String[]
    for col in names(master_df)
        if (eltype(master_df[!, col]) <: Number) && !(col in NON_FEATURE_COLS)
            push!(feature_cols, String(col))
        end
    end

    if isempty(feature_cols)
        error("No numeric feature columns found.")
    end

    # Apply user-supplied feature exclusions (CLI: --exclude-features=...).
    # Done BEFORE cross-sectional standardization so dropped features aren't
    # percentile-ranked either. Removes the column from `feature_cols` only;
    # the underlying `master_df` columns remain untouched so the model
    # bookkeeping stays consistent.
    if !isempty(exclude_features)
        n_excluded = count(x -> x in exclude_features, feature_cols)
        filter!(x -> !(x in exclude_features), feature_cols)
        println("[run_walk_forward] excluded $n_excluded feature column(s) via exclude_features kwarg")
    end

    @info "Standardizing features cross-sectionally (percentile rank + clamp)..."
    rank_columns!(master_df, filter(!=("rank_ret_20"), feature_cols))

    # Remove ret_20 from feature columns (redundant with rank_ret_20)
    filter!(x -> x != "ret_20", feature_cols)

    trading_dates = sort(unique(spy_data.date))
    start_date = DataLoader.START_DATE

    # Resolve effective training window length.
    #   train_window == :expanding -> 0 (sentinel: use full history from index 1)
    #   train_window == :fixed     -> TRAIN_WINDOW_DAYS (module default)
    #   train_window isa Int       -> that many days
    effective_train_window = if train_window === :expanding
        0
    elseif train_window === :fixed
        TRAIN_WINDOW_DAYS
    elseif train_window isa Int
        Int(train_window)
    else
        error("Unsupported train_window=$(repr(train_window)); expected :expanding, :fixed, or an Int")
    end
    println("[run_walk_forward] train_window=$(repr(train_window))  effective_days=$(effective_train_window == 0 ? "expanding (all history)" : effective_train_window)")

    steps = []
    curr_idx = if effective_train_window == 0
        max(2, searchsortedfirst(trading_dates, first(all_dates)) + TRAIN_WINDOW_DAYS)
    else
        max(searchsortedfirst(trading_dates, start_date), searchsortedfirst(trading_dates, first(all_dates))) + effective_train_window
    end
    while curr_idx + STEP_DAYS + 1 <= length(trading_dates)
        train_end   = trading_dates[curr_idx]
        test_start  = trading_dates[curr_idx + 1]
        test_end    = trading_dates[curr_idx + STEP_DAYS + 1]
        train_start = effective_train_window == 0 ?
            trading_dates[1] :
            trading_dates[max(1, curr_idx - effective_train_window + 1)]

        push!(steps, (
            train_start = train_start,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end
        ))
        curr_idx += STEP_DAYS
    end

    if first_test_date !== nothing
        filter!(step -> step.test_start >= first_test_date, steps)
    end

    total_steps = length(steps)
    if max_steps !== nothing && max_steps < total_steps
        total_steps = max_steps
    end

    results = Any[nothing for _ in 1:total_steps]
    feature_importances_history = Any[nothing for _ in 1:total_steps]
    ic_history = Float64[]
    rank_ic_history = Float64[]
    decile_history = Vector{Vector{Float64}}()

    is_debug = max_steps !== nothing
    if is_debug
        println("Bounded run: $total_steps steps")
    end
    p = Progress(total_steps, desc = "Walk-Forward Loop")

    feature_syms = Symbol.(feature_cols)
    short_count = 0
    spy_count = 0
    previous_long_tickers = String[]
    previous_long_weights = Float64[]
    previous_linear_tickers = String[]
    previous_linear_weights = Float64[]

    if ENABLE_SHORT_OVERLAY
        println("Short/SPY overlay is enabled.")
    else
        println("Short/SPY overlay is disabled. Chosen alpha will default to base long-only alpha.")
    end

    # Pre-cache point-in-time universe for all step dates
    println("Pre-caching S&P 500 point-in-time universe...")
    pit_cache = Dict{Date, Set{String}}()
    for step in steps
        date = step.test_start
        if !haskey(pit_cache, date)
            active = membership_provider(date)
            pit_cache[date] = Set(DataLoader.resolve_ticker(t) for t in active)
        end
    end

    for i in 1:total_steps
        step = steps[i]

        # Point-in-time universe: only use tickers active in S&P 500 on test_start
        active_set = pit_cache[step.test_start]

        # Each label must have finished before the model is trained.
        combined_train = training_rows(master_df, step, active_set)

        combined_test = master_df[
            (master_df.date .== step.test_start) .&
            (master_df.ticker .∈ Ref(active_set)), :]

        if verbose && i == 1
            println("Step 1 train rows: $(nrow(combined_train)), test rows: $(nrow(combined_test))")
            println("Step 1 active_set size: $(length(active_set))")
            println("Step 1 master_df rows: $(nrow(master_df))")
        end

        if isempty(feature_cols) || nrow(combined_train) == 0 || nrow(combined_test) == 0
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        try
            # Coverage is checked before expensive fitting and never used to
            # replace a missing candidate with another stock.
            all(isfinite, combined_test.target) || error("Missing forward outcome in signal universe at $(step.test_start)")
            if prediction_mode == :momentum
                pred_rf = combined_test.rank_ret_20
                pred_lr, pred_xgb = pred_rf, pred_rf
                mach_lr, mach_xgb = nothing, nothing
                importances = Dict{String,Float64}()
                _t_total = 0.0
            else
                _t_total = @elapsed (pred_rf, pred_lr, pred_xgb, importances, _mach_rf, mach_lr, mach_xgb) =
                    Model.train_and_predict(combined_train, combined_test, feature_cols)
            end
            @debug "Training and prediction seconds" elapsed=_t_total

            # Fall back to RF-only if XGB fails
            ensemble_pred = ensemble_predictions(pred_rf, mach_xgb === nothing ? nothing : pred_xgb,
                                                 ensemble_rf_weight)

            feature_importances_history[i] = importances

            linear_pred = mach_lr === nothing ? pred_rf : pred_lr

            period_scores = DataFrame(
                ticker      = combined_test.ticker,
                avg_pred    = ensemble_pred,
                linear_pred = linear_pred,
                avg_actual  = combined_test.target,
                idio_vol_20 = combined_test.raw_idio_vol_20
            )

            all(isfinite, period_scores.avg_pred) || error("Invalid ensemble prediction")
            all(isfinite, period_scores.linear_pred) || error("Invalid linear prediction")
            # Refuse incomplete outcome coverage instead of dropping candidates.
            # No defensible delisting payoff can be inferred from a missing bar.
            all(isfinite, period_scores.avg_actual) || error("Missing forward outcome in signal universe at $(step.test_start); supply verified prices/delisting payouts")

            if nrow(period_scores) == 0
                results[i] = nothing
                next!(p)
                continue
            end

            # IC / Rank IC
            pred_vals = Float64.(period_scores.avg_pred)
            actual_vals = Float64.(period_scores.avg_actual)
            ic = NaN
            rank_ic = NaN
            if length(pred_vals) >= 5 && std(pred_vals) > 1e-12 && std(actual_vals) > 1e-12
                ic = cor(pred_vals, actual_vals)
                rank_ic = cor(tiedrank(pred_vals), tiedrank(actual_vals))
                push!(ic_history, ic)
                push!(rank_ic_history, rank_ic)
            else
                push!(ic_history, NaN)
                push!(rank_ic_history, NaN)
            end

            # Deciles
            # Decile 1 = best predicted, Decile 10 = worst predicted
            sort!(period_scores, :avg_pred, rev = true)
            n_dec = nrow(period_scores)
            decs = Float64[]
            for d in 1:10
                lo = max(1, (d - 1) * n_dec ÷ 10 + 1)
                hi = min(n_dec, d * n_dec ÷ 10)
                if lo <= hi
                    push!(decs, mean(period_scores[lo:hi, :avg_actual]))
                else
                    push!(decs, NaN)
                end
            end
            push!(decile_history, decs)
            best_decile_return = decs[1]
            worst_decile_return = decs[end]
            decile_spread = best_decile_return - worst_decile_return

            pred_spread = maximum(period_scores.avg_pred) - minimum(period_scores.avg_pred)

            if verbose && i <= 10
                open(joinpath(pwd(), "debug.txt"), "a") do f
                    println(f, "=== Step $i ===")
                    println(f, "  period_scores rows: $(nrow(period_scores))")
                    println(f, "  pred_spread: $pred_spread")
                end
            end

            # LONG portfolio: pick top `top_n`, sector-capped (mirrors short basket).
            # Preserves current sort order; caps each sector at 2 (3 for Unknown) with a fallback fill.
            long_indices        = Int[]
            long_sector_counts  = Dict{String, Int}()
            for (idx, row) in enumerate(eachrow(period_scores))
                length(long_indices) >= top_n && break
                sector = get(SECTOR_MAP, row.ticker, "Unknown")
                count  = get(long_sector_counts, sector, 0)
                sector_limit = sector == "Unknown" ? 3 : 2
                if count < sector_limit
                    push!(long_indices, idx)
                    long_sector_counts[sector] = count + 1
                end
            end
            long_candidates = period_scores[long_indices, :]

            # Portfolio weighting: parameterized method
            # Methods: equal, shifted, shifted_inv_vol, rank, rank_inv_vol
            preds = [row.avg_pred for row in eachrow(long_candidates)]
            vols = [row.idio_vol_20 for row in eachrow(long_candidates)]
            n_stocks = length(preds)

            if weighting_method == "equal"
                # Experiment A: Equal weight
                raw_weights = fill(1.0, n_stocks)

            elseif weighting_method == "shifted"
                # Experiment B: Shifted prediction weighting
                min_pred = minimum(preds)
                shifted_scores = preds .- min_pred
                raw_weights = shifted_scores

            elseif weighting_method == "shifted_inv_vol"
                # Experiments C & D: Shifted prediction + partial inv-vol
                min_pred = minimum(preds)
                shifted_scores = preds .- min_pred
                inv_vol_adj = [isfinite(v) && v > 1e-8 ? (1.0/v)^inv_vol_exponent : (1.0/0.01)^inv_vol_exponent for v in vols]
                raw_weights = shifted_scores .* inv_vol_adj

            elseif weighting_method == "rank"
                # Experiment E: Rank-based weighting
                ranks = sortperm(sortperm(preds))  # double sortperm gives ranks
                percentiles = (ranks .- 0.5) ./ n_stocks
                raw_weights = percentiles

            elseif weighting_method == "rank_inv_vol"
                # Experiment F: Rank + partial inv-vol
                ranks = sortperm(sortperm(preds))
                percentiles = (ranks .- 0.5) ./ n_stocks
                inv_vol_adj = [isfinite(v) && v > 1e-8 ? (1.0/v)^inv_vol_exponent : (1.0/0.01)^inv_vol_exponent for v in vols]
                raw_weights = percentiles .* inv_vol_adj

            else
                # Fallback to equal weight
                raw_weights = fill(1.0, n_stocks)
            end

            # Normalize weights
            total_raw = sum(raw_weights)
            inv_vol_weights = total_raw > 0 ? raw_weights ./ total_raw : fill(1.0 / n_stocks, n_stocks)

            # Apply position cap if specified
            if max_position_cap > 0.0
                # Iterative capping: cap positions and redistribute excess
                for _ in 1:10  # max 10 iterations to converge
                    excess = sum(max(0, w - max_position_cap) for w in inv_vol_weights)
                    if excess < 1e-6
                        break
                    end
                    # Cap positions
                    inv_vol_weights = [min(w, max_position_cap) for w in inv_vol_weights]
                    # Redistribute excess proportionally to uncapped positions
                    uncapped_mask = inv_vol_weights .< max_position_cap
                    if sum(uncapped_mask) > 0
                        redistribution = excess / sum(uncapped_mask)
                        inv_vol_weights = [w + (uncapped_mask[i] ? redistribution : 0.0) for (i, w) in enumerate(inv_vol_weights)]
                    end
                end
                # Infeasible caps retain the remainder as zero-yield cash.
                inv_vol_weights = min.(inv_vol_weights, max_position_cap)
            end

            # Weighted portfolio return
            actual_long = sum(inv_vol_weights .* skipmissing(long_candidates.avg_actual))
            predicted_long = sum(inv_vol_weights .* long_candidates.avg_pred)

            # Keep equal-weight for comparison
            actual_long_equal_weight = mean(skipmissing(long_candidates.avg_actual))

            # SHORT basket metrics remain available for analysis.
            top_n_high_pred = min(top_n, nrow(period_scores))
            high_pred_candidates = copy(period_scores[1:top_n_high_pred, :])
            actual_high_pred = mean(skipmissing(high_pred_candidates.avg_actual))

            short_returns = Float64[]
            short_tickers = String[]
            short_indices = Int[]
            sector_counts = Dict{String, Int}()

            # Re-sort ascending for short selection (BUG FIX): picks bottom predictions, not top
            sort!(period_scores, :avg_pred, rev = false)

            for (idx, row) in enumerate(eachrow(period_scores))
                length(short_indices) >= BOTTOM_N_STOCKS && break

                ticker = row.ticker
                sector = get(SECTOR_MAP, ticker, "Unknown")
                count  = get(sector_counts, sector, 0)

                sector_limit = sector == "Unknown" ? 3 : 2
                if count < sector_limit
                    push!(short_indices, idx)
                    sector_counts[sector] = count + 1
                end
            end

            # Fallback: guarantee full basket if sector caps prevented it
            if length(short_indices) < BOTTOM_N_STOCKS
                existing = Set(period_scores.ticker[short_indices])
                for (idx, row) in enumerate(eachrow(period_scores))
                    row.ticker in existing && continue
                    push!(short_indices, idx)
                    push!(existing, row.ticker)
                    length(short_indices) >= BOTTOM_N_STOCKS && break
                end
            end
            short_candidates = period_scores[short_indices, :]

            for row in eachrow(short_candidates)
                isnan(row.avg_actual) && continue
                raw_short_ret       = -row.avg_actual - 0.001
                short_ret_with_stop = max(raw_short_ret, -STOP_LOSS_THRESHOLD)
                push!(short_returns, short_ret_with_stop)
                push!(short_tickers, row.ticker)
            end

            actual_alpha_short = isempty(short_returns) ? 0.0 : mean(short_returns)

            # Linear baseline: long portfolio
            sort!(period_scores, :linear_pred, rev = true)
            linear_n = min(top_n, nrow(period_scores))
            actual_equal_weight_long = mean(skipmissing(period_scores[1:linear_n, :].avg_actual))
            linear_tickers = collect(period_scores[1:linear_n, :ticker])
            linear_weights = fill(1.0 / linear_n, linear_n)

            # Linear baseline: short portfolio (sector-filtered)
            sort!(period_scores, :linear_pred, rev = false)
            linear_short_indices = Int[]
            linear_sector_counts = Dict{String, Int}()

            for (idx, row) in enumerate(eachrow(period_scores))
                length(linear_short_indices) >= BOTTOM_N_STOCKS && break

                ticker = row.ticker
                sector = get(SECTOR_MAP, ticker, "Unknown")
                count  = get(linear_sector_counts, sector, 0)

                sector_limit = sector == "Unknown" ? 3 : 2
                if count < sector_limit
                    push!(linear_short_indices, idx)
                    linear_sector_counts[sector] = count + 1
                end
            end

            # Fallback: guarantee full basket
            if length(linear_short_indices) < BOTTOM_N_STOCKS
                existing = Set(period_scores.ticker[linear_short_indices])
                for (idx, row) in enumerate(eachrow(period_scores))
                    row.ticker in existing && continue
                    push!(linear_short_indices, idx)
                    push!(existing, row.ticker)
                    length(linear_short_indices) >= BOTTOM_N_STOCKS && break
                end
            end
            linear_short_candidates = period_scores[linear_short_indices, :]

            linear_short_returns = Float64[]
            for row in eachrow(linear_short_candidates)
                isnan(row.avg_actual) && continue
                push!(linear_short_returns, max(-row.avg_actual - 0.001, -STOP_LOSS_THRESHOLD))
            end
            actual_equal_weight_short = isempty(linear_short_returns) ? 0.0 : mean(linear_short_returns)

            # Save holdings against the final table order by identity, not stale indices.
            long_indices = [findfirst(==(t), period_scores.ticker) for t in long_candidates.ticker]
            short_indices = [findfirst(==(t), period_scores.ticker) for t in short_candidates.ticker]
            chosen_alpha = actual_long

            # SPY return computed unconditionally (used for observability export;
            # also consumed by the short overlay block below when enabled).
            # Same execution timing as the strategy:
            #   entry  = open  of  test_start + 1 trading day
            #   exit   = close of  test_start + FORECAST_HORIZON trading days
            #   spy_ret = close[exit] / open[entry] - 1
            spy_exec_idx = searchsortedfirst(spy_data.date, step.test_start)
            if spy_exec_idx <= nrow(spy_data) && spy_data.date[spy_exec_idx] == step.test_start
                spy_exec_idx += 1  # next trading day = entry
                spy_exit_idx = spy_exec_idx + Features.FORECAST_HORIZON - 1
                if spy_exec_idx <= nrow(spy_data) && spy_exit_idx <= nrow(spy_data)
                    entry_px = spy_data.open[spy_exec_idx]
                    exit_px  = spy_data.close[spy_exit_idx]
                    spy_ret = (!ismissing(entry_px) && !ismissing(exit_px) &&
                               isfinite(entry_px) && isfinite(exit_px) && entry_px > 0.0) ?
                        exit_px / entry_px - 1.0 : NaN
                else
                    spy_ret = NaN
                end
            else
                spy_ret = NaN
            end

            if ENABLE_SHORT_OVERLAY
                if pred_spread >= CONFIDENCE_THRESHOLD
                    chosen_alpha = actual_alpha_short
                    short_count += 1
                else
                    chosen_alpha = spy_ret
                    spy_count += 1
                end

                if verbose && i <= 10
                    println("Step $i short_returns: $short_returns")
                    println("Step $i actual_alpha_short: $(round(actual_alpha_short * 100, digits = 3))%")
                    println("Step $i pred_spread: $(round(pred_spread, digits = 5))")
                    println("Step $i decision: $(pred_spread >= CONFIDENCE_THRESHOLD ? "SHORT" : "HOLD SPY")")
                    flush(stdout)
                end
            else
                if verbose && i <= 10
                    println("Step $i short_returns: $short_returns")
                    println("Step $i actual_alpha_short: $(round(actual_alpha_short * 100, digits = 3))%")
                    println("Step $i pred_spread: $(round(pred_spread, digits = 5))")
                    println("Step $i decision: base long-only (short overlay disabled)")
                    println("Step $i LONG  tickers: $(long_candidates.ticker)")
                    println("Step $i LONG  preds:   $(round.(long_candidates.avg_pred, digits=5))")
                    println("Step $i SHORT tickers: $(short_candidates.ticker)")
                    println("Step $i SHORT preds:   $(round.(short_candidates.avg_pred, digits=5))")
                    overlap_tickers = [t for t in short_candidates.ticker if t in long_candidates.ticker]
                    println("Step $i OVERLAP: $(isempty(overlap_tickers) ? "NONE" : overlap_tickers)")
                    flush(stdout)
                end
            end

            if !isfinite(actual_long)
                actual_long = 0.0
            end
            if !isfinite(chosen_alpha)
                chosen_alpha = 0.0
            end

            long_tickers = collect(long_candidates.ticker)
            # Each window liquidates at its exit close and holds cash until
            # next entry. Charge both traded legs, even for repeated names.
            long_turnover = sum(inv_vol_weights) + sum(inv_vol_weights .* (1 .+ long_candidates.avg_actual))
            linear_turnover = 1.0 + actual_equal_weight_long + 1.0
            net_alpha_long = actual_long - cost_rate * long_turnover
            net_alpha_linear = actual_equal_weight_long - cost_rate * linear_turnover
            if !ENABLE_SHORT_OVERLAY
                chosen_alpha = net_alpha_long
            end

            results[i] = (
                methodology             = "corrected-v2",
                date                    = step.test_start,
                train_start             = step.train_start,
                train_end               = step.train_end,
                test_start              = step.test_start,
                test_end                = step.test_end,
                tickers_long            = long_candidates.ticker,
                tickers_high_pred       = high_pred_candidates.ticker,
                tickers_short           = short_tickers,
                predicted_alpha         = predicted_long,
                actual_alpha_long       = actual_long,
                net_alpha_long          = net_alpha_long,
                net_alpha_linear        = net_alpha_linear,
                long_turnover           = long_turnover,
                linear_turnover         = linear_turnover,
                long_cost              = cost_rate * long_turnover,
                linear_cost            = cost_rate * linear_turnover,
                actual_alpha_high_pred  = actual_high_pred,
                actual_alpha_short      = actual_alpha_short,
                chosen_alpha            = chosen_alpha,
                regime_signal           = :base,
                actual_equal_weight_long      = actual_equal_weight_long,
                actual_equal_weight_short     = actual_equal_weight_short,
                best_decile_return      = best_decile_return,
                worst_decile_return     = worst_decile_return,
                decile_spread           = decile_spread,
                # Observability fields (added for drawdown investigation)
                period_scores_copy      = copy(period_scores),
                long_indices_copy       = copy(long_indices),
                short_indices_copy      = copy(short_indices),
                inv_vol_weights         = inv_vol_weights,
                actual_long_equal_weight_long = actual_long_equal_weight,
                ic                      = ic,
                rank_ic                 = rank_ic,
                spy_ret                 = spy_ret,
                # Validation fields (added for short-book fix verification)
                long_preds_copy         = collect(long_candidates.avg_pred),
                short_preds_copy        = collect(short_candidates.avg_pred),
                n_universe              = nrow(period_scores),
            )
            previous_long_tickers = long_tickers
            previous_long_weights = collect(inv_vol_weights)
            previous_linear_tickers = linear_tickers
            previous_linear_weights = linear_weights

        catch e
            println("\nError at step $i: $e")
            rethrow()
        end

        if !is_debug
            next!(p)
        end
    end

    if ENABLE_SHORT_OVERLAY
        println("Short steps: $(short_count) | SPY hold steps: $(spy_count)")
    else
        println("Short/SPY overlay disabled; chosen_alpha = base long-only alpha.")
    end

    return results, feature_importances_history, ic_history, rank_ic_history, decile_history
end

end
