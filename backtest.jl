module Backtest

using DataFrames, Dates, Statistics, StatsBase, ProgressMeter, Random
using Base.Threads

include("model.jl")
include("features.jl")
include("data_loader.jl")

const TRAIN_WINDOW_DAYS = 252
const STEP_DAYS = 5
const TOP_N_STOCKS = 10
const BOTTOM_N_STOCKS = 5
const STOP_LOSS_THRESHOLD = 0.07

# Keep the old short/SPY overlay available, but off by default.
const ENABLE_SHORT_OVERLAY = false
const CONFIDENCE_THRESHOLD = 0.01

const NON_FEATURE_COLS = Set(["date", "target", "ticker"])

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



function run_walk_forward(all_stock_data, spy_data; max_steps=nothing)
    @info "Pre-computing feature matrices..."
    tickers = collect(keys(all_stock_data))
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
    sort!(master_df, [:date, :ticker])

    all_dates = sort(unique(master_df.date))

    @info "Computing cross-sectional rank_ret_20..."
    master_df[!, :rank_ret_20] = fill(0.5, nrow(master_df))
    for d in all_dates
        mask = master_df.date .== d
        if any(mask)
            master_df[mask, :rank_ret_20] = percentile_rank(master_df[mask, :ret_20])
        end
    end

    feature_cols = String[]
    for col in names(master_df)
        if (eltype(master_df[!, col]) <: Number) && !(col in NON_FEATURE_COLS)
            push!(feature_cols, String(col))
        end
    end

    if isempty(feature_cols)
        error("No numeric feature columns found.")
    end

    @info "Standardizing features cross-sectionally (percentile rank + clamp)..."
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
                sub_df[:, Symbol(col)] = percentile_rank(sub_df[:, Symbol(col)])
            end
        end
    end

    # Remove ret_20 from feature columns (redundant with rank_ret_20)
    filter!(x -> x != "ret_20", feature_cols)

    trading_dates = all_dates
    start_date = DataLoader.START_DATE

    steps = []
    curr_idx = searchsortedfirst(trading_dates, start_date + Day(TRAIN_WINDOW_DAYS))
    while curr_idx + STEP_DAYS <= length(trading_dates)
        train_end   = trading_dates[curr_idx]
        test_start  = trading_dates[curr_idx + 1]
        test_end    = trading_dates[min(curr_idx + STEP_DAYS, length(trading_dates))]
        train_start = trading_dates[max(1, curr_idx - TRAIN_WINDOW_DAYS)]

        push!(steps, (
            train_start = train_start,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end
        ))
        curr_idx += STEP_DAYS
    end

    total_steps = length(steps)
    if max_steps !== nothing && max_steps < total_steps
        total_steps = max_steps
    end

    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)
    ic_history = Float64[]
    rank_ic_history = Float64[]
    decile_history = Vector{Vector{Float64}}()

    is_debug = max_steps !== nothing
    if is_debug
        println("DEBUG MODE: Running $total_steps steps without progress bar")
    end
    p = Progress(total_steps, desc = "Walk-Forward Loop")

    feature_syms = Symbol.(feature_cols)
    short_count = 0
    spy_count = 0

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
            active = DataLoader.get_sp500_constituents(date)
            pit_cache[date] = Set(DataLoader.resolve_ticker(t) for t in active)
        end
    end

    rng = MersenneTwister(42)

    for i in 1:total_steps
        step = steps[i]

        # Point-in-time universe: only use tickers active in S&P 500 on test_start
        active_set = pit_cache[step.test_start]

        # CRITICAL LOOKAHEAD BIAS FIX: Day(8) excludes last 5 trading days
        # whose targets use prices AFTER train_end. NEVER remove this.
        combined_train = master_df[
            (master_df.date .>= step.train_start) .&
            (master_df.date .<= step.train_end - Day(8)) .&
            (master_df.ticker .∈ Ref(active_set)), :]

        combined_test = master_df[
            (master_df.date .== step.test_start) .&
            (master_df.ticker .∈ Ref(active_set)), :]

        if i == 1
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
            _t_total = @elapsed (pred_rf, pred_lr, pred_xgb, importances, _mach_rf, mach_lr, mach_xgb) =
                Model.train_and_predict(combined_train, combined_test, feature_cols)
            println("[BENCHMARK] train_and_predict total: $(round(_t_total; digits=3))s")

            # Fall back to RF-only if XGB fails
            ensemble_pred = if mach_xgb === nothing
                pred_rf
            else
                0.5 .* pred_rf .+ 0.5 .* pred_xgb
            end

            feature_importances_history[i] = importances

            linear_pred = mach_lr === nothing ? pred_rf : pred_lr

            period_scores = DataFrame(
                ticker      = combined_test.ticker,
                avg_pred    = ensemble_pred,
                linear_pred = linear_pred,
                avg_actual  = combined_test.target
            )

            dropmissing!(period_scores, [:avg_pred, :avg_actual, :linear_pred])
            filter!(row -> !isnan(row.avg_pred) && !isnan(row.avg_actual) && !isnan(row.linear_pred), period_scores)

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

            if i <= 10
                open(joinpath(pwd(), "debug.txt"), "a") do f
                    println(f, "=== Step $i ===")
                    println(f, "  period_scores rows: $(nrow(period_scores))")
                    println(f, "  pred_spread: $pred_spread")
                end
            end

            # LONG portfolio: pick top TOP_N_STOCKS, sector-capped (mirrors short basket).
            # Preserves current sort order; caps each sector at 2 (3 for Unknown) with a fallback fill.
            long_indices        = Int[]
            long_sector_counts  = Dict{String, Int}()
            for (idx, row) in enumerate(eachrow(period_scores))
                length(long_indices) >= TOP_N_STOCKS && break
                sector = get(SECTOR_MAP, row.ticker, "Unknown")
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
            long_candidates = period_scores[long_indices, :]
            actual_long    = mean(skipmissing(long_candidates.avg_actual))
            predicted_long = mean(long_candidates.avg_pred)

            # SHORT basket metrics remain available for analysis.
            top_n_high_pred = min(TOP_N_STOCKS, nrow(period_scores))
            high_pred_candidates = copy(period_scores[1:top_n_high_pred, :])
            actual_high_pred = mean(skipmissing(high_pred_candidates.avg_actual))

            short_returns = Float64[]
            short_tickers = String[]
            short_indices = Int[]
            sector_counts = Dict{String, Int}()

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
            linear_n = min(TOP_N_STOCKS, nrow(period_scores))
            actual_equal_weight_long = mean(skipmissing(period_scores[1:linear_n, :].avg_actual))

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

            chosen_alpha = actual_long

            # SPY return computed unconditionally (used for observability export;
            # also consumed by the short overlay block below when enabled).
            spy_date_start = spy_data[spy_data.date .== step.test_start, :close]
            spy_date_end   = spy_data[spy_data.date .== step.test_end, :close]
            spy_ret = if isempty(spy_date_start) || isempty(spy_date_end)
                0.0
            else
                spy_date_end[1] / spy_date_start[1] - 1.0
            end

            if ENABLE_SHORT_OVERLAY
                if pred_spread >= CONFIDENCE_THRESHOLD
                    chosen_alpha = actual_alpha_short
                    short_count += 1
                else
                    chosen_alpha = spy_ret
                    spy_count += 1
                end

                if i <= 10
                    println("Step $i short_returns: $short_returns")
                    println("Step $i actual_alpha_short: $(round(actual_alpha_short * 100, digits = 3))%")
                    println("Step $i pred_spread: $(round(pred_spread, digits = 5))")
                    println("Step $i decision: $(pred_spread >= CONFIDENCE_THRESHOLD ? "SHORT" : "HOLD SPY")")
                    flush(stdout)
                end
            else
                if i <= 10
                    println("Step $i short_returns: $short_returns")
                    println("Step $i actual_alpha_short: $(round(actual_alpha_short * 100, digits = 3))%")
                    println("Step $i pred_spread: $(round(pred_spread, digits = 5))")
                    println("Step $i decision: base long-only (short overlay disabled)")
                    flush(stdout)
                end
            end

            if !isfinite(actual_long)
                actual_long = 0.0
            end
            if !isfinite(chosen_alpha)
                chosen_alpha = 0.0
            end

            results[i] = (
                date                    = step.test_start,
                tickers_long            = long_candidates.ticker,
                tickers_high_pred       = high_pred_candidates.ticker,
                tickers_short           = short_tickers,
                predicted_alpha         = predicted_long,
                actual_alpha_long       = actual_long,
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
                ic                      = ic,
                rank_ic                 = rank_ic,
                spy_ret                 = spy_ret,
            )

        catch e
            println("\nError at step $i: $e")
            results[i] = nothing
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