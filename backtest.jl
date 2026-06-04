module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

const TRAIN_WINDOW_DAYS = 252
const STEP_DAYS = 5
const TOP_N_STOCKS = 5
const BOTTOM_N_STOCKS = 5
const STOP_LOSS_THRESHOLD = 0.07

const SECTOR_MAP = Dict(
    "AAPL" => "Tech",     "MSFT" => "Tech",     "GOOGL" => "Tech",
    "AMZN" => "Tech",     "NVDA" => "Tech",      "META" => "Tech",
    "AMD"  => "Tech",     "INTC" => "Tech",      "ADBE" => "Tech",
    "CSCO" => "Tech",     "TXN" => "Tech",       "QCOM" => "Tech",
    "AVGO" => "Tech",     "ACN" => "Tech",       "IBM" => "Tech",
    "HPQ"  => "Tech",     "ORCL" => "Tech",
    "JPM"  => "Finance",  "BAC" => "Finance",    "GS" => "Finance",
    "AXP"  => "Finance",  "BLK" => "Finance",    "SCHW" => "Finance",
    "USB"  => "Finance",  "PNC" => "Finance",    "COF" => "Finance",
    "MET"  => "Finance",  "AIG" => "Finance",    "TRV" => "Finance",
    "PGR"  => "Finance",  "MS" => "Finance",     "BK" => "Finance",
    "STT"  => "Finance",  "CI" => "Finance",     "UNH" => "Finance",
    "HUM"  => "Finance",  "AFL" => "Finance",    "PRU" => "Finance",
    "ALL"  => "Finance",  "BEN" => "Finance",    "TROW" => "Finance",
    "IVZ"  => "Finance",  "NTRS" => "Finance",   "DFS" => "Finance",
    "SYF"  => "Finance",  "CINF" => "Finance",   "L" => "Finance",
    "BRK-B"=> "Finance",  "V" => "Finance",      "MA" => "Finance",
    "C"    => "Finance",
    "XOM"  => "Energy",   "CVX" => "Energy",     "COP" => "Energy",
    "EOG"  => "Energy",   "SLB" => "Energy",     "HAL" => "Energy",
    "NOV"  => "Energy",   "OXY" => "Energy",     "MRO" => "Energy",
    "DVN"  => "Energy",   "CHK" => "Energy",     "RRC" => "Energy",
    "SWN"  => "Energy",   "EQT" => "Energy",     "AR" => "Energy",
    "PXD"  => "Energy",
    "JNJ"  => "Health",   "PFE" => "Health",     "MRK" => "Health",
    "AMGN" => "Health",   "BMY" => "Health",     "ABT" => "Health",
    "MDT"  => "Health",   "TMO" => "Health",     "DHR" => "Health",
    "DIS"  => "Media",    "CMCSA" => "Media",
    "KO"   => "Consumer", "WMT" => "Consumer",   "PG" => "Consumer",
    "PEP"  => "Consumer", "MCD" => "Consumer",   "NKE" => "Consumer",
    "TGT"  => "Consumer", "HD" => "Consumer",    "EMR" => "Industrial",
    "HON"  => "Industrial","BA" => "Industrial", "MMM" => "Industrial",
    "CAT"  => "Industrial","GE" => "Industrial", "LMT" => "Industrial",
    "GD"   => "Industrial","NOC" => "Industrial","UPS" => "Industrial",
    "FDX"  => "Industrial","CMI" => "Industrial","DE" => "Industrial",
    "F"    => "Auto",
    "NEE"  => "Utilities",
    "T"    => "Telecom",   "VZ" => "Telecom",
    "LUV"  => "Airlines",  "DAL" => "Airlines",  "UAL" => "Airlines",
)

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
    master_df = vcat(all_dfs...)

    trading_dates = sort(unique(master_df.date))
    start_date = DataLoader.START_DATE
    end_date = today()

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
    
    short_count = 0
    spy_count = 0

    is_debug = max_steps !== nothing
    if is_debug
        println("DEBUG MODE: Running $total_steps steps without progress bar")
    end
    p = Progress(total_steps, desc="Walk-Forward Loop")

    feature_cols = String[]
    for col in names(master_df)
        if eltype(master_df[!, col]) <: Number && col ∉ ["date", "target", "ticker"]
            push!(feature_cols, String(col))
        end
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

    for i in 1:total_steps
        step = steps[i]

        # Compute SPY return for the current step (regardless of shorting decision)
        spy_date_start = spy_data[spy_data.date .== step.test_start, :close]
        spy_date_end   = spy_data[spy_data.date .== step.test_end, :close]
        spy_ret = if isempty(spy_date_start) || isempty(spy_date_end)
            0.0
        else
            spy_date_end[1] / spy_date_start[1] - 1.0
        end
        
        # Point-in-time universe: only use tickers active in S&P 500 on test_start
        active_set = pit_cache[step.test_start]
        # Cap universe size to 200 tickers for performance
        if length(active_set) > 200
            sorted_tickers = sort(collect(active_set))
            active_set = Set(sorted_tickers[1:200])
        end

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
            pred_rf, pred_lr, pred_xgb, importances, mach_rf, mach_lr, mach_xgb =
                Model.train_and_predict(combined_train, combined_test, feature_cols)

            if mach_xgb === nothing
                results[i] = nothing
                feature_importances_history[i] = nothing
                next!(p)
                continue
            end

            feature_importances_history[i] = importances

            period_scores = DataFrame(
                ticker     = combined_test.ticker,
                avg_pred   = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
                avg_actual = combined_test.target
            )
            dropmissing!(period_scores, [:avg_pred, :avg_actual])
            filter!(row -> !isnan(row.avg_pred) && !isnan(row.avg_actual), period_scores)

            if i <= 10
                open(joinpath(pwd(), "debug.txt"), "a") do f
                    println(f, "=== Step $i ===")
                    println(f, "  period_scores rows: $(nrow(period_scores))")
                    println(f, "  pred_spread: $(maximum(period_scores.avg_pred) - minimum(period_scores.avg_pred))")
                    println(f, "  spy_ret: $spy_ret")
                end
            end

            if nrow(period_scores) == 0
                results[i] = nothing
                next!(p)
                continue
            end
            
            # LONG: rev=false (inverted signal — our working strategy)
            sort!(period_scores, :avg_pred, rev=false)
            top_n_false       = min(TOP_N_STOCKS, nrow(period_scores))
            best_stocks_false = period_scores[1:top_n_false, :]
            actual_false      = mean(skipmissing(best_stocks_false.avg_actual))
            predicted_false   = mean(best_stocks_false.avg_pred)

            # SHORT: highest predicted = worst actual (sector filtered, stop lossed)
            sort!(period_scores, :avg_pred, rev=true)
            
            # Spread-based confidence: only short if model sees meaningful spread
            pred_spread = maximum(period_scores.avg_pred) - minimum(period_scores.avg_pred)
            
            # Build short candidates regardless (needed for :short metric)
            short_returns = Float64[]
            short_tickers = String[]
            short_candidates = DataFrame()
            sector_counts = Dict{String, Int}()
            for row in eachrow(period_scores)
                nrow(short_candidates) >= BOTTOM_N_STOCKS && break
                ticker = row.ticker
                sector = get(SECTOR_MAP, ticker, "Unknown")
                count  = get(sector_counts, sector, 0)
                if count < 2
                    push!(short_candidates, row)
                    sector_counts[sector] = count + 1
                end
            end

            for row in eachrow(short_candidates)
                isnan(row.avg_actual) && continue
                raw_short_ret      = -row.avg_actual - 0.001
                short_ret_with_stop = max(raw_short_ret, -STOP_LOSS_THRESHOLD)
                push!(short_returns, short_ret_with_stop)
                push!(short_tickers, row.ticker)
            end

            actual_alpha_short = isempty(short_returns) ? 0.0 : mean(short_returns)
            
            if i <= 10
                println("Step $i short_returns: $short_returns")
                println("Step $i actual_alpha_short: $(round(actual_alpha_short * 100, digits=3))%")
                println("Step $i pred_spread: $(round(pred_spread, digits=5))")
                println("Step $i decision: $(pred_spread >= 0.01 ? "SHORT" : "HOLD SPY")")
                flush(stdout)
            end
            
            # chosen_alpha: short when confident, otherwise hold SPY
            if pred_spread >= 0.01
                chosen_alpha = actual_alpha_short
                short_count += 1
            else
                chosen_alpha = spy_ret
                spy_count += 1
            end

            # rev=true for metrics comparison
            sort!(period_scores, :avg_pred, rev=true)
            top_n_true       = min(TOP_N_STOCKS, nrow(period_scores))
            best_stocks_true = period_scores[1:top_n_true, :]
            actual_true      = mean(skipmissing(best_stocks_true.avg_actual))
            predicted_true   = mean(best_stocks_true.avg_pred)

            if isnan(actual_false) && isnan(actual_true)
                results[i] = nothing
                next!(p)
                continue
            end

            results[i] = (
                date               = step.test_start,
                tickers_false      = best_stocks_false.ticker,
                tickers_true       = best_stocks_true.ticker,
                tickers_short      = short_tickers,
                predicted_alpha    = predicted_false,
                actual_alpha_false = actual_false,
                actual_alpha_true  = actual_true,
                actual_alpha_short = actual_alpha_short,
                chosen_alpha       = chosen_alpha,
                regime_signal      = :base
            )

        catch e
            println("\nError at step $i: $e")
            results[i] = nothing
        end

        if !is_debug
            next!(p)
        end
    end

    println("Short steps: $(short_count) | SPY hold steps: $(spy_count)")

    return results, feature_importances_history
end

end