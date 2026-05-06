module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

const TRAIN_WINDOW_DAYS = 252
const STEP_DAYS = 5
const TOP_N_STOCKS = 5

function run_walk_forward(all_stock_data, spy_data)
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

    # Get actual trading dates from data
    trading_dates = sort(unique(master_df.date))

    start_date = DataLoader.START_DATE
    end_date = today()

    # Build steps using trading dates only
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
    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)

    p = Progress(total_steps, desc="Walk-Forward Loop")

    feature_cols = String[]
    for col in names(master_df)
        if eltype(master_df[!, col]) <: Number && col ∉ ["date", "target", "ticker"]
            push!(feature_cols, String(col))
        end
    end

    for i in 1:total_steps
        step = steps[i]

        # CRITICAL LOOKAHEAD BIAS FIX: target[i] uses prices[i+5].
        # The last 5 trading days before train_end have targets computed
        # from future prices. Day(8) excludes them. NEVER remove this.
        combined_train = master_df[
            (master_df.date .>= step.train_start) .&
            (master_df.date .<= step.train_end - Day(8)), :]

        combined_test = master_df[
            (master_df.date .>= step.test_start) .&
            (master_df.date .<= step.test_end), :]

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

            # Predictions use all test days for better consensus ranking
            test_predictions = DataFrame(
                ticker        = combined_test.ticker,
                pred_ensemble = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
                actual        = combined_test.target
            )

            # Rank by average prediction across all test days
            pred_scores = combine(
                groupby(test_predictions, :ticker),
                :pred_ensemble => mean => :avg_pred
            )

            # Actual return uses ONLY the first trading day of the test window
            # This avoids overlapping forward return windows inflating alpha
            first_date = minimum(combined_test.date)
            first_day = combined_test[combined_test.date .== first_date, :]
            actual_scores = combine(
                groupby(first_day, :ticker),
                :target => mean => :avg_actual
            )

            period_scores = innerjoin(pred_scores, actual_scores, on=:ticker)
            dropmissing!(period_scores, :avg_pred)
            filter!(row -> !isnan(row.avg_pred) && !isnan(row.avg_actual), period_scores)

            if nrow(period_scores) == 0
                results[i] = nothing
                next!(p)
                continue
            end

            sort!(period_scores, :avg_pred, rev=false)
            top_n       = min(TOP_N_STOCKS, nrow(period_scores))
            best_stocks = period_scores[1:top_n, :]

            actual_portfolio_return    = mean(skipmissing(best_stocks.avg_actual))
            predicted_portfolio_return = mean(best_stocks.avg_pred)

            if isnan(actual_portfolio_return)
                results[i] = nothing
                next!(p)
                continue
            end

            results[i] = (
                date            = step.test_start,
                tickers         = best_stocks.ticker,
                predicted_alpha = predicted_portfolio_return,
                actual_alpha    = actual_portfolio_return
            )

        catch e
            println("\nError at step $i: $e")
            results[i] = nothing
        end

        next!(p)
    end

    return results, feature_importances_history
end

end