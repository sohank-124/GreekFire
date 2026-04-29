module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

# --- TRADING PARAMS ---
const TRAIN_WINDOW_DAYS = 252 # 1 Trading Year of lookback
const STEP_DAYS = 5           # Step forward 1 week at a time
const TOP_N_STOCKS = 5        # Buy the top 5 predicted stocks

function run_walk_forward(all_stock_data, spy_data)
    @info "Pre-computing feature matrices..."
    tickers = collect(keys(all_stock_data))
    all_features_vec = Vector{Pair{String, DataFrame}}(undef, length(tickers))
    
    # Use 8 threads to build features
    @threads for i in 1:length(tickers)
        ticker = tickers[i]
        all_features_vec[i] = ticker => Features.calculate_features(all_stock_data[ticker], spy_data)
    end
    
    @info "Concatenating all data for fast memory access..."
    all_dfs = DataFrame[]
    for (ticker, df) in all_features_vec
        if nrow(df) > 0
            df_copy = copy(df)
            df_copy.ticker .= ticker
            push!(all_dfs, df_copy)
        end
    end
    # Pre-build the master dataframe ONCE. This saves ~10 minutes of runtime.
    master_df = vcat(all_dfs...)

    start_date = DataLoader.START_DATE
    end_date = today()

    steps = []
    curr_start = start_date
    while curr_start + Day(TRAIN_WINDOW_DAYS) < end_date
        train_end = curr_start + Day(TRAIN_WINDOW_DAYS)
        test_start = train_end + Day(1)
        test_end = test_start + Day(STEP_DAYS - 1)
        push!(steps, (train_start = curr_start, train_end = train_end, test_start = test_start, test_end = test_end))
        curr_start = curr_start + Day(STEP_DAYS)
    end

    total_steps = length(steps)
    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)

    p = Progress(total_steps, desc="Walk-Forward Loop")

    # Extract feature columns once
    feature_cols = String[]
    for col in names(master_df)
        if eltype(master_df[!, col]) <: Number && col ∉ ["date", "target", "ticker"]
            push!(feature_cols, String(col))
        end
    end

    for i in 1:total_steps
        step = steps[i]

        # Fast logical slicing directly from the master dataframe
        combined_train = master_df[(master_df.date .>= step.train_start) .& (master_df.date .<= step.train_end), :]
        combined_test  = master_df[(master_df.date .>= step.test_start) .& (master_df.date .<= step.test_end), :]

        if isempty(feature_cols) || nrow(combined_train) == 0 || nrow(combined_test) == 0
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        try
            pred_rf, pred_lr, pred_xgb, importances, mach_rf, mach_lr, mach_xgb = Model.train_and_predict(
                combined_train, combined_test, feature_cols
            )

            if mach_xgb === nothing
                results[i] = nothing
                feature_importances_history[i] = nothing
                next!(p)
                continue
            end

            feature_importances_history[i] = importances

            test_predictions = DataFrame(
                ticker   = combined_test.ticker,
                pred_ensemble = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
                actual   = combined_test.target
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
            predicted_portfolio_return = mean(best_stocks.avg_pred)

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