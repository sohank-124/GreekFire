module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

function run_walk_forward(all_stock_data, spy_data)
    @info "Pre-computing feature matrices..."
    all_features = Dict{String, DataFrame}()
    for (ticker, df) in all_stock_data
        all_features[ticker] = Features.calculate_features(df, spy_data)
    end
    
    start_date = DataLoader.START_DATE
    end_date = today()
    
    steps = []
    curr_start = start_date
    while curr_start < end_date
        curr_end = lastdayofmonth(curr_start)
        next_month_start = curr_end + Day(1)
        next_month_end = lastdayofmonth(next_month_start)
        push!(steps, (train_end = curr_end, test_start = next_month_start, test_end = next_month_end))
        curr_start = next_month_start
    end
    
    total_steps = length(steps)
    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)
    
    p = Progress(total_steps, desc="Walk-Forward Loop")
    
    Threads.@threads for i in 1:total_steps
        step = steps[i]
        
        combined_train = DataFrame()
        combined_test = DataFrame()
        
        for (ticker, feat_df) in all_features
            # FIX: subtract Day(8) to exclude last 5 trading days whose targets
            # were computed using prices AFTER the cutoff — this was the lookahead bias
            train_slice = feat_df[(feat_df.date .>= start_date) .& (feat_df.date .<= step.train_end - Day(8)), :]
            test_slice = feat_df[(feat_df.date .>= step.test_start) .& (feat_df.date .<= step.test_end), :]
            
            if !isempty(train_slice)
                train_slice = copy(train_slice)
                train_slice.ticker = fill(ticker, nrow(train_slice))
                append!(combined_train, train_slice)
            end
            
            if !isempty(test_slice)
                test_slice = copy(test_slice)
                test_slice.ticker = fill(ticker, nrow(test_slice))
                append!(combined_test, test_slice)
            end
        end
        
        feature_cols = String[]
        for col in names(combined_train)
            if eltype(combined_train[!, col]) <: Number
                if col != "date" && col != "target" && col != "ticker"
                    push!(feature_cols, String(col))
                end
            end
        end
        
        try
            total_rows_before_filter = nrow(combined_train)
            test_rows_before_filter = nrow(combined_test)
            
            if mod(i, 10) == 1 || total_rows_before_filter < 100 || test_rows_before_filter == 0
                tickers_with_train = String[]
                tickers_with_test = String[]
                for (ticker, feat_df) in all_features
                    train_slice = feat_df[(feat_df.date .>= start_date) .& (feat_df.date .<= step.train_end - Day(8)), :]
                    test_slice = feat_df[(feat_df.date .>= step.test_start) .& (feat_df.date .<= step.test_end), :]
                    !isempty(train_slice) && push!(tickers_with_train, ticker)
                    !isempty(test_slice) && push!(tickers_with_test, ticker)
                end
                println("  [Step $i] Train rows: $total_rows_before_filter, Test rows: $test_rows_before_filter")
                println("  [Step $i] Train tickers: $(length(tickers_with_train)), Test tickers: $(length(tickers_with_test))")
                println("  [Step $i] Train dates: $(step.train_end - Day(8)), Test: $(step.test_start) to $(step.test_end)")
                if test_rows_before_filter == 0
                    println("  [Step $i] WARNING: Test window has NO data - check date range in feature DataFrame")
                end
            end
            
            if isempty(feature_cols)
                @warn "Step $i: No feature columns found. DataFrame columns: $(names(combined_train))"
                results[i] = nothing
                next!(p)
                continue
            end
            
            X_train = combined_train[:, feature_cols]
            y_train = combined_train.target
            X_matrix = try
                Matrix{Float64}(X_train)
            catch err
                @warn "Step $i: Matrix conversion failed: $err"
                results[i] = nothing
                next!(p)
                continue
            end
            
            valid_mask = vec(.!any(isnan, X_matrix, dims=2)) .& .!isnan.(y_train)
            rows_after_nan_drop = sum(valid_mask)
            
            if mod(i, 10) == 1 || rows_after_nan_drop < 50
                println("  [Step $i] Feature cols: $(length(feature_cols)), Rows after NaN drop: $rows_after_nan_drop")
            end
            
            pred_rf, pred_lr, importances = Model.train_and_predict(combined_train, combined_test, feature_cols)
            
            if pred_rf === nothing
                results[i] = nothing
                next!(p)
                continue
            end
            
            feature_importances_history[i] = importances
            
            test_predictions = DataFrame(
                ticker = combined_test.ticker,
                pred_rf = pred_rf,
                pred_lr = pred_lr,
                actual = combined_test.target
            )
            
            stock_ranks = groupby(test_predictions, :ticker)
            monthly_scores = combine(stock_ranks, :pred_rf => mean => :avg_pred, :actual => mean => :avg_actual)
            
            preds = monthly_scores.avg_pred
            clean_preds = filter(!isnan, preds)
            
            if isempty(clean_preds)
                results[i] = nothing
                next!(p)
                continue
            end
            
            max_val = maximum(clean_preds)
            best_idx = findfirst(x -> x == max_val, preds)
            best_stock_row = monthly_scores[best_idx, :]
            
            if ismissing(best_stock_row[:avg_actual]) || isnan(best_stock_row[:avg_actual])
                results[i] = nothing
                next!(p)
                continue
            end
            
            results[i] = (
                date = step.test_start,
                ticker = best_stock_row[:ticker],
                predicted_alpha = best_stock_row[:avg_pred],
                actual_alpha = best_stock_row[:avg_actual]
            )
            
        catch e
            println("========== ERROR AT STEP $i ==========")
            println("Error type: $(typeof(e))")
            println("Error message: $e")
            println("Stack trace:")
            bt = catch_backtrace()
            for (idx, frame) in enumerate(bt)
                println("  [$idx] $frame")
                idx >= 10 && break
            end
            println("Train DataFrame shape before error: $(size(combined_train))")
            println("Test DataFrame shape: $(size(combined_test))")
            println("Feature columns found: $(length(feature_cols))")
            println("======================================")
            results[i] = nothing
        end
        next!(p)
    end
    
    return results, feature_importances_history
end

function run_walk_forward_debug(all_stock_data, spy_data, num_steps::Int=5)
    @info "Pre-computing feature matrices..."
    all_features = Dict{String, DataFrame}()
    for (ticker, df) in all_stock_data
        all_features[ticker] = Features.calculate_features(df, spy_data)
    end
    
    start_date = DataLoader.START_DATE
    end_date = today()
    
    steps = []
    curr_start = start_date
    while curr_start < end_date && length(steps) < num_steps
        curr_end = lastdayofmonth(curr_start)
        next_month_start = curr_end + Day(1)
        next_month_end = lastdayofmonth(next_month_start)
        push!(steps, (train_end = curr_end, test_start = next_month_start, test_end = next_month_end))
        curr_start = next_month_start
    end
    
    total_steps = length(steps)
    println("DEBUG: Running $total_steps steps: first=$(steps[1].test_start), last=$(steps[end].test_start)")
    
    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)
    
    for i in 1:total_steps
        println("DEBUG Step $i of $total_steps: $(steps[i].test_start)")
        step = steps[i]
        
        combined_train = DataFrame()
        combined_test = DataFrame()
        
        for (ticker, feat_df) in all_features
            # FIX: same lookahead fix applied to debug mode
            train_slice = feat_df[(feat_df.date .>= start_date) .& (feat_df.date .<= step.train_end - Day(8)), :]
            test_slice = feat_df[(feat_df.date .>= step.test_start) .& (feat_df.date .<= step.test_end), :]
            
            if !isempty(train_slice)
                train_slice = copy(train_slice)
                train_slice.ticker = fill(ticker, nrow(train_slice))
                append!(combined_train, train_slice)
            end
            
            if !isempty(test_slice)
                test_slice = copy(test_slice)
                test_slice.ticker = fill(ticker, nrow(test_slice))
                append!(combined_test, test_slice)
            end
        end
        
        total_rows = nrow(combined_train)
        println("  DEBUG: Combined train rows: $total_rows, test rows: $(nrow(combined_test))")
        
        feature_cols = String[]
        for col in names(combined_train)
            if eltype(combined_train[!, col]) <: Number
                if col != "date" && col != "target" && col != "ticker"
                    push!(feature_cols, String(col))
                end
            end
        end
        
        println("  DEBUG: Feature columns: $(length(feature_cols))")
        
        pred_rf, pred_lr, importances = Model.train_and_predict(combined_train, combined_test, feature_cols)
        
        if pred_rf === nothing
            results[i] = nothing
            continue
        end
        
        feature_importances_history[i] = importances
        
        test_predictions = DataFrame(
            ticker = combined_test.ticker,
            pred_rf = pred_rf,
            pred_lr = pred_lr,
            actual = combined_test.target
        )
        
        stock_ranks = groupby(test_predictions, :ticker)
        monthly_scores = combine(stock_ranks, :pred_rf => mean => :avg_pred, :actual => mean => :avg_actual)
        
        preds = monthly_scores.avg_pred
        clean_preds = filter(!isnan, preds)
        
        if isempty(clean_preds)
            results[i] = nothing
            continue
        end
        
        max_val = maximum(clean_preds)
        best_idx = findfirst(x -> x == max_val, preds)
        best_stock_row = monthly_scores[best_idx, :]
        
        results[i] = (
            date = step.test_start,
            ticker = best_stock_row[:ticker],
            predicted_alpha = best_stock_row[:avg_pred],
            actual_alpha = best_stock_row[:avg_actual]
        )
        
        println("  DEBUG: Result - $(results[i])")
    end
    
    return results, feature_importances_history
end

end