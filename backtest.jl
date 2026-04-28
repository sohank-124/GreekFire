module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

# Retrain model every N months. Higher = faster but less adaptive.
const TRAIN_INTERVAL = 3

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

    # current_model stores fitted machines for reuse between training intervals
    # Sequential loop required — current_model is shared mutable state
    current_model = nothing

    for i in 1:total_steps
        step = steps[i]

        combined_train = DataFrame()
        combined_test = DataFrame()

        for (ticker, feat_df) in all_features
            train_slice = feat_df[(feat_df.date .>= start_date) .& (feat_df.date .<= step.train_end - Day(8)), :]
            test_slice  = feat_df[(feat_df.date .>= step.test_start) .& (feat_df.date .<= step.test_end), :]

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

        if nrow(combined_test) == 0 || ncol(combined_test) == 0
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        if nrow(combined_train) == 0 || ncol(combined_train) == 0
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        feature_cols = String[]
        for col in names(combined_train)
            if eltype(combined_train[!, col]) <: Number
                if col != "date" && col != "target" && col != "ticker"
                    push!(feature_cols, String(col))
                end
            end
        end

        if isempty(feature_cols)
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        try
            should_retrain = current_model === nothing || mod(i, TRAIN_INTERVAL) == 0

            if should_retrain
                pred_rf, pred_lr, importances, mach_rf, mach_lr = Model.train_and_predict(
                    combined_train, combined_test, feature_cols
                )

                if mach_rf === nothing
                    results[i] = nothing
                    feature_importances_history[i] = nothing
                    next!(p)
                    continue
                end

                X_train_clean = begin
                    X = combined_train[:, feature_cols]
                    X_mat = Matrix{Float64}(X)
                    valid = .!any(isnan, X_mat, dims=2)[:] .& .!isnan.(combined_train.target)
                    X[valid, :]
                end

                current_model = (
                    mach_rf       = mach_rf,
                    mach_lr       = mach_lr,
                    feature_cols  = feature_cols,
                    importances   = importances,
                    X_train_clean = X_train_clean
                )

                # Store predictions for this step
                step_pred_rf = pred_rf
                step_pred_lr = pred_lr
            else
                # Reuse trained machines, predict fresh on this step's test data
                step_pred_rf, step_pred_lr = Model.predict_with_machines(
                    current_model.mach_rf,
                    current_model.mach_lr,
                    combined_test,
                    current_model.feature_cols,
                    current_model.X_train_clean
                )
            end

            feature_importances_history[i] = current_model.importances

            test_predictions = DataFrame(
                ticker  = combined_test.ticker,
                pred_rf = step_pred_rf,
                pred_lr = step_pred_lr,
                actual  = combined_test.target
            )

            stock_ranks    = groupby(test_predictions, :ticker)
            monthly_scores = combine(stock_ranks, :pred_rf => mean => :avg_pred, :actual => mean => :avg_actual)

            preds       = monthly_scores.avg_pred
            clean_preds = filter(!isnan, preds)

            if isempty(clean_preds)
                results[i] = nothing
                next!(p)
                continue
            end

            max_val        = maximum(clean_preds)
            best_idx       = findfirst(x -> x == max_val, preds)
            best_stock_row = monthly_scores[best_idx, :]

            if ismissing(best_stock_row[:avg_actual]) || isnan(best_stock_row[:avg_actual])
                results[i] = nothing
                next!(p)
                continue
            end

            results[i] = (
                date            = step.test_start,
                ticker          = best_stock_row[:ticker],
                predicted_alpha = best_stock_row[:avg_pred],
                actual_alpha    = best_stock_row[:avg_actual]
            )

        catch e
            println("========== ERROR AT STEP $i ==========")
            println("Error type: $(typeof(e))")
            println("Error message: $e")
            bt = catch_backtrace()
            for (idx, frame) in enumerate(bt)
                println("  [$idx] $frame")
                idx >= 10 && break
            end
            println("Train shape: $(size(combined_train)), Test shape: $(size(combined_test))")
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
    end_date   = today()

    steps = []
    curr_start = start_date
    while curr_start < end_date && length(steps) < num_steps
        curr_end         = lastdayofmonth(curr_start)
        next_month_start = curr_end + Day(1)
        next_month_end   = lastdayofmonth(next_month_start)
        push!(steps, (train_end = curr_end, test_start = next_month_start, test_end = next_month_end))
        curr_start = next_month_start
    end

    total_steps = length(steps)
    println("DEBUG: Running $total_steps steps")

    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)
    current_model = nothing

    for i in 1:total_steps
        println("DEBUG Step $i: $(steps[i].test_start)")
        step = steps[i]

        combined_train = DataFrame()
        combined_test  = DataFrame()

        for (ticker, feat_df) in all_features
            train_slice = feat_df[(feat_df.date .>= start_date) .& (feat_df.date .<= step.train_end - Day(8)), :]
            test_slice  = feat_df[(feat_df.date .>= step.test_start) .& (feat_df.date .<= step.test_end), :]

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

        println("  Train rows: $(nrow(combined_train)), Test rows: $(nrow(combined_test))")

        if nrow(combined_train) == 0 || nrow(combined_test) == 0
            results[i] = nothing
            continue
        end

        feature_cols = String[]
        for col in names(combined_train)
            if eltype(combined_train[!, col]) <: Number
                if col != "date" && col != "target" && col != "ticker"
                    push!(feature_cols, String(col))
                end
            end
        end

        should_retrain = current_model === nothing || mod(i, TRAIN_INTERVAL) == 0

        if should_retrain
            pred_rf, pred_lr, importances, mach_rf, mach_lr = Model.train_and_predict(
                combined_train, combined_test, feature_cols
            )
            if mach_rf === nothing
                results[i] = nothing
                continue
            end
            X_train_clean = begin
                X = combined_train[:, feature_cols]
                X_mat = Matrix{Float64}(X)
                valid = .!any(isnan, X_mat, dims=2)[:] .& .!isnan.(combined_train.target)
                X[valid, :]
            end
            current_model = (
                mach_rf       = mach_rf,
                mach_lr       = mach_lr,
                feature_cols  = feature_cols,
                importances   = importances,
                X_train_clean = X_train_clean
            )
            step_pred_rf = pred_rf
            step_pred_lr = pred_lr
        else
            step_pred_rf, step_pred_lr = Model.predict_with_machines(
                current_model.mach_rf, current_model.mach_lr,
                combined_test, current_model.feature_cols, current_model.X_train_clean
            )
        end

        feature_importances_history[i] = current_model.importances

        test_predictions = DataFrame(
            ticker  = combined_test.ticker,
            pred_rf = step_pred_rf,
            pred_lr = step_pred_lr,
            actual  = combined_test.target
        )

        stock_ranks    = groupby(test_predictions, :ticker)
        monthly_scores = combine(stock_ranks, :pred_rf => mean => :avg_pred, :actual => mean => :avg_actual)
        preds          = monthly_scores.avg_pred
        clean_preds    = filter(!isnan, preds)

        if isempty(clean_preds)
            results[i] = nothing
            continue
        end

        max_val        = maximum(clean_preds)
        best_idx       = findfirst(x -> x == max_val, preds)
        best_stock_row = monthly_scores[best_idx, :]

        results[i] = (
            date            = step.test_start,
            ticker          = best_stock_row[:ticker],
            predicted_alpha = best_stock_row[:avg_pred],
            actual_alpha    = best_stock_row[:avg_actual]
        )

        println("  Result: $(results[i])")
    end

    return results, feature_importances_history
end

end