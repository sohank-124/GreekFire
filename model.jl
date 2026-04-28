module Model

using MLJ, DecisionTree, MLJDecisionTreeInterface, MLJLinearModels, DataFrames, Statistics
import MLJ: predict

function train_and_predict(train_df::DataFrame, test_df::DataFrame, feature_cols::Vector{String})
    if isempty(feature_cols) || nrow(train_df) == 0
        return nothing, nothing, nothing, nothing, nothing
    end

    X_train = train_df[:, feature_cols]
    y_train = train_df.target

    X_matrix = try
        Matrix{Float64}(X_train)
    catch
        return nothing, nothing, nothing, nothing, nothing
    end

    valid_train_mask = .!any(isnan, X_matrix, dims=2)[:] .& .!isnan.(y_train)
    X_train_clean = X_train[valid_train_mask, :]
    y_train_clean = y_train[valid_train_mask]

    if isempty(y_train_clean) || all(isnan, y_train_clean)
        return nothing, nothing, nothing, nothing, nothing
    end

    n_subfeatures = max(3, floor(Int, sqrt(length(feature_cols))))
    rf_model = MLJDecisionTreeInterface.RandomForestRegressor(
        n_trees=100,
        n_subfeatures=n_subfeatures,
        max_depth=10
    )
    mach_rf = machine(rf_model, X_train_clean, y_train_clean)
    MLJ.fit!(mach_rf)

    lr_model = MLJLinearModels.LinearRegressor()
    mach_lr = machine(lr_model, X_train_clean, y_train_clean)
    MLJ.fit!(mach_lr)

    pred_rf = nothing
    pred_lr = nothing
    if nrow(test_df) > 0
        pred_rf, pred_lr = predict_with_machines(mach_rf, mach_lr, test_df, feature_cols, X_train_clean)
    end

    raw_importances = MLJ.feature_importances(mach_rf)
    importance_dict = Dict{String, Float64}()
    for item in raw_importances
        importance_dict[String(item[1])] = Float64(item[2])
    end

    return pred_rf, pred_lr, importance_dict, mach_rf, mach_lr
end

function predict_with_machines(mach_rf, mach_lr, test_df::DataFrame, feature_cols::Vector{String}, X_train_clean::DataFrame)
    X_test = test_df[:, feature_cols]
    X_test_filled = copy(X_test)

    for col in feature_cols
        col_data = X_train_clean[:, col]
        valid_data = col_data[(!isnan).(col_data)]
        m = isempty(valid_data) ? 0.0 : mean(valid_data)
        nan_indices = findall(x -> ismissing(x) || isnan(x), X_test_filled[:, col])
        for idx in nan_indices
            X_test_filled[idx, col] = m
        end
    end

    pred_rf = predict(mach_rf, X_test_filled)
    pred_lr = predict(mach_lr, X_test_filled)
    return pred_rf, pred_lr
end

end