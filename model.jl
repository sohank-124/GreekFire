module Model

using MLJ, DecisionTree, MLJDecisionTreeInterface, MLJLinearModels, XGBoost, DataFrames, Statistics
import MLJ: predict

function train_and_predict(train_df::DataFrame, test_df::DataFrame, feature_cols::Vector{String})
    X_train = train_df[:, feature_cols]
    y_train = train_df.target

    X_matrix = try
        Matrix{Float64}(X_train)
    catch
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing
    end

    valid_train_mask = .!any(isnan, X_matrix, dims=2)[:] .& .!isnan.(y_train)
    X_train_clean = X_train[valid_train_mask, :]
    y_train_clean = y_train[valid_train_mask]

    if isempty(y_train_clean)
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing
    end

    n_subfeatures = max(3, floor(Int, sqrt(length(feature_cols))))
    
    # 1. Random Forest (Reduced trees and depth for speed)
    rf_model = MLJDecisionTreeInterface.RandomForestRegressor(
        n_trees=50,  
        n_subfeatures=n_subfeatures,
        max_depth=5
    )
    mach_rf = machine(rf_model, X_train_clean, y_train_clean)
    _t_rf = @elapsed MLJ.fit!(mach_rf, verbosity=0)

    # 2. Linear Regression
    lr_model = MLJLinearModels.LinearRegressor()
    mach_lr = machine(lr_model, X_train_clean, y_train_clean)
    _t_lr = @elapsed MLJ.fit!(mach_lr, verbosity=0)

    # 3. XGBoost (threaded: nthread=8)
    X_matrix_clean = Matrix{Float64}(X_train_clean)
    y_vector_clean = Vector{Float64}(y_train_clean)
    
    dtrain = XGBoost.DMatrix(X_matrix_clean, label=y_vector_clean)
    _t_xgb = @elapsed mach_xgb = XGBoost.xgboost(dtrain,
        num_round=150,
        nthread=8,  # explicit: 8 threads
        max_depth=3,
        eta=0.01,
        max_delta_step=1,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1,
        subsample=0.7,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        verbosity=0,
    )

    pred_rf = nothing
    pred_lr = nothing
    pred_xgb = nothing
    if nrow(test_df) > 0
        pred_rf, pred_lr, pred_xgb = predict_with_machines(mach_rf, mach_lr, mach_xgb, test_df, feature_cols, X_train_clean)
    end

    raw_importances = MLJ.feature_importances(mach_rf)
    importance_dict = Dict{String, Float64}()
    for item in raw_importances
        importance_dict[String(item[1])] = Float64(item[2])
    end

    println("[BENCHMARK] RF=$(round(_t_rf; digits=3))s LR=$(round(_t_lr; digits=3))s XGB=$(round(_t_xgb; digits=3))s")
    return pred_rf, pred_lr, pred_xgb, importance_dict, mach_rf, mach_lr, mach_xgb
end

function predict_with_machines(mach_rf, mach_lr, mach_xgb, test_df::DataFrame, feature_cols::Vector{String}, X_train_clean::DataFrame)
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
    
    X_test_matrix = Matrix{Float64}(X_test_filled)
    dtest = XGBoost.DMatrix(X_test_matrix)
    pred_xgb = XGBoost.predict(mach_xgb, dtest)
    
    return pred_rf, pred_lr, pred_xgb
end

end