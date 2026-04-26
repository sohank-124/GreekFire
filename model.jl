module Model

using MLJ, DecisionTree, MLJDecisionTreeInterface, MLJLinearModels, DataFrames, Statistics
import MLJ: predict

function train_and_predict(train_df::DataFrame, test_df::DataFrame, feature_cols::Vector{String})
    # Handle case with 0 columns
    if isempty(feature_cols) || nrow(train_df) == 0
        return nothing, nothing, nothing
    end

    # Train set cleaning - properly handle columns as symbols
    X_train = train_df[:, feature_cols]
    y_train = train_df.target
    
    # Check for NaNs properly - make sure X_train is Float64
    X_matrix = Matrix{Float64}(X_train)
    valid_train_mask = .!any(isnan, X_matrix, dims=2)[:] .& .!isnan.(y_train)
    X_train_clean = X_train[valid_train_mask, :]
    y_train_clean = y_train[valid_train_mask]
    
    if isempty(y_train_clean) || all(isnan, y_train_clean)
        return nothing, nothing, nothing
    end

    # 1. Random Forest Regressor
    n_subfeatures = max(3, floor(Int, sqrt(length(feature_cols))))
    rf_model = MLJDecisionTreeInterface.RandomForestRegressor(n_trees=100, n_subfeatures=n_subfeatures, max_depth=10)
    mach_rf = machine(rf_model, X_train_clean, y_train_clean)
    MLJ.fit!(mach_rf)
    
    # 2. Linear Regression
    lr_model = MLJLinearModels.LinearRegressor()
    mach_lr = machine(lr_model, X_train_clean, y_train_clean)
    MLJ.fit!(mach_lr)
    
    # Test set filling
    X_test = test_df[:, feature_cols]
    X_test_filled = copy(X_test)
    for col in feature_cols
        col_data = X_train_clean[:, col]
        valid_data = col_data[(!isnan).(col_data)]
        m = isempty(valid_data) ? 0.0 : mean(valid_data)
        
        # Get indices of NaN/issing values in this column
        nan_indices = findall(x -> ismissing(x) || isnan(x), X_test_filled[:, col])
        for idx in nan_indices
            X_test_filled[idx, col] = m
        end
    end
    
    pred_rf = predict(mach_rf, X_test_filled)
    pred_lr = predict(mach_lr, X_test_filled)
    
    # Feature Importance
    raw_importances = MLJ.feature_importances(mach_rf)
    importance_dict = Dict{String, Float64}()
    for item in raw_importances
        feat_name = String(item[1])
        imp_val = Float64(item[2])
        importance_dict[feat_name] = imp_val
    end
    
    return pred_rf, pred_lr, importance_dict
end

end
