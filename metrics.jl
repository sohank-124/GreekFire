module Metrics

using DataFrames, Statistics, Dates

function compute_performance(results)
    # results is a Vector of named tuples or nothing
    # Filter out nothing results AND NaN values (BUG 1 fix)
    valid_results = filter(r -> r !== nothing && !ismissing(r) && !isnan(r.actual_alpha), results)
    if isempty(valid_results)
        return nothing
    end

    # Convert to DataFrame for easier manipulation
    df = DataFrame([r for r in valid_results])
    
    # Calculate cumulative returns
    # Actual alpha is the excess return relative to SPY
    alphas = df.actual_alpha
    
    # Filter out any remaining NaN in alphas array
    alphas = alphas[.!isnan.(alphas)]
    
    if isempty(alphas)
        return nothing
    end
    
    # Cumulative Strategy Return
    # Use cumulative sum for excess returns (more stable when alphas can be negative)
    cum_excess = cumsum(alphas)
    
    # Annualized Return (CAGR)
    years = length(valid_results) / 12.0
    
    # Safe total return computation (handle edge cases)
    valid_alphas = alphas[alphas .> -1.0]  # Filter out returns < -100%
    if isempty(valid_alphas)
        total_return = 0.0
    else
        # Use arithmetic sum for stability
        total_return = sum(valid_alphas)
    end
    
    # Avoid negative base for CAGR
    cagr = total_return >= -1.0 ? (1.0 + total_return)^(1.0 / years) - 1.0 : 0.0
    
    # Sharpe Ratio - handle edge cases
    rf_monthly = 0.05 / 12.0
    excess_returns = alphas .- rf_monthly
    std_excess = std(excess_returns)
    sharpe = std_excess > 0 ? (mean(excess_returns) / std_excess) * sqrt(12) : 0.0
    
    # Max Drawdown
    equity_curve = cumprod(1.0 .+ alphas)
    equity_curve = equity_curve[.!isnan.(equity_curve)]  # Remove NaNs
    
    peak = equity_curve[1]
    max_dd = 0.0
    for val in equity_curve
        if isfinite(val) && val > 0
            if val > peak
                peak = val
            end
            dd = (peak - val) / peak
            if isfinite(dd) && dd > max_dd
                max_dd = dd
            end
        end
    end
    
    # Win Rate (this is correctly comparing actual_alpha > 0, which IS the excess return)
    win_rate = sum(alphas .> 0) / length(alphas)
    
    # Avg Excess Return
    avg_excess = mean(alphas)
    
    return (
        cagr = cagr,
        sharpe = sharpe,
        max_dd = max_dd,
        win_rate = win_rate,
        avg_excess = avg_excess,
        equity_curve = equity_curve
    )
end

end
