module Metrics

using DataFrames, Statistics, Dates

function compute_performance(results)
    valid_results = filter(r -> r !== nothing && !ismissing(r) && !isnan(r.actual_alpha), results)
    if isempty(valid_results)
        return nothing
    end

    df = DataFrame([r for r in valid_results])
    alphas = collect(Float64, df.actual_alpha)
    alphas = alphas[isfinite.(alphas)]

    if isempty(alphas)
        return nothing
    end

    # FIX 4: Prepend 1.0 so equity path starts at $1
    equity_curve = vcat([1.0], cumprod(1.0 .+ alphas))
    # FIX 5: isfinite catches both NaN and Inf
    equity_curve = collect(Float64, equity_curve[isfinite.(equity_curve)])

    if isempty(equity_curve) || equity_curve[end] <= 0.0
        return nothing
    end

    steps_per_year = 252.0 / 5.0
    years = length(alphas) / steps_per_year

    # FIX 1: total_return from prepended-1.0 equity curve
    total_return = equity_curve[end] - 1.0
    cagr = (total_return >= -1.0 && years > 0) ? (1.0 + total_return)^(1.0 / years) - 1.0 : 0.0

    rf_per_step = 0.05 / steps_per_year
    excess_over_rf = alphas .- rf_per_step
    sharpe = std(excess_over_rf) > 0 ? (mean(excess_over_rf) / std(excess_over_rf)) * sqrt(steps_per_year) : 0.0

    # FIX 3: peak starts at 1.0 (the prepended starting capital)
    peak = 1.0
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

    win_rate = sum(alphas .> 0) / length(alphas)
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
