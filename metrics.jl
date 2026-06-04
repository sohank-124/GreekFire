module Metrics

using DataFrames, Statistics, Dates

function compute_performance(results; signal::Symbol = :chosen)
    alpha_field = if signal == :base
        :actual_alpha_false
    elseif signal == :fallback
        :actual_alpha_true
    elseif signal == :short
        :actual_alpha_short
    else
        :chosen_alpha
    end

    valid_results = filter(r -> r !== nothing && !ismissing(r), results)
    if isempty(valid_results)
        return nothing
    end

    alphas = Float64[]
    for r in valid_results
        if hasproperty(r, alpha_field)
            a = getproperty(r, alpha_field)
            if isfinite(a)
                push!(alphas, a)
            else
                push!(alphas, 0.0)
            end
        end
    end

    if isempty(alphas)
        return nothing
    end

    equity_curve = vcat([1.0], cumprod(1.0 .+ alphas))
    equity_curve = collect(Float64, equity_curve[isfinite.(equity_curve)])

    if isempty(equity_curve) || equity_curve[end] <= 0.0
        return nothing
    end

    steps_per_year = 252.0 / 5.0
    years = length(alphas) / steps_per_year

    total_return = equity_curve[end] - 1.0
    cagr = (total_return >= -1.0 && years > 0) ? (1.0 + total_return)^(1.0 / years) - 1.0 : 0.0

    rf_per_step = 0.05 / steps_per_year
    excess_over_rf = alphas .- rf_per_step
    sharpe = std(excess_over_rf) > 0 ? (mean(excess_over_rf) / std(excess_over_rf)) * sqrt(steps_per_year) : 0.0

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

function compute_all_signals(results)
    signals = Dict{Symbol, Any}()
    for sig in (:base, :fallback, :short, :chosen)
        signals[sig] = compute_performance(results; signal=sig)
    end
    return signals
end

end