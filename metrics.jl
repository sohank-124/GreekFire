module Metrics
using Statistics, Dates

"""Metrics for a complete sequence of non-overlapping simple period returns.
Default annualization is 252/5 and risk-free rate is zero. Missing/invalid
returns are errors; zero volatility makes Sharpe undefined (NaN).
"""
function return_metrics(returns; periods_per_year=252/5, risk_free_rate=0.0,
                        benchmark=nothing)
    r = Float64.(returns)
    isempty(r) && throw(ArgumentError("No returns"))
    all(isfinite, r) && all(r .>= -1) || throw(ArgumentError("Invalid simple returns"))
    isfinite(periods_per_year) && periods_per_year > 0 || throw(ArgumentError("Invalid frequency"))
    isfinite(risk_free_rate) && risk_free_rate > -1 || throw(ArgumentError("Invalid risk-free rate"))
    equity = vcat(1.0, cumprod(1 .+ r))
    all(isfinite, equity) || throw(ArgumentError("Equity overflow"))
    peaks = accumulate(max, equity)
    rf = (1 + risk_free_rate)^(1/periods_per_year) - 1
    excess = r .- rf
    sigma = length(r) > 1 ? std(excess) : NaN
    avg_excess = NaN
    relative_return = NaN
    if benchmark !== nothing
        b = Float64.(benchmark)
        length(b) == length(r) || throw(ArgumentError("Unaligned benchmark"))
        all(isfinite, b) && all(b .>= -1) || throw(ArgumentError("Invalid benchmark"))
        avg_excess = mean(r .- b)
        benchmark_equity = prod(1 .+ b)
        relative_return = benchmark_equity > 0 ? equity[end]/benchmark_equity - 1 : NaN
    end
    return (cagr=equity[end]^(periods_per_year/length(r))-1,
            sharpe=isfinite(sigma) && sigma > 0 ? mean(excess)/sigma*sqrt(periods_per_year) : NaN,
            max_dd=maximum(1 .- equity ./ peaks), win_rate=mean(r .> 0),
            mean_return=mean(r), avg_excess=avg_excess,
            benchmark_relative_return=relative_return, equity_curve=equity)
end

function compute_performance(results; signal::Symbol=:chosen, periods_per_year=252/5,
                             risk_free_rate=0.0)
    isempty(results) && return nothing
    any(r -> r === nothing || ismissing(r), results) && error("Incomplete backtest; refusing selective-period metrics")
    field = get(Dict(:base=>:net_alpha_long, :gross=>:actual_alpha_long,
        :fallback=>:actual_alpha_high_pred, :short=>:actual_alpha_short,
        :linear=>:net_alpha_linear, :linear_gross=>:actual_equal_weight_long), signal, :chosen_alpha)
    all(r -> hasproperty(r, field), results) || error("Missing return field $field")
    benchmark = all(r -> hasproperty(r, :spy_ret) && isfinite(r.spy_ret), results) ?
        [r.spy_ret for r in results] : nothing
    return return_metrics([getproperty(r, field) for r in results];
        periods_per_year, risk_free_rate, benchmark)
end

compute_all_signals(results) = Dict(sig=>compute_performance(results; signal=sig)
    for sig in (:base, :gross, :fallback, :short, :linear, :linear_gross, :chosen))
end
