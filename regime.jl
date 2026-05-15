module Regime

using DataFrames, Dates, Statistics

include("hmm.jl")
using .HMM

const TRAIN_WINDOW = 252
const MIN_STEPS_FOR_HMM = 20
const BORROW_COST = 0.001
# Long/short allocation in bear regime
# 1.5 = 150% long + 150% short (aggressive, net neutral) changed to .8 less aggressive
const BEAR_LONG_WEIGHT = 0.5
const BEAR_SHORT_WEIGHT = .8

function train_hmm_on_history(alphas::Vector{Float64}, current_idx::Int)
    start_idx = max(1, current_idx - TRAIN_WINDOW)
    hist_alphas = alphas[start_idx:(current_idx - 1)]
    hist_alphas = hist_alphas[isfinite.(hist_alphas)]

    if length(hist_alphas) < MIN_STEPS_FOR_HMM
        return nothing
    end

    model = HMM.train_hmm(hist_alphas)

    means = model["means"]
    if means[1] > means[2]
        model["state_labels"] = Dict(1 => :bull, 2 => :bear)
    else
        model["state_labels"] = Dict(1 => :bear, 2 => :bull)
    end

    return model
end

function predict_current_regime(model, alphas::Vector{Float64}, current_idx::Int)
    if model === nothing
        return :bull
    end

    if current_idx <= 1
        return :bull
    end

    last_known_alpha = alphas[current_idx - 1]

    if !isfinite(last_known_alpha)
        return :bull
    end

    states = HMM.predict_state(model, [last_known_alpha])
    state_num = states[1]

    state_labels = model["state_labels"]
    if haskey(state_labels, state_num)
        return state_labels[state_num]
    else
        return :bull
    end
end

function apply_regime_switching(results::Vector{Any}, spy_data::DataFrame)
    n = length(results)
    if n == 0
        return results
    end

    spy_returns = Float64[]
    for r in results
        if r !== nothing && hasproperty(r, :date)
            date = r.date
            idx = searchsortedlast(spy_data.date, date)
            if idx > 1 && idx <= nrow(spy_data)
                ret = (spy_data.close[idx] / spy_data.close[idx-1]) - 1.0
                push!(spy_returns, ret)
            else
                push!(spy_returns, NaN)
            end
        else
            push!(spy_returns, NaN)
        end
    end

    alphas_base = Float64[]
    alphas_true = Float64[]

    for r in results
        if r !== nothing && hasproperty(r, :actual_alpha_false)
            push!(alphas_base, r.actual_alpha_false)
            push!(alphas_true, r.actual_alpha_true)
        else
            push!(alphas_base, NaN)
            push!(alphas_true, NaN)
        end
    end

    new_results = Any[]

    for i in 1:n
        r = results[i]

        if r === nothing
            push!(new_results, nothing)
            continue
        end

        model = train_hmm_on_history(alphas_base, i)
        regime = predict_current_regime(model, alphas_base, i)

        base_alpha  = hasproperty(r, :actual_alpha_false) ? r.actual_alpha_false : 0.0
        short_alpha = hasproperty(r, :actual_alpha_short) ? r.actual_alpha_short : 0.0

        chosen_alpha = 0.0

        if regime == :bull
            # Bull: full long position using base strategy
            chosen_alpha = isfinite(base_alpha) ? base_alpha : 0.0

        elseif regime == :bear
            # Bear: aggressive long-short
            # BEAR_LONG_WEIGHT of base strategy (reduced long)
            # BEAR_SHORT_WEIGHT of short portfolio (aggressive short)
            long_component  = isfinite(base_alpha) ? BEAR_LONG_WEIGHT * base_alpha : 0.0
            short_component = isfinite(short_alpha) ? BEAR_SHORT_WEIGHT * short_alpha : 0.0
            chosen_alpha = long_component + short_component
        end

        if !isfinite(chosen_alpha)
            chosen_alpha = 0.0
        end

        new_r = merge(
            r,
            (
                chosen_alpha  = chosen_alpha,
                regime_signal = regime,
            )
        )

        push!(new_results, new_r)
    end

    return new_results
end

end