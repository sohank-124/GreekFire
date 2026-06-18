module Regime

using DataFrames, Dates, Statistics, CSV

include("hmm.jl")
using .HMM

const TRAIN_WINDOW = 126
const MIN_STEPS_FOR_HMM = 20
const BORROW_COST = 0.001
const BEAR_LONG_WEIGHT = 0.5
const BEAR_SHORT_WEIGHT = 0.8

const PROB_EMA_ALPHA = 0.15
const OBS_WINDOW = 20
const VOL_LOOKBACK = 20
const VOL_BASELINE = 63
const RETRAIN_INTERVAL = 12

const BEAR_ENTER = 0.60
const BEAR_EXIT = 0.40
const BULL_ENTER = 0.60
const CASH_ENTER = 0.45

function compute_vol_spike(spy_data::DataFrame)
    n = nrow(spy_data)
    closes = Float64.(spy_data.close)

    daily_rets = fill(NaN, n)
    for i in 2:n
        if isfinite(closes[i]) && isfinite(closes[i-1]) && closes[i-1] > 0
            daily_rets[i] = closes[i] / closes[i-1] - 1.0
        end
    end

    vol_20 = fill(NaN, n)
    min_valid = VOL_LOOKBACK ÷ 2
    for i in (VOL_LOOKBACK + 1):n
        window = daily_rets[(i - VOL_LOOKBACK + 1):i]
        valid = window[isfinite.(window)]
        if length(valid) >= min_valid
            vol_20[i] = std(valid) * sqrt(252)
        end
    end

    sma_63 = fill(NaN, n)
    for i in (VOL_BASELINE + 1):n
        window = vol_20[(i - VOL_BASELINE + 1):i]
        valid = window[isfinite.(window)]
        if length(valid) >= 10
            sma_63[i] = mean(valid)
        end
    end

    spike = fill(NaN, n)
    for i in 1:n
        if isfinite(vol_20[i]) && isfinite(sma_63[i]) && sma_63[i] > 0
            spike[i] = vol_20[i] / sma_63[i]
        end
    end

    return spike, vol_20
end

function train_hmm_on_history(features::Vector{Float64}, current_idx::Int)
    start_idx = max(1, current_idx - TRAIN_WINDOW)
    hist_features = features[start_idx:(current_idx - 1)]
    hist_features = hist_features[isfinite.(hist_features)]

    if length(hist_features) < MIN_STEPS_FOR_HMM
        return nothing
    end

    model = HMM.train_hmm(hist_features)

    means = model["means"]
    if means[1] > means[2]
        model["state_labels"] = Dict(1 => :bear, 2 => :bull)
    else
        model["state_labels"] = Dict(1 => :bull, 2 => :bear)
    end

    return model
end

function predict_regime_with_probs(model, features::Vector{Float64}, current_idx::Int;
                                   prev_posterior::Union{Vector{Float64},Nothing}=nothing)
    if model === nothing
        return :bull, Dict(:bull => 0.5, :bear => 0.5), [0.5, 0.5]
    end

    if current_idx <= 1
        return :bull, Dict(:bull => 0.5, :bear => 0.5), [0.5, 0.5]
    end

    lookback = min(current_idx - 1, OBS_WINDOW)
    start_idx = current_idx - lookback
    obs_window = features[start_idx:(current_idx - 1)]

    valid_count = sum(isfinite.(obs_window))
    if valid_count < div(lookback, 2)
        return :bull, Dict(:bull => 0.5, :bear => 0.5), prev_posterior !== nothing ? prev_posterior : [0.5, 0.5]
    end

    for j in 1:length(obs_window)
        if !isfinite(obs_window[j])
            obs_window[j] = 0.0
        end
    end

    probs_matrix = HMM.predict_state_probs(model, obs_window; init_probs=prev_posterior)
    bull_prob = probs_matrix[end, 1]
    bear_prob = probs_matrix[end, 2]

    state_labels = model["state_labels"]
    if haskey(state_labels, 1) && state_labels[1] == :bear
        bull_prob, bear_prob = bear_prob, bull_prob
    end

    bull_prob = clamp(bull_prob, 0.0, 1.0)
    bear_prob = clamp(bear_prob, 0.0, 1.0)

    raw_posterior = [bull_prob, bear_prob]

    prob_dict = Dict(:bull => bull_prob, :bear => bear_prob)

    if bull_prob >= bear_prob
        return :bull, prob_dict, raw_posterior
    else
        return :bear, prob_dict, raw_posterior
    end
end

function apply_regime_switching(results::Vector{Any}, spy_data::DataFrame)
    n = length(results)
    if n == 0
        return results
    end

    spy_dates = Date[]
    for r in results
        if r !== nothing && hasproperty(r, :date)
            push!(spy_dates, r.date)
        else
            push!(spy_dates, Date(0))
        end
    end

    spike, vol_20 = compute_vol_spike(spy_data)

    spike_by_step = Float64[]
    for i in 1:n
        date = spy_dates[i]
        if date == Date(0)
            push!(spike_by_step, NaN)
            continue
        end
        idx = searchsortedlast(spy_data.date, date)
        if idx >= 1 && idx <= nrow(spy_data) && isfinite(spike[idx])
            push!(spike_by_step, spike[idx])
        elseif idx > nrow(spy_data) && isfinite(spike[end])
            push!(spike_by_step, spike[end])
        else
            push!(spike_by_step, NaN)
        end
    end

    prev_regime = :cash
    prev_posterior = nothing
    smoothed_bull = 0.5
    smoothed_bear = 0.5
    last_trained_model = nothing
    last_trained_idx = 0

    diag_dates = Date[]
    diag_p_bull = Float64[]
    diag_p_bear = Float64[]
    diag_smoothed_bull = Float64[]
    diag_smoothed_bear = Float64[]
    diag_regime = Symbol[]
    diag_spike = Float64[]
    diag_raw_regime = Symbol[]

    new_results = Any[]

    for i in 1:n
        r = results[i]

        if r === nothing
            push!(new_results, nothing)
            continue
        end

        if i == 1 || last_trained_model === nothing || (i - last_trained_idx) >= RETRAIN_INTERVAL
            model = train_hmm_on_history(spike_by_step, i)
            if model !== nothing
                last_trained_model = model
                last_trained_idx = i
            end
        end
        model = last_trained_model

        raw_regime, probs, raw_posterior = predict_regime_with_probs(
            model, spike_by_step, i; prev_posterior=prev_posterior
        )

        bull_prob = probs[:bull]
        bear_prob = probs[:bear]

        smoothed_bull = PROB_EMA_ALPHA * bull_prob + (1.0 - PROB_EMA_ALPHA) * smoothed_bull
        smoothed_bear = PROB_EMA_ALPHA * bear_prob + (1.0 - PROB_EMA_ALPHA) * smoothed_bear
        total = smoothed_bull + smoothed_bear
        if total > 0
            smoothed_bull /= total
            smoothed_bear /= total
        end

        prev_posterior = raw_posterior

        if prev_regime == :cash
            if smoothed_bull >= BULL_ENTER
                prev_regime = :bull
            elseif smoothed_bear >= BEAR_ENTER
                prev_regime = :bear
            end
        elseif prev_regime == :bull
            if smoothed_bear >= BEAR_ENTER
                prev_regime = :bear
            elseif max(smoothed_bull, smoothed_bear) < CASH_ENTER
                prev_regime = :cash
            end
        elseif prev_regime == :bear
            if smoothed_bull >= BULL_ENTER
                prev_regime = :bull
            elseif smoothed_bear < BEAR_EXIT
                prev_regime = :cash
            end
        end

        regime = prev_regime

        base_alpha  = hasproperty(r, :actual_alpha_long) ? r.actual_alpha_long : 0.0
        short_alpha = hasproperty(r, :actual_alpha_short) ? r.actual_alpha_short : 0.0

        chosen_alpha = 0.0

        if regime == :bull
            chosen_alpha = isfinite(base_alpha) ? base_alpha : 0.0
        elseif regime == :bear
            long_component  = isfinite(base_alpha)  ? BEAR_LONG_WEIGHT * base_alpha  : 0.0
            short_component = isfinite(short_alpha) ? BEAR_SHORT_WEIGHT * short_alpha : 0.0
            chosen_alpha = long_component + short_component
        elseif regime == :cash
            chosen_alpha = 0.0
        end

        if !isfinite(chosen_alpha)
            chosen_alpha = 0.0
        end

        new_r = merge(
            r,
            (
                chosen_alpha   = chosen_alpha,
                regime_signal  = regime,
            )
        )

        push!(new_results, new_r)

        push!(diag_dates, r.date)
        push!(diag_p_bull, bull_prob)
        push!(diag_p_bear, bear_prob)
        push!(diag_smoothed_bull, smoothed_bull)
        push!(diag_smoothed_bear, smoothed_bear)
        push!(diag_regime, regime)
        push!(diag_spike, i <= length(spike_by_step) && isfinite(spike_by_step[i]) ? spike_by_step[i] : NaN)
        push!(diag_raw_regime, raw_regime)
    end

    try
        diag_df = DataFrame(
            date = diag_dates,
            p_bull = diag_p_bull,
            p_bear = diag_p_bear,
            smoothed_bull = diag_smoothed_bull,
            smoothed_bear = diag_smoothed_bear,
            regime = string.(diag_regime),
            raw_regime = string.(diag_raw_regime),
            vol_spike = diag_spike
        )
        CSV.write(joinpath("octave", "regime_diagnostics.csv"), diag_df)
        @info "Regime diagnostics written to octave/regime_diagnostics.csv"
    catch e
        @warn "Failed to write regime diagnostics: $e"
    end

    return new_results
end

end
