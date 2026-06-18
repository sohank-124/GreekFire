module HMM

using Statistics, LinearAlgebra, Distributions

export train_hmm, predict_state, forward_algorithm

function logsumexp(x::Vector{Float64})
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end

function forward_algorithm(obs::Vector{Float64}, log_A::Matrix{Float64},
                           log_pi::Vector{Float64}, means::Vector{Float64},
                           stds::Vector{Float64})
    T = length(obs)
    K = 2
    alpha = Matrix{Float64}(undef, T, K)

    for k in 1:K
        alpha[1, k] = log_pi[k] + logpdf(Normal(means[k], stds[k]), obs[1])
    end

    for t in 2:T
        for k in 1:K
            log_probs = [alpha[t-1, j] + log_A[j, k] for j in 1:K]
            alpha[t, k] = logsumexp(log_probs) + logpdf(Normal(means[k], stds[k]), obs[t])
        end
    end

    log_likelihood = logsumexp(alpha[T, :])
    return alpha, log_likelihood
end

function backward_algorithm(obs::Vector{Float64}, log_A::Matrix{Float64},
                            means::Vector{Float64}, stds::Vector{Float64})
    T = length(obs)
    K = 2
    beta = Matrix{Float64}(undef, T, K)
    beta[T, :] .= 0.0

    for t in (T-1):-1:1
        for j in 1:K
            log_probs = [log_A[j, k] + logpdf(Normal(means[k], stds[k]), obs[t+1]) + beta[t+1, k] for k in 1:K]
            beta[t, j] = logsumexp(log_probs)
        end
    end

    return beta
end

function train_hmm(data::Vector{Float64}; n_states::Int=2, max_iter::Int=100,
                   tol::Float64=1e-6, seed::Int=42)
    T = length(data)
    K = n_states

    if T < 2
        return Dict(
            "A" => [0.9 0.1; 0.1 0.9],
            "pi" => [0.5, 0.5],
            "means" => [0.0, 1e-4],
            "stds" => [1e-4, 1e-4],
            "log_likelihood" => -Inf
        )
    end

    mid_idx = max(1, T ÷ 2)
    sorted_data = sort(data)
    low_half  = sorted_data[1:mid_idx]
    high_half = sorted_data[(mid_idx+1):end]

    means = [mean(low_half), isempty(high_half) ? mean(low_half) + 1e-4 : mean(high_half)]
    stds  = [max(std(low_half), 1e-6), max(isempty(high_half) ? std(low_half) : std(high_half), 1e-6)]

    if any(isnan, means) || any(isnan, stds) || any(x -> x <= 0, stds)
        fallback_mean = isfinite(mean(data)) ? mean(data) : 0.0
        return Dict(
            "A"              => [0.9 0.1; 0.1 0.9],
            "pi"             => [0.5, 0.5],
            "means"          => [fallback_mean, fallback_mean + 1e-4],
            "stds"           => [1e-4, 1e-4],
            "log_likelihood" => -Inf
        )
    end

    A  = [0.9 0.1; 0.1 0.9]
    pi = [0.5, 0.5]

    prev_log_likelihood = -Inf

    for iter in 1:max_iter
        log_A  = log.(A)
        log_pi = log.(pi)

        alpha, log_likelihood = forward_algorithm(data, log_A, log_pi, means, stds)
        beta = backward_algorithm(data, log_A, means, stds)

        if abs(log_likelihood - prev_log_likelihood) < tol
            prev_log_likelihood = log_likelihood
            break
        end
        prev_log_likelihood = log_likelihood

        gamma = Matrix{Float64}(undef, T, K)
        for t in 1:T
            denom = logsumexp([alpha[t, 1] + beta[t, 1], alpha[t, 2] + beta[t, 2]])
            for k in 1:K
                gamma[t, k] = exp(alpha[t, k] + beta[t, k] - denom)
            end
        end

        for k in 1:K
            gamma_sum = sum(gamma[:, k])
            means[k] = sum(gamma[:, k] .* data) / max(gamma_sum, 1e-10)
            variance  = sum(gamma[:, k] .* (data .- means[k]).^2) / max(gamma_sum, 1e-10)
            stds[k]   = sqrt(max(variance, 1e-8))
            if isnan(stds[k]) || stds[k] <= 0
                stds[k] = 1e-6
            end
            if isnan(means[k])
                means[k] = mean(data)
            end
        end

        for j in 1:K
            for k in 1:K
                numerator   = 0.0
                denominator = 0.0
                for t in 2:T
                    numerator   += exp(alpha[t-1, j] + log_A[j, k] +
                                       logpdf(Normal(means[k], stds[k]), data[t]) +
                                       beta[t, k] - log_likelihood)
                    denominator += exp(alpha[t-1, j] + beta[t-1, j] - log_likelihood)
                end
                A[j, k] = numerator / max(denominator, 1e-10)
            end
            row_sum = sum(A[j, :])
            if row_sum > 1e-10
                A[j, :] ./= row_sum
            else
                A[j, :] .= 1.0 / K
            end
        end

        pi = gamma[1, :] ./ max(sum(gamma[1, :]), 1e-10)
    end

    return Dict(
        "A"              => A,
        "pi"             => pi,
        "means"          => means,
        "stds"           => stds,
        "log_likelihood" => prev_log_likelihood
    )
end

function predict_state(model::Dict, obs::Vector{Float64})
    means  = model["means"]
    stds   = model["stds"]
    pi     = model["pi"]
    log_pi = log.(pi)

    T      = length(obs)
    states = Vector{Int}(undef, T)

    for t in 1:T
        log_probs  = [log_pi[k] + logpdf(Normal(means[k], stds[k]), obs[t]) for k in 1:2]
        states[t]   = argmax(log_probs)
    end

    return states
end

function predict_state_probs(model::Dict, obs::Vector{Float64};
                             init_probs::Union{Vector{Float64},Nothing}=nothing)
    means  = model["means"]
    stds   = model["stds"]
    A      = model["A"]
    pi     = model["pi"]
    log_A  = log.(A)

    T = length(obs)
    K = length(means)
    alpha = Matrix{Float64}(undef, T, K)

    log_init = if init_probs !== nothing
        log.(init_probs)
    else
        log.(pi)
    end

    for k in 1:K
        alpha[1, k] = log_init[k] + logpdf(Normal(means[k], stds[k]), obs[1])
    end

    for t in 2:T
        for k in 1:K
            log_probs = [alpha[t-1, j] + log_A[j, k] for j in 1:K]
            alpha[t, k] = logsumexp(log_probs) + logpdf(Normal(means[k], stds[k]), obs[t])
        end
    end

    probs = Matrix{Float64}(undef, T, K)
    for t in 1:T
        denom = logsumexp(alpha[t, :])
        for k in 1:K
            probs[t, k] = exp(alpha[t, k] - denom)
        end
    end

    return probs
end

end
