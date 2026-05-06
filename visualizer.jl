module Visualizer

using DataFrames, CSV, Statistics

function export_results(results::Vector{Any}, feature_importances::Vector{Any}, performance)
    
    valid_results = []
    for i in 1:length(results)
        try
            r = results[i]
            if r !== nothing && !ismissing(r)
                push!(valid_results, r)
            end
        catch
            continue
        end
    end

    if isempty(valid_results)
        @warn "No valid results to export"
        return
    end

    mkpath("octave")

    alphas_raw = [r.actual_alpha for r in valid_results]
    nan_mask = .!isnan.(alphas_raw)
    alphas = alphas_raw[nan_mask]
    tickers = [join(valid_results[i].tickers, ", ") for i in findall(nan_mask)]
    preds = [valid_results[i].predicted_alpha for i in findall(nan_mask)]

    spy_step_rate = (1.07)^(5.0/252.0) - 1.0
    strategy_cum = vcat([1.0], cumprod(1.0 .+ alphas))
    spy_cum = vcat([1.0], cumprod(fill(1.0 + spy_step_rate, length(alphas))))

    n_points = length(strategy_cum)

    CSV.write(joinpath("octave", "equity_curve.csv"), DataFrame(
        step_index = 0:(n_points - 1),
        strategy_cumulative_return = strategy_cum,
        spy_cumulative_return = spy_cum
    ))

    CSV.write(joinpath("octave", "monthly_alpha.csv"), DataFrame(
        month_index = 1:length(valid_results),
        ticker = tickers,
        predicted_alpha = preds,
        actual_alpha = alphas
    ))

    equity = strategy_cum
    drawdowns = Float64[]
    peak = 1.0
    for val in equity
        if val > peak peak = val end
        push!(drawdowns, (peak - val) / peak)
    end
    CSV.write(joinpath("octave", "drawdown.csv"), DataFrame(
        step_index = 0:(length(drawdowns) - 1),
        strategy_drawdown = drawdowns
    ))

    wins = [a > 0 ? 1.0 : 0.0 for a in alphas]
    winrates = Float64[]
    for i in 1:length(wins)
        window = wins[max(1, i-11):i]
        push!(winrates, mean(window))
    end
    CSV.write(joinpath("octave", "rolling_winrate.csv"), DataFrame(
        step_index = 0:(length(winrates) - 1),
        rolling_12month_winrate = winrates
    ))

    valid_importances = []
    for i in 1:length(feature_importances)
        try
            imp = feature_importances[i]
            if isa(imp, Dict)
                push!(valid_importances, imp)
            end
        catch
            continue
        end
    end

    if !isempty(valid_importances)
        feat_names = collect(keys(valid_importances[1]))
        rows = []
        for (i, imp) in enumerate(valid_importances)
            row = Dict{String,Any}("step_index" => i)
            for name in feat_names
                row[name] = get(imp, name, 0.0)
            end
            push!(rows, row)
        end
        CSV.write(joinpath("octave", "feature_importance.csv"), DataFrame(rows))
    end

end

end