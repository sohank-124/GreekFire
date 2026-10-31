module Visualizer

using DataFrames, CSV, Statistics
using ..Backtest

export export_results

function extract_alphas(results)
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
        return nothing, nothing, nothing, nothing, nothing, nothing
    end

    dates = [r.date for r in valid_results]
    alphas_false = Float64[]
    alphas_true = Float64[]
    alphas_short = Float64[]
    alphas_chosen = Float64[]
    regimes = String[]

    for r in valid_results
        a_false = hasproperty(r, :actual_alpha_long) ? getproperty(r, :actual_alpha_long) : NaN
        a_true = hasproperty(r, :actual_alpha_high_pred) ? getproperty(r, :actual_alpha_high_pred) : NaN
        a_short = hasproperty(r, :actual_alpha_short) ? getproperty(r, :actual_alpha_short) : NaN
        a_chosen = hasproperty(r, :chosen_alpha) ? getproperty(r, :chosen_alpha) : NaN
        regime = hasproperty(r, :regime_signal) ? string(getproperty(r, :regime_signal)) : "none"

        push!(alphas_false, isfinite(a_false) ? a_false : 0.0)
        push!(alphas_true, isfinite(a_true) ? a_true : 0.0)
        push!(alphas_short, isfinite(a_short) ? a_short : 0.0)
        push!(alphas_chosen, isfinite(a_chosen) ? a_chosen : 0.0)
        push!(regimes, regime)
    end

    return dates, alphas_false, alphas_true, alphas_short, alphas_chosen, regimes
end

function export_results(results::Vector{Any}, feature_importances::Vector{Any}, performance;
                        output_dir=joinpath(pwd(), "results", "corrected"))

    alphas_data = extract_alphas(results)
    if alphas_data[1] === nothing
        @warn "No valid results to export"
        return
    end

    dates, alphas_false, alphas_true, alphas_short, alphas_chosen, regimes = alphas_data

    mkpath(output_dir)

    valid_results = [r for r in results if r !== nothing && !ismissing(r)]
    ec_false = vcat(1.0, cumprod(1.0 .+ alphas_false))
    ec_true  = vcat(1.0, cumprod(1.0 .+ alphas_true))
    ec_chosen = vcat(1.0, cumprod(1.0 .+ alphas_chosen))
    spy_cum = vcat(1.0, cumprod(1 .+ [r.spy_ret for r in valid_results]))

    n = length(ec_false)

    CSV.write(joinpath(output_dir, "equity_curve.csv"), DataFrame(
        step_index = 0:(n - 1),
        strategy = ec_chosen,
        spy_benchmark = spy_cum
    ))

    CSV.write(joinpath(output_dir, "regime_signal.csv"), DataFrame(
        step = 1:length(regimes),
        regime = regimes
    ))

    CSV.write(joinpath(output_dir, "monthly_alpha.csv"), DataFrame(
        month_index = 1:length(alphas_false),
        alpha_false = alphas_false,
        alpha_true = alphas_true,
        alpha_short = alphas_short,
        alpha_chosen = alphas_chosen
    ))

    function calc_drawdown(ec)
        peak = 1.0
        dd = Float64[]
        for val in ec
            if val > peak
                peak = val
            end
            push!(dd, (peak - val) / peak)
        end
        return dd
    end

    dd_false = calc_drawdown(ec_false)
    dd_true = calc_drawdown(ec_true)
    dd_chosen = calc_drawdown(ec_chosen)

    CSV.write(joinpath(output_dir, "drawdown.csv"), DataFrame(
        step_index = 0:(length(dd_false) - 1),
        dd_false = dd_false,
        dd_true = dd_true,
        dd_chosen = dd_chosen
    ))

    function rolling_winrate(alphas, window=50)
        wins = [a > 0 ? 1.0 : 0.0 for a in alphas]
        rates = Float64[]
        for i in 1:length(wins)
            w = wins[max(1, i - window + 1):i]
            push!(rates, mean(w))
        end
        return rates
    end

    wr_false = rolling_winrate(alphas_false)
    wr_true = rolling_winrate(alphas_true)
    wr_chosen = rolling_winrate(alphas_chosen)

    CSV.write(joinpath(output_dir, "rolling_winrate.csv"), DataFrame(
        step_index = 0:(length(wr_false) - 1),
        winrate_false = wr_false,
        winrate_true = wr_true,
        winrate_chosen = wr_chosen
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
        CSV.write(joinpath(output_dir, "feature_importance.csv"), DataFrame(rows))
    end

    # Observability: filter valid results once for the per-step exports below.
    valid_results = [r for r in results if r !== nothing && !ismissing(r)]

    # Export holdings per rebalance (drawdown investigation).
    holdings_rows = []
    for (i, r) in enumerate(valid_results)
        if r === nothing
            continue
        end
        # Long holdings
        n_long = length(r.long_indices_copy)
        for (rank, idx) in enumerate(r.long_indices_copy)
            row = r.period_scores_copy[idx, :]
            sector = get(Backtest.SECTOR_MAP, row.ticker, "Unknown")
            equal_weight = 1.0 / n_long
            inv_vol_w = hasproperty(r, :inv_vol_weights) && rank <= length(r.inv_vol_weights) ? r.inv_vol_weights[rank] : equal_weight
            contribution = inv_vol_w * row.avg_actual
            push!(holdings_rows, (
                step_index = i - 1,
                date = r.date,
                ticker = row.ticker,
                rank = rank,
                predicted_score = row.avg_pred,
                sector = sector,
                equal_weight = equal_weight,
                inv_vol_weight = inv_vol_w,
                weight = inv_vol_w,
                side = "long",
                actual_return = row.avg_actual,
                contribution = contribution
            ))
        end
        # Short holdings
        n_short = length(r.short_indices_copy)
        for (rank, idx) in enumerate(r.short_indices_copy)
            row = r.period_scores_copy[idx, :]
            sector = get(Backtest.SECTOR_MAP, row.ticker, "Unknown")
            short_equal_w = -1.0 / n_short
            short_inv_vol_w = short_equal_w
            contribution = short_inv_vol_w * row.avg_actual
            push!(holdings_rows, (
                step_index = i - 1,
                date = r.date,
                ticker = row.ticker,
                rank = rank,
                predicted_score = row.avg_pred,
                sector = sector,
                equal_weight = short_equal_w,
                inv_vol_weight = short_inv_vol_w,
                weight = short_inv_vol_w,
                side = "short",
                actual_return = row.avg_actual,
                contribution = contribution
            ))
        end
    end
    holdings_df = DataFrame(holdings_rows)
    CSV.write(joinpath(output_dir, "holdings.csv"), holdings_df)

    # Export weight diagnostics per rebalance
    weight_diag_rows = []
    for (i, r) in enumerate(valid_results)
        if r === nothing
            continue
        end
        n_long = length(r.long_indices_copy)
        inv_vols = hasproperty(r, :inv_vol_weights) ? r.inv_vol_weights : fill(1.0/n_long, n_long)
        hhi = sum(inv_vols .^ 2)
        push!(weight_diag_rows, (
            step_index = i - 1,
            date = r.date,
            n_stocks = n_long,
            max_weight = maximum(inv_vols),
            min_weight = minimum(inv_vols),
            hhi = hhi,
            equal_weight = 1.0 / n_long,
            concentration_ratio = maximum(inv_vols) / (1.0 / n_long)
        ))
    end
    weight_diag_df = DataFrame(weight_diag_rows)
    CSV.write(joinpath(output_dir, "weight_diagnostics.csv"), weight_diag_df)

    # Export model health metrics per step (drawdown investigation).
    health_rows = []
    for (i, r) in enumerate(valid_results)
        if r === nothing
            continue
        end
        push!(health_rows, (
            step_index = i - 1,
            date = r.date,
            ic = r.ic,
            rank_ic = r.rank_ic,
            portfolio_return = r.actual_alpha_long,
            spy_return = r.spy_ret,
            pred_spread = maximum(r.period_scores_copy.avg_pred) - minimum(r.period_scores_copy.avg_pred),
            decile_spread = r.decile_spread
        ))
    end
    health_df = DataFrame(health_rows)
    CSV.write(joinpath(output_dir, "model_health.csv"), health_df)

end

end
