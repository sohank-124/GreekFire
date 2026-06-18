include("data_loader.jl")
include("features.jl")
include("model.jl")
include("backtest.jl")
include("metrics.jl")
include("visualizer.jl")
include("regime.jl")

using DataFrames, Dates, Statistics

DEBUG_MODE = false

function main()
    println("Running on $(Threads.nthreads()) threads")
    println("Starting GreekFire Walk-Forward Backtesting Engine with Regime Switching...")

    all_stock_data = DataLoader.load_all_data()
    # SPY benchmark is not in S&P 500 membership csv, fetch it separately
    spy_data = DataLoader.load_ticker_data("SPY")
    if spy_data === nothing
        error("Failed to load SPY benchmark data. Cannot proceed with backtest.")
    end

    if DEBUG_MODE
        println("=== DEBUG MODE ===")
        results, importances = Backtest.run_walk_forward_debug(all_stock_data, spy_data)
    else
        runtime = @elapsed results, importances, ic_history, rank_ic_history, decile_history =
            Backtest.run_walk_forward(all_stock_data, spy_data)

        # HMM regime switching DISABLED — was destroying alpha (bear 40% of bull market)
        # All steps use long-only alpha. Revisit as separate research problem.
        for i in 1:length(results)
            r = results[i]
            if r !== nothing
                results[i] = merge(r, (chosen_alpha = r.actual_alpha_long, regime_signal = :bull))
            end
        end

        valid_imps = []
        for i in 1:length(importances)
            try
                imp = importances[i]
                if isa(imp, Dict)
                    push!(valid_imps, imp)
                end
            catch
                continue
            end
        end
        if !isempty(valid_imps)
            first_imp = valid_imps[1]
            sorted = sort(collect(first_imp), by=x->x[2], rev=true)
            println("\n--- Top 3 Feature Importances ---")
            for i in 1:min(3, length(sorted))
                println("$(i). $(sorted[i][1]): $(round(sorted[i][2], digits=4))")
            end
            println("-------------------------------\n")
        end
    end

    valid_r = [r for r in results if r !== nothing]
    println("Number of valid result steps: ", length(valid_r))
    if !isempty(valid_r)
        println("Date range: ", valid_r[1].date, " to ", valid_r[end].date)
    end

    all_perf = Metrics.compute_all_signals(results)

    println("\n--- Strategy Performance ---")
    for s in (:base, :linear, :chosen)
        p = all_perf[s]
        label = s == :base ? "base (RF+XGB long)" : s == :linear ? "linear (equal-weight)" : "chosen (long-only)"
        if p !== nothing
            println("$(label):")
            println("  CAGR: $(round(p.cagr * 100, digits=2))%")
            println("  Sharpe: $(round(p.sharpe, digits=2))")
            println("  Max DD: $(round(p.max_dd * 100, digits=2))%")
            println("  Win Rate: $(round(p.win_rate * 100, digits=2))%")
            println("  Avg Excess: $(round(p.avg_excess * 100, digits=2))%")
        else
            println("$(label): No valid results")
        end
    end
    println("-------------------------------------\n")

    perf = all_perf[:base]

    if perf === nothing
        println("No valid results found. Backtest failed.")
        return
    end

    # Cross-sectional diagnostics
    valid_ic = [v for v in ic_history if isfinite(v)]
    valid_rank_ic = [v for v in rank_ic_history if isfinite(v)]
    valid_deciles = [d for d in decile_history if all(isfinite, d)]
    top_decile = [d[1] for d in valid_deciles]
    bot_decile = [d[10] for d in valid_deciles]

    println("\n--- Cross-Sectional Diagnostics ---")
    if !isempty(valid_ic)
        println("Mean IC:       $(round(mean(valid_ic), digits=4))")
        println("IC Std:        $(round(std(valid_ic), digits=4))")
        println("IC t-stat:     $(round(mean(valid_ic) / std(valid_ic) * sqrt(length(valid_ic)), digits=2))")
    end
    if !isempty(valid_rank_ic)
        println("Mean Rank IC:  $(round(mean(valid_rank_ic), digits=4))")
    end
    if !isempty(top_decile)
        println("Top Decile Avg:   $(round(mean(top_decile) * 100, digits=2))%")
        println("Bottom Decile Avg: $(round(mean(bot_decile) * 100, digits=2))%")
        println("Decile Spread:    $(round((mean(top_decile) - mean(bot_decile)) * 100, digits=2))%")
    end
    println("----------------------------------\n")

    linear_perf = all_perf[:linear]

    Visualizer.export_results(results, importances, perf)
    println("Results exported to /octave directory.")

    try
        log_path = joinpath(pwd(), "EXPERIMENT_LOG.md")

        run_num = 1
        if isfile(log_path)
            run_num = sum(occursin("## Run", line) for line in readlines(log_path)) + 1
        end

        num_tickers = length(DataLoader.get_all_sp500_tickers())

        valid_imps = []
        for i in 1:length(importances)
            try
                imp = importances[i]
                if isa(imp, Dict) push!(valid_imps, imp) end
            catch
                continue
            end
        end

        top_features_str = "N/A"
        if !isempty(valid_imps)
            first_imp = valid_imps[1]
            sorted = sort(collect(first_imp), by=x->x[2], rev=true)
            top_features_str = join([
                "$(sorted[i][1]) ($(round(sorted[i][2], digits=4)))"
                for i in 1:min(3, length(sorted))
            ], ", ")
        end

        n_sub = "N/A"
        if !isempty(valid_imps)
            n_feat = length(collect(keys(valid_imps[1])))
            n_sub = max(3, floor(Int, sqrt(n_feat)))
        end

        ic_str = !isempty(valid_ic) ? "$(round(mean(valid_ic), digits=4))" : "N/A"
        rank_ic_str = !isempty(valid_rank_ic) ? "$(round(mean(valid_rank_ic), digits=4))" : "N/A"
        spread_str = (!isempty(top_decile) && !isempty(bot_decile)) ? "$(round((mean(top_decile) - mean(bot_decile)) * 100, digits=2))%" : "N/A"
        linear_cagr = (linear_perf !== nothing) ? "$(round(linear_perf.cagr * 100, digits=2))%" : "N/A"

        run_date_str = Dates.format(today(), "yyyy-mm-dd")

        # Format runtime as "Xm Ys" or "Xs" (N/A in DEBUG_MODE)
        runtime_str = @isdefined(runtime) ?
            (let m = floor(Int, runtime / 60), s = round(Int, runtime - m * 60)
                m > 0 ? "$(m)m $(s)s" : "$(s)s"
            end) : "N/A"

        log_entry =
            "\n## Run $run_num — $run_date_str\n" *
            "- Date range: $(DataLoader.START_DATE) to present\n" *
            "- Stock universe: $num_tickers tickers (point-in-time S&P 500)\n" *
            "- Model: Ensemble (RF + XGBoost, 50/50 weighted average)\n" *
            "- Training Window: $(Backtest.TRAIN_WINDOW_DAYS) days rolling\n" *
            "- Test Step: $(Backtest.STEP_DAYS) days\n" *
            "- Top N Stocks: $(Backtest.TOP_N_STOCKS)\n" *
            "- Short N Stocks: $(Backtest.BOTTOM_N_STOCKS)\n" *
            "- Stop Loss: $(Backtest.STOP_LOSS_THRESHOLD * 100)%\n" *
            "- Cross-sectional percentile rank + clamp [0.02, 0.98]\n" *
            "- Linear baseline: equal-weight factor mean\n" *
            "- HMM: disabled (was destroying alpha)\n" *
            "- RF n_subfeatures: $n_sub\n" *
            "- CAGR: $(round(perf.cagr * 100, digits=2))%\n" *
            "- Sharpe Ratio: $(round(perf.sharpe, digits=2))\n" *
            "- Max Drawdown: $(round(perf.max_dd * 100, digits=2))%\n" *
            "- Win Rate: $(round(perf.win_rate * 100, digits=2))%\n" *
            "- Avg Excess Return: $(round(perf.avg_excess * 100, digits=2))%\n" *
            "- Mean IC: $ic_str\n" *
            "- Mean Rank IC: $rank_ic_str\n" *
            "- Decile Spread: $spread_str\n" *
            "- Linear Baseline CAGR: $linear_cagr\n" *
            "- Top 3 features: $top_features_str\n" *
            "- Runtime: $runtime_str\n" *
            "- Notes:\n"

        open(log_path, "a") do io
            write(io, log_entry)
        end
        println("Experiment log updated.")
    catch e
        @warn "Failed to update experiment log: $e"
    end
end

if isempty(ARGS)
    main()
end