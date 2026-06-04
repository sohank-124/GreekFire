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
        results, importances = Backtest.run_walk_forward(all_stock_data, spy_data)

        # Regime is now embedded in backtest: either short portfolio (when possible) or hold SPY
        # HMM regime switching removed per user request

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
    for s in (:short, :chosen)
        p = all_perf[s]
        label = s == :short ? "shorting only (always short)" : "chosen (short or hold SPY)"
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

    perf = Metrics.compute_performance(results; signal=:chosen)

    if perf === nothing
        println("No valid results found. Backtest failed.")
        return
    end

    println("\n--- Performance Metrics ---")
    println("CAGR: ", round(perf.cagr * 100, digits=2), "%")
    println("Sharpe Ratio: ", round(perf.sharpe, digits=2))
    println("Max Drawdown: ", round(perf.max_dd * 100, digits=2), "%")
    println("Win Rate: ", round(perf.win_rate * 100, digits=2), "%")
    println("Avg Excess Return: ", round(perf.avg_excess * 100, digits=2), "%")
    println("--------------------------\n")

    try
        regime_counts = Dict()
        for r in results
            if r !== nothing && hasproperty(r, :regime_signal)
                sig = r.regime_signal
                regime_counts[sig] = get(regime_counts, sig, 0) + 1
            end
        end
        println("\n--- Regime Signal Distribution ---")
        for (sig, count) in regime_counts
            println("  $(sig): $count steps")
        end
        println("--------------------------------\n")
    catch e
        @warn "Failed to print regime breakdown: $e"
    end

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

        log_entry = "\n## Run $run_num — $(Dates.format(today(), "yyyy-mm-dd"))\n" *
                    "- Date range: $(DataLoader.START_DATE) to present\n" *
                    "- Stock universe: $num_tickers tickers (point-in-time S&P 500)\n" *
                    "- Model: Ensemble (RF + XGBoost, 50/50 weighted average)\n" *
                    "- Training Window: $(Backtest.TRAIN_WINDOW_DAYS) days rolling\n" *
                    "- Test Step: $(Backtest.STEP_DAYS) days\n" *
                    "- Top N Stocks: $(Backtest.TOP_N_STOCKS)\n" *
                    "- Short N Stocks: $(Backtest.BOTTOM_N_STOCKS)\n" *
                    "- Stop Loss: $(Backtest.STOP_LOSS_THRESHOLD * 100)%\n" *
                    "- RF n_subfeatures: $n_sub\n" *
                    "- CAGR: $(round(perf.cagr * 100, digits=2))%\n" *
                    "- Sharpe Ratio: $(round(perf.sharpe, digits=2))\n" *
                    "- Max Drawdown: $(round(perf.max_dd * 100, digits=2))%\n" *
                    "- Win Rate: $(round(perf.win_rate * 100, digits=2))%\n" *
                    "- Avg Excess Return: $(round(perf.avg_excess * 100, digits=2))%\n" *
                    "- Top 3 features: $top_features_str\n" *
                    "- Notes: \n"

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