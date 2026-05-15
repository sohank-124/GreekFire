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

    # 1. Load Data
    all_stock_data = DataLoader.load_all_data()
    spy_data = all_stock_data["SPY"]

    if DEBUG_MODE
        println("=== DEBUG MODE: Running first 5 steps sequentially ===")
        results, importances = Backtest.run_walk_forward_debug(all_stock_data, spy_data)
    else
        # 2. Run Backtest
        results, importances = Backtest.run_walk_forward(all_stock_data, spy_data)

        # 3. Apply HMM regime switching with protective puts
        println("Applying HMM regime detector...")
        results = Regime.apply_regime_switching(results, spy_data)

        # Print top 3 feature importances
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

    # 4. Compute Metrics
    valid_r = [r for r in results if r !== nothing]
    println("Number of valid result steps: ", length(valid_r))
    if !isempty(valid_r)
        println("Date range: ", valid_r[1].date, " to ", valid_r[end].date)
    end

    # Compute and compare all three signals
    all_perf = Metrics.compute_all_signals(results)

    println("\n--- Regime-Switching Performance ---")
    for s in (:base, :fallback, :chosen)
        p = all_perf[s]
        label = s == :base ? "rev=false (base) only" :
                s == :fallback ? "rev=true (fallback) only" : "regime-switched"
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

    # Legacy single-metric for compatibility
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

    # 5. Export Regime Breakdown
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

    # 6. Export Visualizations
    Visualizer.export_results(results, importances, perf)
    println("Results exported to /octave directory.")

    # 7. Update Experiment Log
    try
        log_path = joinpath(pwd(), "EXPERIMENT_LOG.md")

        run_num = 1
        if isfile(log_path)
            run_num = sum(occursin("## Run", line) for line in readlines(log_path)) + 1
        end

        num_tickers = length(DataLoader.TICKERS)

        top_features_str = "N/A"
        valid_imps = []
        for i in 1:length(importances)
            try
                imp = importances[i]
                if isa(imp, Dict) push!(valid_imps, imp) end
            catch
                continue
            end
        end
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
                    "- Stock universe: $num_tickers tickers\n" *
                    "- Model: Ensemble (RF + XGBoost, 50/50 weighted average)\n" *
                    "- Training Window: $(Backtest.TRAIN_WINDOW_DAYS) days rolling\n" *
                    "- Test Step: $(Backtest.STEP_DAYS) days\n" *
                    "- Top N Stocks: $(Backtest.TOP_N_STOCKS)\n" *
                    "- RF n_trees: 100\n" *
                    "- RF n_subfeatures: $n_sub\n" *
                    "- RF max_depth: 10\n" *
                    "- XGBoost num_round: 300\n" *
                    "- XGBoost max_depth: 3\n" *
                    "- XGBoost eta (learning rate): 0.03\n" *
                    "- XGBoost min_child_weight: 3\n" *
                    "- XGBoost gamma: 0.1\n" *
                    "- XGBoost reg_lambda: 5\n" *
                    "- XGBoost subsample: 0.7\n" *
                    "- XGBoost colsample_bytree: 0.7\n" *
                    "- **HMM REGIME-SWITCHING SETUP**\n" *
                    "- Regime detector: Manual 2-state Gaussian HMM on SPY regime index\n" *
                    "- HMM training window: $(Regime.TRAIN_WINDOW) steps (rolling, no peeking)\n" *
                    "- Bear market hedge: Protective put (cost: $(Regime.BORROW_COST * 100)%, ATM strike)\n" *
                    "- Max positions bear: $(Regime.MAX_POSITIONS_BEAR)\n" *
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

if @isdefined(ARGS) && isempty(ARGS)
    main()
end