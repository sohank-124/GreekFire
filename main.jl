include("data_loader.jl")
include("features.jl")
include("model.jl")
include("backtest.jl")
include("metrics.jl")
include("visualizer.jl")

using DataFrames, Dates, Statistics

DEBUG_MODE = false

function main()
    println("Running on $(Threads.nthreads()) threads")
    println("Starting GreekFire Walk-Forward Backtesting Engine...")
    
    # 1. Load Data
    all_stock_data = DataLoader.load_all_data()
    spy_data = all_stock_data["SPY"]
    
    if DEBUG_MODE
        println("=== DEBUG MODE: Running first 5 steps sequentially ===")
        results, importances = Backtest.run_walk_forward_debug(all_stock_data, spy_data)
    else
        # 2. Run Backtest
        results, importances = Backtest.run_walk_forward(all_stock_data, spy_data)
        
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
    
    # 3. Compute Metrics
    perf = Metrics.compute_performance(results)
    
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
    
    # 4. Export Visualizations
    Visualizer.export_results(results, importances, perf)
    println("Results exported to /octave directory.")
    
    # 5. Update Experiment Log
    try
        log_path = joinpath(pwd(), "EXPERIMENT_LOG.md")
        
        # Calculate run number
        run_num = 1
        if isfile(log_path)
            run_num = sum(occursin("## Run", line) for line in readlines(log_path)) + 1
        end
        
        # Gather info for log
        num_tickers = length(DataLoader.TICKERS)
        
        # Get top 3 features for the log
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
            top_features_str = join([ "$(sorted[i][1]) ($(round(sorted[i][2], digits=4)))" for i in 1:min(3, length(sorted))], ", ")
        end
        
        # Calculate n_subfeatures (same logic as model.jl)
        # Need a feature_cols length. We can get it from first valid importance dict.
        n_sub = "N/A"
        if !isempty(valid_imps)
            n_feat = length(collect(keys(valid_imps[1])))
            n_sub = max(3, floor(Int, sqrt(n_feat)))
        end

        log_entry = "\n## Run $run_num — $(Dates.format(today(), "yyyy-mm-dd"))\n" *
                    "- Date range: $(DataLoader.START_DATE) to present\n" *
                    "- Stock universe: $num_tickers tickers\n" *
                    "- Model: Random Forest\n" *
                    "- n_subfeatures: $n_sub\n" *
                    "- max_depth: 10\n" *
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