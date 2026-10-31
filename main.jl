include("data_loader.jl")
include("features.jl")
include("model.jl")
include("backtest.jl")
include("metrics.jl")
include("visualizer.jl")
# Archived regime.jl/hmm.jl are not loaded by the supported engine.
include("validation.jl")

using DataFrames, Dates, Statistics, Random

const DEBUG_MODE = false

# Note: `run_validation()` is defined by the include of validation.jl above.
# It is dispatched through the ARGS[1] == "validate" branch at the bottom of main.jl.

function main()
    Random.seed!(42)
    println("Running on $(Threads.nthreads()) threads")
    println("Starting archived GreekFire walk-forward research engine (regime switching disabled)...")

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
                results[i] = merge(r, (chosen_alpha = r.net_alpha_long, regime_signal = :bull))
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
        label = s == :base ? "base (RF+XGB long, net)" : s == :linear ? "linear (equal-weight, net)" : "chosen (long-only, net)"
        if p !== nothing
            println("$(label):")
            println("  CAGR: $(round(p.cagr * 100, digits=2))%")
            println("  Sharpe: $(round(p.sharpe, digits=2))")
            println("  Max DD: $(round(p.max_dd * 100, digits=2))%")
            println("  Win Rate: $(round(p.win_rate * 100, digits=2))%")
            println("  Mean Period Return: $(round(p.mean_return * 100, digits=2))%")
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
    println("Results exported to results/corrected/.")

    try
        log_path = joinpath(pwd(), "results", "corrected", "RUN_LOG.md")

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
            "- Mean Period Return Return: $(round(perf.mean_return * 100, digits=2))%\n" *
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

# ---------------------------------------------------------------------------
# CLI argument parser (Phase 1 - experiment support)
# ---------------------------------------------------------------------------
# Parse --key=value pairs from ARGS[2:end] into a Dict{String,Any} suitable
# for splatting into run_validation's keyword arguments.
#
# Supported keys:
#   --top-n=<Int>              (alias: --top_n)
#   --universe=<String>        ("all" | "tech" | "semis")
#   --exclude-features=<a,b,c> (comma-separated, no spaces)
#   --inv-vol-exp=<Float64>    (alias: --inv_vol_exp)
#   --exclude-tickers=<a,b,c>  (alias: --exclude_tickers)
#   --holdout-start=<YYYY-MM-DD>  (alias: --holdout_start)
#   --train-window=<Symbol|Int>   (alias: --train_window; :fixed|:expanding|Int days)
#   --max-steps=<Int>          (alias: --max_steps)
#   --weighting-method=<equal|shifted|shifted_inv_vol|rank|rank_inv_vol>
#   --transaction-cost-bps=<Float64>  --slippage-bps=<Float64>
#   --rf-weight=<Float64>       RF share of RF/XGBoost blend, default 0.5
#
# Unrecognized args are ignored with a warning so future experiments can
# safely add new keys without breaking older scripts.
function parse_cli_args(args::Vector{String})
    parsed = Dict{String,Any}()
    aliases = Dict{String,String}(
        "top-n"             => "top_n",
        "top_n"             => "top_n",
        "universe"          => "universe",
        "exclude-features"  => "exclude_features",
        "exclude_features"  => "exclude_features",
        "inv-vol-exp"       => "inv_vol_exp",
        "inv_vol_exp"       => "inv_vol_exp",
        "exclude-tickers"   => "exclude_tickers",
        "exclude_tickers"   => "exclude_tickers",
        "holdout-start"     => "holdout_start",
        "holdout_start"     => "holdout_start",
        "train-window"      => "train_window",
        "train_window"      => "train_window",
        "max-steps"         => "max_steps",
        "max_steps"         => "max_steps",
        "weighting-method"  => "weighting_method",
        "transaction-cost-bps" => "transaction_cost_bps",
        "slippage-bps"      => "slippage_bps",
        "rf-weight"         => "ensemble_rf_weight",
        "prediction-mode"   => "prediction_mode",
    )

    for arg in args
        if !startswith(arg, "--")
            @warn "Ignoring non-flag CLI argument: $(repr(arg))"
            continue
        end
        kv = split(arg, "="; limit=2)
        length(kv) != 2 && begin
            @warn "Ignoring malformed CLI argument (expected --key=value): $(repr(arg))"
            continue
        end
        key = String(kv[1][3:end])  # strip leading "--"
        val = String(kv[2])
        haskey(aliases, key) || begin
            @warn "Unknown CLI argument: --$(key) (ignored)"
            continue
        end
        canonical = aliases[key]
        # Don't overwrite an earlier value (first-write-wins keeps behavior
        # deterministic if the same key is supplied twice).
        haskey(parsed, canonical) && begin
            @warn "Duplicate CLI argument --$(key); keeping first value"
            continue
        end

        try
            if canonical == "top_n"
                parsed[canonical] = parse(Int, val)
            elseif canonical == "prediction_mode"
                val in ("ensemble", "momentum") || error("Unknown prediction mode")
                parsed[canonical] = Symbol(val)
            elseif canonical == "universe"
                lowered = lowercase(strip(val))
                lowered in ("all", "tech", "semis") || error("universe must be one of all/tech/semis, got $(repr(val))")
                parsed[canonical] = lowered
            elseif canonical == "exclude_features"
                items = filter(!isempty, split(val, ","))
                parsed[canonical] = Set{String}(String(s) for s in items)
            elseif canonical == "inv_vol_exp"
                parsed[canonical] = parse(Float64, val)
            elseif canonical == "exclude_tickers"
                items = filter(!isempty, split(val, ","))
                parsed[canonical] = Set{String}(uppercase(String(s)) for s in items)
            elseif canonical == "holdout_start"
                parsed[canonical] = Date(val)  # ISO YYYY-MM-DD
            elseif canonical == "train_window"
                lowered = lowercase(strip(val))
                if lowered in ("fixed", "expanding")
                    parsed[canonical] = Symbol(lowered)
                else
                    parsed[canonical] = parse(Int, val)
                end
            elseif canonical == "max_steps"
                parsed[canonical] = parse(Int, val)
            elseif canonical == "weighting_method"
                val in ("equal", "shifted", "shifted_inv_vol", "rank", "rank_inv_vol") || error("Unknown weighting method")
                parsed[canonical] = val
            elseif canonical in ("transaction_cost_bps", "slippage_bps", "ensemble_rf_weight")
                parsed[canonical] = parse(Float64, val)
            end
        catch e
            @warn "Failed to parse CLI argument --$(key)=$(repr(val)): $(sprint(showerror, e))"
        end
    end

    return parsed
end

if isempty(ARGS)
    main()
end

# Previously explored historical validation harness.
# Usage:
#     julia --project=. main.jl validate
#     julia --project=. main.jl validate --top-n=3 --universe=tech
#     GREEKFIRE_MAX_STEPS=20 julia --project=. main.jl validate
if !isempty(ARGS) && ARGS[1] == "validate"
    if !@isdefined run_validation
        # ensure the include actually defined the function
        include("validation.jl")
    end
    Random.seed!(42)
    cli_kwargs = parse_cli_args(ARGS[2:end])
    if !isempty(cli_kwargs)
        println("[main] CLI kwargs: $(cli_kwargs)")
    end
    # Julia kwarg splat requires a NamedTuple or iterable of Pair{Symbol,...},
    # not a Dict. Convert Dict{String,Any} -> NamedTuple via Symbol keys.
    if isempty(cli_kwargs)
        run_validation()
    else
        kw_pairs = (Symbol(k) => v for (k, v) in cli_kwargs)
        run_validation(; kw_pairs...)
    end
end
