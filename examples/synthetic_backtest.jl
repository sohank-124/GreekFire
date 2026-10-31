# Full deterministic model/portfolio integration, using invented prices only.
using Dates, DataFrames, Test, CSV, Statistics
if !isdefined(Main, :Backtest)
    include(joinpath(@__DIR__, "..", "backtest.jl"))
    include(joinpath(@__DIR__, "..", "metrics.jl"))
end
dates_demo = [Date(2020,1,1)+Day(i) for i in 0:550
              if dayofweek(Date(2020,1,1)+Day(i)) <= 5]
function synthetic_prices(k)
    x = collect(eachindex(dates_demo))
    p = 100 .* exp.(0.0002 .* x .+ 0.03 .* sin.(x ./ (5+k/3) .+ k))
    DataFrame(date=dates_demo, open=p .* (1 .+ 0.001 .* cos.(x)), close=p,
              volume=10000 .+ 100 .* sin.(x ./ 3 .+ k))
end
stocks_demo = Dict("S$(k)"=>synthetic_prices(k) for k in 1:12)
spy_demo = synthetic_prices(20)
membership_demo = _ -> sort(collect(keys(stocks_demo)))
@testset "Synthetic CSV loading" begin
    # Exercise the actual loader in an isolated directory. No local market
    # input is changed and temporary files are removed after the check.
    fixture_root = joinpath(@__DIR__, "..", "results", "synthetic")
    mkpath(fixture_root)
    mktempdir(fixture_root) do temp
        mkpath(joinpath(temp, "data"))
        CSV.write(joinpath(temp, "data", "SPY.csv"), spy_demo)
        for (ticker, prices) in stocks_demo
            CSV.write(joinpath(temp, "data", "$ticker.csv"), prices)
        end
        CSV.write(joinpath(temp, "sp500_ticker_start_end.csv"), DataFrame(
            ticker=sort(collect(keys(stocks_demo))),
            start_date=fill(first(dates_demo), 12), end_date=fill(missing, 12)))
        loader = Backtest.DataLoader
        previous_cache = loader._SP500_CACHE[]
        try
            loader._SP500_CACHE[] = nothing
            cd(temp) do
                loaded = loader.load_all_data()
                @test Set(keys(loaded)) == Set(keys(stocks_demo))
                @test loaded["S1"].close ≈ stocks_demo["S1"].close
                @test loader.load_ticker_data("SPY").open ≈ spy_demo.open
            end
        finally
            loader._SP500_CACHE[] = previous_cache
        end
    end
end
demo_time = @elapsed demo_results, demo_importances, _, _, _ = Backtest.run_walk_forward(
    stocks_demo, spy_demo; train_window=40, top_n=4, max_steps=2,
    weighting_method="equal", max_position_cap=0.2, membership_provider=membership_demo)
@testset "Full synthetic walk-forward" begin
    @test length(demo_results) == 2
    @test all(r -> r !== nothing, demo_results)
    for r in demo_results
        s = findfirst(==(r.test_start), dates_demo)
        @test r.test_end == dates_demo[s+5]
        @test r.train_end < r.test_start
        @test maximum(r.inv_vol_weights) <= 0.2 + 1e-10
        @test sum(r.inv_vol_weights) <= 1
        @test length(r.tickers_long) == 3 # strict unknown-sector cap
        expected = sum(w * (stocks_demo[t].close[s+5]/stocks_demo[t].open[s+1]-1)
                       for (t,w) in zip(r.tickers_long, r.inv_vol_weights))
        @test r.actual_alpha_long ≈ expected
        @test r.net_alpha_long ≈ expected - 0.0015*r.long_turnover
        @test r.long_turnover ≈ sum(r.inv_vol_weights) + sum(r.inv_vol_weights) + expected
        @test r.spy_ret ≈ spy_demo.close[s+5]/spy_demo.open[s+1]-1
        @test r.period_scores_copy.ticker[r.long_indices_copy] == r.tickers_long
        @test r.ic ≈ cor(r.period_scores_copy.avg_pred, r.period_scores_copy.avg_actual)
        @test r.rank_ic ≈ cor(Backtest.tiedrank(r.period_scores_copy.avg_pred),
                             Backtest.tiedrank(r.period_scores_copy.avg_actual))
        ordered = sort(r.period_scores_copy, :avg_pred; rev=true)
        @test r.decile_spread ≈ ordered.avg_actual[1] - mean(ordered.avg_actual[11:12])
    end
    @test demo_results[1].test_end == demo_results[2].test_start
end
if abspath(PROGRAM_FILE) == @__FILE__
    out = joinpath(@__DIR__, "..", "results", "synthetic")
    mkpath(out)
    CSV.write(joinpath(out,"steps.csv"), DataFrame([
        (date=r.date, gross=r.actual_alpha_long, net=r.net_alpha_long,
         spy=r.spy_ret, rank_ic=r.rank_ic) for r in demo_results]))
    include(joinpath(@__DIR__, "..", "visualizer.jl"))
    Visualizer.export_results(demo_results, demo_importances,
        Metrics.compute_performance(demo_results); output_dir=joinpath(out, "diagnostics"))
    equity_demo = CSV.read(joinpath(out, "diagnostics", "equity_curve.csv"), DataFrame)
    @test equity_demo.spy_benchmark ≈ vcat(1.0, cumprod(1 .+ [r.spy_ret for r in demo_results]))
    println("Synthetic integration elapsed seconds: ", demo_time)
    println("Invented prices test mechanics only; these are not market research results.")
end
