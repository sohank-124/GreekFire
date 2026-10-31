using Test, Dates, DataFrames, Statistics
include(joinpath(@__DIR__, "..", "test_research_validity.jl"))

@testset "Historical membership and security identity" begin
    original = DataLoader._SP500_CACHE[]
    try
        DataLoader._SP500_CACHE[] = DataFrame(ticker=["OLD", "NEW", "OLD"],
            start_date=["2020-01-01", "2020-02-01", "2021-01-01"],
            end_date=Union{Missing,String}["2020-01-31", missing, "2021-02-01"])
        @test DataLoader.get_sp500_constituents(Date(2020,1,15)) == ["OLD"]
        @test DataLoader.get_sp500_constituents(Date(2020,2,1)) == ["NEW"]
        @test Set(DataLoader.get_sp500_constituents(Date(2021,1,2))) == Set(["NEW", "OLD"])
        @test DataLoader._to_end_date(missing) == Date(2999,12,31)
        @test_throws ArgumentError DataLoader._to_end_date("bad-date")
        @test DataLoader.resolve_ticker(" atvi ") == "ATVI"
        @test DataLoader.resolve_ticker("GOOG") == "GOOG"
        @test DataLoader.resolve_ticker("CELG") == "CELG"
    finally
        DataLoader._SP500_CACHE[] = original
    end
end

@testset "Independent metric arithmetic" begin
    f = Metrics.return_metrics
    m = f([0.1, -0.2]; periods_per_year=2, benchmark=[0.02, -0.01])
    @test m.equity_curve ≈ [1, 1.1, 0.88]
    @test m.cagr ≈ -0.12
    @test m.max_dd ≈ 0.2
    @test m.win_rate == 0.5
    @test m.mean_return ≈ -0.05
    @test m.avg_excess ≈ -0.055
    @test m.sharpe ≈ mean([0.1,-0.2])/std([0.1,-0.2])*sqrt(2)
    @test m.benchmark_relative_return ≈ 0.88/(1.02*0.99)-1
    @test f([-1.0]).max_dd == 1
    @test f([-1.0]).cagr == -1
    @test isnan(f([0.0]).sharpe)
    @test isnan(f([0.01]).avg_excess)
    @test f([0.1,0.2]; periods_per_year=2, risk_free_rate=0.21).sharpe ≈ 1
    @test_throws ArgumentError f([NaN])
    @test_throws ArgumentError f([-1.01])
    @test_throws ArgumentError f([0.1]; benchmark=[0.1,0.2])
    @test_throws ErrorException Metrics.compute_performance([nothing])
    @test Backtest.percentile_rank([2.,2.,4.]) ≈ [0.25,0.25,0.98]
    ranked = DataFrame(date=[Date(2020,1,1), Date(2020,1,1), Date(2020,1,2)],
                       value=[1.,3.,9.])
    Backtest.rank_columns!(ranked, [:value])
    @test ranked.value ≈ [0.02,0.98,0.5]
end

include(joinpath(@__DIR__, "..", "examples", "synthetic_backtest.jl"))

@testset "Causal features and missing outcomes" begin
    original = Features.calculate_features(stocks_demo["S1"], spy_demo)
    altered_prices = copy(stocks_demo["S1"])
    altered_prices.close[150:end] .*= 2
    altered_prices.open[150:end] .*= 2
    altered = Features.calculate_features(altered_prices, spy_demo)
    predictors = setdiff(names(original), ["target", "target_end_date"])
    past = original.date .< dates_demo[150]
    @test original[past, predictors] == altered[past, predictors]
    @test maximum(original.date) == dates_demo[end]
    @test all(isnan, original.target[end-4:end])
    gapped = stocks_demo["S1"][setdiff(1:length(dates_demo), [107]), :]
    gap_features = Features.calculate_features(gapped, spy_demo)
    row = only(eachrow(gap_features[gap_features.date .== dates_demo[102], :]))
    @test row.target_end_date == dates_demo[107]
    @test isnan(row.target)
    incomplete = copy(stocks_demo)
    incomplete["S1"] = gapped
    @test_throws ErrorException Backtest.run_walk_forward(incomplete, spy_demo;
        train_window=40, top_n=3, max_steps=2, prediction_mode=:momentum,
        membership_provider=membership_demo)
    @test_throws ArgumentError Features.calculate_features(spy_demo, spy_demo; forecast_horizon=0)
    @test_throws ErrorException DataLoader._standardize_price_frame(vcat(spy_demo, spy_demo[1:1,:]))
end
