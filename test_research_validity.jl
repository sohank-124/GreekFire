using Test, Dates, DataFrames

include("features.jl")
include("validation.jl")
include("backtest.jl")
include("metrics.jl")
include("data_loader.jl")

@testset "Wayback CSV loader compatibility (synthetic fixture)" begin
    raw = CSV.read(joinpath(@__DIR__, "tests", "fixtures", "wayback_prices.csv"), DataFrame)
    normalized = DataLoader._standardize_price_frame(raw)
    @test nrow(normalized) == 3
    @test normalized.date == [Date(2009, 1, 2), Date(2009, 1, 5), Date(2009, 1, 6)]
    @test normalized.close ≈ [51.0, 52.0, 53.0]
    @test normalized.open ≈ [50.0, 51.0, 52.0]
    @test normalized.volume == [1000.0, 1100.0, 1200.0]
end

@testset "Missing execution prices" begin
    partial = DataFrame(date=[Date(2020, 1, 1), Date(2020, 1, 2)],
                        open=[missing, 102.0], close=[101.0, missing],
                        high=[missing, 103.0], low=[missing, 100.0],
                        volume=[missing, 1000.0])
    for standardize in (DataLoader._standardize_price_frame, Features._standardize_price_df)
        cleaned = standardize(partial)
        @test nrow(cleaned) == 1
        @test isnan(cleaned.open[1])
        @test isnan(cleaned.high[1])
        @test isnan(cleaned.low[1])
        @test isnan(cleaned.volume[1])
    end
    dates = [Date(2020, 1, 1) + Day(i) for i in 0:90]
    prices = collect(101.0:191.0)
    no_open = DataFrame(date=dates, close=prices, volume=fill(1000.0, length(dates)))
    @test all(isnan, DataLoader._standardize_price_frame(no_open).open)
    @test all(isnan, Features.calculate_features(no_open, no_open).target)
    no_raw_close = DataFrame(date=[dates[1]], open=[100.0], adjclose=[50.0])
    @test isnan(DataLoader._standardize_price_frame(no_raw_close).open[1])
    with_raw_close = DataFrame(date=[dates[1]], open=[100.0], close=[100.0], adjclose=[50.0])
    @test DataLoader._standardize_price_frame(with_raw_close).open[1] ≈ 50.0
end

@testset "Position turnover and costs" begin
    turnover = Backtest.one_way_turnover
    @test turnover(String[], Float64[], ["A", "B"], [0.5, 0.5]) ≈ 1.0
    @test turnover(["A", "B"], [0.5, 0.5], ["A", "B"], [0.5, 0.5]) ≈ 0.0
    @test turnover(["A", "B"], [0.5, 0.5], ["A", "C"], [0.5, 0.5]) ≈ 0.5
    @test 0.02 - (10 + 5) / 10_000 * 0.5 ≈ 0.01925
end

@testset "Ensemble weighting" begin
    @test Backtest.ensemble_predictions([1.0, 0.0], [0.0, 1.0], 0.5) ≈ [0.5, 0.5]
    @test Backtest.ensemble_predictions([1.0], [0.0], 0.75) ≈ [0.75]
    @test Backtest.ensemble_predictions([1.0], nothing, 0.75) ≈ [1.0]
    @test_throws ArgumentError Backtest.ensemble_predictions([1.0], [0.0], 1.5)
end

@testset "Forward label availability" begin
    dates = [Date(2020, 1, 1) + Day(i) for i in 0:220
             if dayofweek(Date(2020, 1, 1) + Day(i)) <= 5 &&
                !(Date(2020, 5, 20) <= Date(2020, 1, 1) + Day(i) <= Date(2020, 5, 31))]
    prices = [100.0 + 0.08 * i + 2 * sin(i / 7) for i in eachindex(dates)]
    data = DataFrame(date=dates, close=prices, open=prices,
                     volume=fill(1000.0, length(dates)))
    features = Features.calculate_features(data, data)
    @test nrow(features) > 0
    @test all(features.target_end_date .> features.date)
    @test any(row -> row.date < Date(2020, 5, 20) &&
                     row.target_end_date > Date(2020, 5, 31), eachrow(features))
    train_end = Date(2020, 5, 19)
    @test all(row -> row.target_end_date <= train_end,
              eachrow(filter(row -> row.target_end_date <= train_end, features)))
    @test any(row -> row.date <= train_end && row.target_end_date > train_end,
              eachrow(features))
    features.ticker .= "TEST"
    step = (train_start=dates[1], train_end=train_end)
    selected = Backtest.training_rows(features, step, Set(["TEST"]))
    @test nrow(selected) > 0
    @test all(selected.target_end_date .<= train_end)
    @test isempty(Backtest.training_rows(features, step, Set(["OTHER"])))
end

@testset "Benchmark execution alignment" begin
    dates = [Date(2020, 1, 6) + Day(i) for i in 0:10]
    prices = DataFrame(date=dates, open=collect(100.0:110.0),
                       close=collect(101.0:111.0))
    @test _execution_return(prices, dates[1], dates[6]) ≈ 106.0 / 101.0 - 1
    @test ismissing(_execution_return(prices, dates[1], dates[5]))
    @test ismissing(_execution_return(nothing, dates[1], dates[6]))
    @test ismissing(_execution_return(prices, dates[8], dates[11]))
    result = (date=Date(2020, 1, 6), train_start=Date(2019, 1, 1),
              train_end=Date(2020, 1, 5))
    @test length(filter_holdout([result]; holdout_start=Date(2020, 1, 1))) == 1
end

@testset "Gross and net reporting" begin
    sample = [(actual_alpha_long=0.03, net_alpha_long=0.0285,
               actual_equal_weight_long=0.02, net_alpha_linear=0.0185,
               chosen_alpha=0.0285)]
    @test Metrics.compute_performance(sample; signal=:base).mean_return ≈ 0.0285
    @test Metrics.compute_performance(sample; signal=:gross).mean_return ≈ 0.03
    @test Metrics.compute_performance(sample; signal=:linear).mean_return ≈ 0.0185
end

@testset "Incomplete validation rejected" begin
    @test assert_complete_validation([(date=Date(2020, 1, 6),)]) === nothing
    @test_throws ErrorException assert_complete_validation([nothing])
end
