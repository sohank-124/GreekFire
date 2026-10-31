# Deterministic mechanics demonstration: no market data, training, or backtest.
using Dates, DataFrames, Test
include(joinpath(@__DIR__, "..", "features.jl"))

dates = [Date(2020, 1, 1) + Day(i) for i in 0:220
         if dayofweek(Date(2020, 1, 1) + Day(i)) <= 5]
prices = [100.0 + 0.08 * i + 2 * sin(i / 7) for i in eachindex(dates)]
stock = DataFrame(date=dates, close=prices, open=prices .+ 0.25,
                  volume=fill(1000.0, length(dates)))
features = Features.calculate_features(stock, stock)

@testset "Synthetic feature and label mechanics" begin
    @test nrow(features) > 0
    first_row = features[1, :]
    index = findfirst(==(first_row.date), stock.date)
    @test first_row.target ≈ stock.close[index + 5] / stock.open[index + 1] - 1
    @test first_row.target_end_date == stock.date[index + 5]

    cutoff = features.date[20]
    eligible = filter(row -> row.target_end_date <= cutoff, features)
    @test nrow(eligible) > 0
    @test all(eligible.target_end_date .<= cutoff)
    @test any(row -> row.date <= cutoff < row.target_end_date, eachrow(features))

    # Verify that unknown future outcomes do not remove signals.
    @test maximum(features.date) == stock.date[end]
    absent_open = copy(stock)
    absent_open.open .= NaN
    @test all(isnan, Features.calculate_features(absent_open, stock).target)
end

println("Synthetic observations: $(nrow(stock)); signal feature rows: $(nrow(features))")
println("Signals use the current close; labels enter at next open and exit five sessions later.")
println("Training must wait until each label's target_end_date.")
println("Final signals remain present even when their future targets are unavailable.")
println("No model was trained, no backtest was run, and no files were written.")
