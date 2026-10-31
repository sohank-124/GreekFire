using Dates, DataFrames, Statistics
include(joinpath(pwd(), "features.jl"))
dates = [Date(2010,1,1)+Day(i) for i in 0:5000 if dayofweek(Date(2010,1,1)+Day(i))<=5]
x = collect(eachindex(dates))
p = 100 .* exp.(0.0001 .* x .+ 0.02 .* sin.(x ./ 7))
df = DataFrame(date=dates, close=p, open=p.*1.001, volume=1000 .+ x)
Features.calculate_features(df, df)
t = [@elapsed Features.calculate_features(df, df) for _ in 1:5]
println("feature_seconds=", t, "; median=", median(t))
