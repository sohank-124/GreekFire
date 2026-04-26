using Pkg
Pkg.activate(".")
Pkg.add(["MarketData", "DataFrames", "CSV", "Dates", "Statistics", "LinearAlgebra", "Distributions", "MLJ", "DecisionTree", "MLJDecisionTreeInterface", "MLJLinearModels", "ProgressMeter", "JSON3", "TimeSeries", "HTTP"])

include("data_loader.jl")
include("features.jl")

using DataFrames, Dates

# Setup
DataLoader.load_ticker_data("AAPL")
DataLoader.load_ticker_data("SPY")

aapl = DataLoader.load_ticker_data("AAPL")
spy = DataLoader.load_ticker_data("SPY")

# Ensure alignment
common_dates = intersect(aapl.date, spy.date)
aapl = aapl[in.(aapl.date, Ref(common_dates)), :]
spy = spy[in.(spy.date, Ref(common_dates)), :]

# Sort by date
sort!(aapl, :date)
sort!(spy, :date)

# Compute features
feat_df = Features.calculate_features(aapl, spy)

# Filter for 2009
res = feat_df[(feat_df.date .>= Date(2009, 1, 1)) .& (feat_df.date .<= Date(2009, 12, 31)), :]

println(res)
