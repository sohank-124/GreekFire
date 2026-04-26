using MarketData, TimeSeries, DataFrames
ts = MarketData.yahoo(:AAPL)
println("Type: ", typeof(ts))
println("Fields: ", fieldnames(typeof(ts)))
println("First few rows of DataFrame(ts):")
println(first(DataFrame(ts), 5))
