module DataLoader

using MarketData, DataFrames, CSV, Dates, JSON3, HTTP, TimeSeries

const TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "TSLA", "AMD", "INTC", "DIS", "NFLX", "KO",
    "WMT", "BRK-B", "UNH", "LLY", "V", "MA", "AVGO", "HD", "MRK", "ABBV",
    "PEP", "COST", "ADBE", "CSCO", "ACN", "MCD", "NKE", "TMO", "DHR", "NEE",
    "TXN", "QCOM", "HON", "PM", "RTX", "T", "VZ", "AMGN", "IBM", "CAT", "GE",
    "BA", "MMM", "CVS", "MDLZ", "GILD", "ISRG", "REGN", "VRTX", "SLB", "COP",
    "EOG", "PGR", "TRV", "AIG", "MET", "AXP", "BLK", "SCHW", "CME", "ICE",
    "USB", "PNC", "TFC", "COF", "SPG", "PLD", "AMT", "CCI", "EQIX", "DLR",
    "UPS", "FDX", "CSX", "NSC", "DE", "EMR", "ETN", "ITW", "PH", "ROK",
    "SBUX", "TGT", "LOW", "ROST", "TJX", "EL", "CL", "PG", "KMB", "GIS", "SPY"
]
const START_DATE = Date(2009, 1, 1)

function get_data_path(ticker::String)
    return joinpath("data", ticker * ".csv")
end

function fetch_yahoo_finance_json(ticker::String)
    # Yahoo Finance JSON API often requires a crumb. This is a simplified attempt.
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" * ticker * "?period1=$(Int(Dates.value(DateTime(START_DATE)) * 86400))&period2=$(Int(Dates.value(DateTime(today())) * 86400))&interval=1d"
    try
        response = HTTP.get(url)
        data = JSON3.read(response.body)
        result = data.chart.result[1]
        
        timestamps = result.timestamp
        indicators = result.indicators[1]
        
        # Extract columns
        close = indicators.quote[1].close
        open = indicators.quote[1].open
        high = indicators.quote[1].high
        low = indicators.quote[1].low
        volume = indicators.quote[1].volume
        
        # Convert timestamps to Dates
        dates = [Date(Unix.datetime(t)) for t in timestamps]
        
        # Handle NaNs in Yahoo JSON (they are often nulls)
        # In Julia, we can use filter! or just accept NaNs if we convert them
        
        df = DataFrame(
            date = dates,
            open = Float64.(open),
            high = Float64.(high),
            low = Float64.(low),
            close = Float64.(close),
            volume = Float64.(volume)
        )
        return df
    catch e
        @warn "Failed to fetch $ticker from Yahoo JSON: $e"
        return nothing
    end
end

function load_ticker_data(ticker::String)
    path = get_data_path(ticker)
    if isfile(path)
        return CSV.read(path, DataFrame)
    end

    @info "Fetching data for $ticker..."
    df = nothing
    try
        ts = MarketData.yahoo(Symbol(ticker))
        df = DataFrame(ts)
        # The index column is named 'timestamp' in the resulting DataFrame
        if hasproperty(df, :timestamp)
            DataFrames.rename!(df, :timestamp => :date)
        end
    catch e
        @warn "MarketData.jl failed for $ticker, trying Yahoo JSON fallback: $e"
        df = fetch_yahoo_finance_json(ticker)
    end

    if df !== nothing
        # Normalize column names to lowercase
        DataFrames.rename!(df, [lowercase(name) for name in names(df)])
        println("Columns for $ticker: ", names(df))
        # Filter for start date
        if hasproperty(df, :date)
            df = df[df.date .>= START_DATE, :]
        end
        CSV.write(path, df)
        return df
    else
        @error "Could not retrieve data for $ticker"
        return nothing
    end
end

function load_all_data()
    all_data = Dict{String, DataFrame}()
    for ticker in TICKERS
        df = load_ticker_data(ticker)
        if df !== nothing
            all_data[ticker] = df
        end
    end
    return all_data
end

end
