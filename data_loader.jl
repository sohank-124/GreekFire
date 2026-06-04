module DataLoader

using MarketData, DataFrames, CSV, Dates, JSON3, HTTP, TimeSeries

const TICKERS = [
    "XOM",  "MSFT", "WMT",  "PG",   "JNJ",  "AAPL", "GE",   "GOOGL", "BRK-B", "PFE",
    "T",    "JPM",  "KO",   "BAC",  "CVX",  "IBM",  "VZ",   "WFC",  "MRK",   "CSCO",
    "PEP",  "INTC", "MCD",  "HD",   "BA",   "MMM",  "AXP",  "C",    "DIS",   "TGT",
    "NKE",  "CMCSA","HON",  "EMR",  "LMT",  "GD",   "NOC",  "GS",   "MS",    "USB",
    "PNC",  "COF",  "BK",   "STT",  "MET",  "AIG",  "TRV",  "PGR",  "UNH",   "CI",
    "HUM",  "AFL",  "CINF", "L",    "PRU",  "ALL",  "SCHW", "BLK",  "BEN",   "TROW",
    "IVZ",  "NTRS", "DFS",  "SYF",  "COP",  "EOG",  "SLB",  "HAL",  "NOV",   "OXY",
    "MRO",  "DVN",  "CHK",  "RRC",  "SWN",  "EQT",  "AR",   "PXD",  "HPQ",   "ORCL",
    "ACN",  "TMO",  "DHR",  "NEE",  "TXN",  "QCOM", "AMGN", "BMY",  "ABT",   "MDT",
    "F",    "CMI",  "DE",   "CAT",  "UPS",  "FDX",  "LUV",  "DAL",  "UAL",   "SPY"
]
const START_DATE = Date(2009, 1, 1)

const TICKER_REMAP = Dict(
    "UTX"   => "RTX",
    "FB"    => "META",
    "GOOG"  => "GOOGL",
    "BRK.A" => "BRK-B",
    "TWX"   => "WBD",
    "ALTR"  => "INTC",
    "FISV"  => "FIS",
)

const TICKER_BLACKLIST = Set([
    "CHK", "RRC", "SWN", "EQT", "AR", "NOV", "HAL", "MRO", "DVN",
    "TWX", "WBD", "TWTR", "LEH", "BSC", "WM", "FNM", "FRE", "WPO",
    "APSO", "KSO", "MRSH", "SYF", "DFS", "NTRS", "IVZ", "BEN",
    "CINF", "L", "PRU", "AFL", "HUM", "CI", "STT", "BK", "LUV",
    "DAL", "UAL", "F", "CMI", "HPQ", "ORCL", "NOC", "GD", "LMT",
    "BEN", "TROW", "COF", "C"
])

function get_data_path(ticker::String)
    return joinpath("data", ticker * ".csv")
end

function fetch_yahoo_finance_json(ticker::String)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" * ticker *
          "?period1=$(Int(Dates.value(DateTime(START_DATE)) * 86400))" *
          "&period2=$(Int(Dates.value(DateTime(today())) * 86400))&interval=1d"
    try
        response = HTTP.get(url)
        data = JSON3.read(response.body)
        result = data.chart.result[1]
        timestamps = result.timestamp
        indicators = result.indicators[1]
        close  = indicators.quote[1].close
        open_  = indicators.quote[1].open
        high   = indicators.quote[1].high
        low    = indicators.quote[1].low
        volume = indicators.quote[1].volume
        dates  = [Date(Dates.unix2datetime(t)) for t in timestamps]
        df = DataFrame(
            date   = dates,
            open   = Float64.(open_),
            high   = Float64.(high),
            low    = Float64.(low),
            close  = Float64.(close),
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
        if hasproperty(df, :timestamp)
            DataFrames.rename!(df, :timestamp => :date)
        end
    catch e
        @warn "MarketData.jl failed for $ticker, trying Yahoo JSON fallback: $e"
        df = fetch_yahoo_finance_json(ticker)
    end

    if df !== nothing
        DataFrames.rename!(df, [lowercase(name) for name in names(df)])
        # Replace nothing with missing so CSV.write doesn't choke
        for col in names(df)
            if any(x -> x === nothing, df[!, col])
                df[!, col] = [x === nothing ? missing : x for x in df[!, col]]
            end
        end
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

function resolve_ticker(ticker::AbstractString)
    get(TICKER_REMAP, String(ticker), String(ticker))
end

const _SP500_CACHE = Ref{Union{Nothing, DataFrame}}(nothing)

function _get_sp500_df()
    if _SP500_CACHE[] === nothing
        csv_path = joinpath("data", "sp500_ticker_start_end.csv")
        if !isfile(csv_path)
            return nothing
        end
        _SP500_CACHE[] = CSV.read(csv_path, DataFrame)
    end
    return _SP500_CACHE[]
end

function get_sp500_constituents(date::Date)
    df = _get_sp500_df()
    if df === nothing
        @warn "S&P 500 constituents CSV not found. Falling back to hardcoded TICKERS."
        return TICKERS
    end

    active = String[]
    for row in eachrow(df)
        s = isa(row.start_date, Date) ? row.start_date : Date(string(row.start_date))
        if ismissing(row.end_date) || row.end_date === nothing || strip(string(row.end_date)) == ""
            e = today()
        else
            e = isa(row.end_date, Date) ? row.end_date : Date(string(row.end_date))
        end
        if s <= date <= e
            push!(active, row.ticker)
        end
    end
    return active
end

function get_all_sp500_tickers()
    df = _get_sp500_df()
    if df === nothing
        @warn "S&P 500 constituents CSV not found. Falling back to hardcoded TICKERS."
        return TICKERS
    end
    today_date = today()
    relevant = String[]

    for row in eachrow(df)
        s = isa(row.start_date, Date) ? row.start_date : Date(string(row.start_date))
        if ismissing(row.end_date) || row.end_date === nothing || strip(string(row.end_date)) == ""
            e = today_date
        else
            e = isa(row.end_date, Date) ? row.end_date : Date(string(row.end_date))
        end
        # Overlap check: ticker was in S&P 500 at some point during [START_DATE, today]
        if s <= today_date && e >= START_DATE
            resolved = resolve_ticker(row.ticker)
            resolved ∈ TICKER_BLACKLIST && continue
            push!(relevant, resolved)
        end
    end

    return unique(relevant)
end

function load_all_data()
    tickers = get_all_sp500_tickers()
    all_data = Dict{String, DataFrame}()
    for ticker in tickers
        resolved = resolve_ticker(ticker)
        resolved ∈ TICKER_BLACKLIST && continue
        df = load_ticker_data(resolved)
        if df !== nothing
            all_data[resolved] = df
        end
    end
    return all_data
end

end