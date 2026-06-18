module DataLoader

using CSV, DataFrames, Dates, Statistics

export START_DATE, resolve_ticker, get_sp500_constituents, get_all_sp500_tickers,
       load_ticker_data, load_all_data, TICKER_REMAP, TICKER_BLACKLIST

const START_DATE = Date(2000, 1, 3)

const TICKER_REMAP = Dict{String, String}(
    "FB"   => "META",
    "GOOG" => "GOOGL",
    "UTX"  => "RTX",
    "RTN"  => "RTX",
    "TWTR" => "TWTR",
    "CME"  => "CME",
    "LBRDK" => "LBRDK",
    "LBRDA" => "LBRDA",
    "SPLK" => "SPLK",
    "KHC"  => "KHC",
    "MNST" => "MNST",
    "XRX"  => "XRX",
    "NBL"  => "NBL",
    "PCLN" => "BKNG",
    "WAG"  => "WBA",
    "CELG" => "BMY",
    "AGN"  => "ABBV",
    "HRS"  => "LHX",
    "DOW"  => "DOW",
    "NAVI" => "NAVI",
    "UAA"  => "UA",
    "UA"   => "UA",
    "LUMN" => "LUMN",
    "SYMC" => "GEN",
    "VIAB" => "PARA",
    "CBS"  => "PARA",
    "DISCA" => "WBD",
    "DISCK" => "WBD",
    "ATVI" => "MSFT",
    "SIRI" => "SIRI",
    "FRC"  => "FRC",
    "SIVB" => "SIVB",
    "SNPS" => "SNPS",
    "ADSK" => "ADSK",
    "CSRA" => "GD",
    "BRCM" => "AVGO"
)

const TICKER_BLACKLIST = Set{String}([
    "ENRN", "LEH", "MER", "MHP", "FNM", "FRE", "BSC", "WB", "WAMU", "WM", "ALXN",
    "TWC", "TWX", "CERN", "XLNX", "LBTYA", "LBTYK", "NWL", "NBLX",
    "XEC", "PSX", "KSU", "NSC", "HPE", "JDSU", "ERIC", "NOK", "LGND",
    "BBBY", "BBBYQ", "HTZ", "CHK", "MHK", "DTV", "EMC", "SPLS", "GGP",
    "TIF", "CBRE", "CBG", "KODK", "YHOO", "AABA",
    "JCP", "RSH", "PBI", "KFT", "ABN", "MCI", "NOKIA"
])

const _SP500_CACHE = Ref{Union{Nothing,DataFrame}}(nothing)

function _candidate_sp500_paths()
    return [
        joinpath(pwd(), "sp500_ticker_start_end.csv"),
        joinpath(pwd(), "data", "sp500_ticker_start_end.csv"),
        joinpath(@__DIR__, "sp500_ticker_start_end.csv"),
        joinpath(@__DIR__, "..", "data", "sp500_ticker_start_end.csv")
    ]
end

function _candidate_price_paths(ticker::String)
    t = uppercase(ticker)
    return [
        joinpath(pwd(), "data", "$t.csv"),
        joinpath(pwd(), "$t.csv"),
        joinpath(@__DIR__, "data", "$t.csv"),
        joinpath(@__DIR__, "..", "data", "$t.csv")
    ]
end

function _normalize_name(s::AbstractString)
    return lowercase(replace(String(s), " " => "_"))
end

function _parse_date_column(df::DataFrame)
    date_col = nothing
    for nm in names(df)
        n = _normalize_name(String(nm))
        if n in ("date", "timestamp", "time")
            date_col = nm
            break
        end
    end
    date_col === nothing && error("No date column found.")
    if eltype(df[!, date_col]) <: Date
        return Symbol(date_col)
    end
    df[!, date_col] = Date.(string.(df[!, date_col]))
    return Symbol(date_col)
end

function _coalesce_numeric_column(df::DataFrame, candidates::Vector{String})
    for cand in candidates
        for nm in names(df)
            if _normalize_name(String(nm)) == cand
                return Symbol(nm)
            end
        end
    end
    return nothing
end

function _standardize_price_frame(df::DataFrame)
    date_sym = _parse_date_column(df)

    close_sym = _coalesce_numeric_column(df, ["adj_close", "adjclose", "adjusted_close", "close"])
    close_sym === nothing && error("No close / adjusted close column found.")

    open_sym  = _coalesce_numeric_column(df, ["open"])
    high_sym  = _coalesce_numeric_column(df, ["high"])
    low_sym   = _coalesce_numeric_column(df, ["low"])
    vol_sym   = _coalesce_numeric_column(df, ["volume", "vol"])

    out = DataFrame(
        date = Date.(df[!, date_sym]),
        close = Float64.(coalesce.(df[!, close_sym], missing))
    )

    if open_sym !== nothing
        out.open = Float64.(coalesce.(df[!, open_sym], missing))
    else
        out.open = copy(out.close)
    end

    if high_sym !== nothing
        out.high = Float64.(coalesce.(df[!, high_sym], missing))
    else
        out.high = copy(out.close)
    end

    if low_sym !== nothing
        out.low = Float64.(coalesce.(df[!, low_sym], missing))
    else
        out.low = copy(out.close)
    end

    if vol_sym !== nothing
        out.volume = Float64.(coalesce.(df[!, vol_sym], missing))
    else
        out.volume = fill(0.0, nrow(out))
    end

    sort!(out, :date)
    unique!(out, :date)
    filter!(row -> isfinite(row.close), out)
    return out
end

function _get_sp500_df()
    if _SP500_CACHE[] !== nothing
        return _SP500_CACHE[]
    end

    for path in _candidate_sp500_paths()
        if isfile(path)
            df = CSV.read(path, DataFrame)
            _SP500_CACHE[] = df
            return df
        end
    end

    error("Could not find sp500_ticker_start_end.csv. Place it in project root or data/ directory.")
end

function resolve_ticker(ticker::AbstractString)
    t = uppercase(strip(String(ticker)))
    return get(TICKER_REMAP, t, t)
end

function _normalize_sp500_columns(df::DataFrame)
    colmap = Dict{Symbol, Symbol}()

    for nm in names(df)
        n = _normalize_name(String(nm))
        if n in ("ticker", "symbol")
            colmap[Symbol(nm)] = :ticker
        elseif n in ("start", "start_date", "date_start", "from")
            colmap[Symbol(nm)] = :start_date
        elseif n in ("end", "end_date", "date_end", "to")
            colmap[Symbol(nm)] = :end_date
        end
    end

    rename!(df, colmap)

    if !haskey(colmap, :ticker)
        error("SP500 constituents CSV must contain a ticker/symbol column.")
    end
    if !haskey(colmap, :start_date)
        error("SP500 constituents CSV must contain a start/start_date column.")
    end

    if :end_date ∉ names(df)
        df.end_date = fill(missing, nrow(df))
    end

    df.ticker = uppercase.(strip.(string.(df.ticker)))
    df.start_date = Date.(string.(df.start_date))
    df.end_date = [ismissing(x) || isempty(string(x)) ? Date(9999,12,31) : Date(string(x)) for x in df.end_date]

    return df
end

function get_sp500_constituents(date::Date)
    df = _normalize_sp500_columns(copy(_get_sp500_df()))
    mask = (df.start_date .<= date) .& (df.end_date .>= date)
    tickers = unique(df.ticker[mask])
    tickers = filter(t -> !(t in TICKER_BLACKLIST), tickers)
    return tickers
end

function get_all_sp500_tickers()
    df = _normalize_sp500_columns(copy(_get_sp500_df()))
    tickers = unique(df.ticker)
    tickers = filter(t -> !(t in TICKER_BLACKLIST), tickers)
    return unique(resolve_ticker.(tickers))
end

function load_ticker_data(ticker::AbstractString)
    t = resolve_ticker(ticker)
    t in TICKER_BLACKLIST && return nothing

    for path in _candidate_price_paths(t)
        if isfile(path)
            raw = CSV.read(path, DataFrame)
            return _standardize_price_frame(raw)
        end
    end

    return nothing
end

function load_all_data()
    all_data = Dict{String, DataFrame}()
    for ticker in get_all_sp500_tickers()
        try
            df = load_ticker_data(ticker)
            if df !== nothing && nrow(df) > 0
                all_data[ticker] = df
            end
        catch
            continue
        end
    end
    return all_data
end

end
