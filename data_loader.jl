module DataLoader

using CSV, DataFrames, Dates, Statistics

export START_DATE, resolve_ticker, get_sp500_constituents, get_all_sp500_tickers,
       load_ticker_data, load_all_data, TICKER_REMAP, TICKER_BLACKLIST,
       validate_ticker_data, run_data_validation, verify_pit_filter,
       MIN_PRICE_THRESHOLD, MAX_RETURN_THRESHOLD, MIN_PRICE_MOVEMENT_THRESHOLD,
       TECH_TICKERS, SEMIS_TICKERS

const START_DATE = Date(2000, 1, 3)

# ---------------------------------------------------------------------------
# Data validation constants (Phase 4 - Data Validation Layer)
# ---------------------------------------------------------------------------
# MIN_PRICE_THRESHOLD: reject any observation with close/open < this value
#   (catches penny-stock data fabrications like CPWR's $0.0064 quotes).
# MAX_RETURN_THRESHOLD: flag 5-day returns whose absolute value exceeds this
#   (catches impossible price jumps; legitimate 5-day moves rarely exceed 50%).
# MIN_PRICE_MOVEMENT_THRESHOLD: zero-volume days with |return| > this trigger
#   a "stale quote" rejection (real zero-volume days are flat or near-flat).
# FUTURE_END_DATE: sentinel used by the PIT filter to mean "still active".
#   Chosen as 2999-12-31 (well outside any reasonable backtest window) so the
#   comparison `end_date >= date` is unambiguously true for active tickers.
const MIN_PRICE_THRESHOLD = 0.01
const MAX_RETURN_THRESHOLD = 0.50
const MIN_PRICE_MOVEMENT_THRESHOLD = 0.01
const FUTURE_END_DATE = Date(2999, 12, 31)

# Keep security identity intact. Even a genuine rename requires a verified
# continuous vendor security series; never substitute an acquirer's prices.
const TICKER_REMAP = Dict{String,String}()

# Legacy forensic exclusions retained for inspection only. This set is NOT
# applied to membership or loading: future events must not erase earlier members.
const TICKER_BLACKLIST = Set{String}([
    # --- Group A: pre-existing (50 tickers) ---
    "ENRN", "LEH", "MER", "MHP", "FNM", "FRE", "BSC", "WB", "WAMU", "WM", "ALXN",
    "TWC", "TWX", "CERN", "XLNX", "LBTYA", "LBTYK", "NWL", "NBLX",
    "XEC", "PSX", "KSU", "NSC", "HPE", "JDSU", "ERIC", "NOK", "LGND",
    "BBBY", "BBBYQ", "HTZ", "CHK", "MHK", "DTV", "EMC", "SPLS", "GGP",
    "TIF", "CBRE", "CBG", "KODK", "YHOO", "AABA",
    "JCP", "RSH", "PBI", "KFT", "ABN", "MCI", "NOKIA",
    # --- Group B: forensic-investigation triggers (3 tickers) ---
    # CPWR (Compuware, delisted 2012-01-03): post-delisting data 2009-2026
    #   with prices as low as $0.0001 and 5,100% max return. Primary suspect
    #   for the +736% A1 return.
    # EP (El Paso Corp, delisted 2012-05-25): 3,512 rows of post-delisting
    #   data, max return 263.64%, zero-volume periods.
    # SBNY (SVB Financial, bankrupt 2023-03-15): 439 rows of post-delisting
    #   data, max return 95.52% - the backtest was buying a dead bank.
    "CPWR", "EP", "SBNY",
    # --- Group C: full delisting audit (157 tickers) ---
    # 166 audited tickers - 8 already in Group A (MER, WB, NWL, MHK, BBBY,
    # EMC, SPLS, PBI) - 3 in Group B (CPWR, EP, SBNY) = 157 below.
    "AAL", "AAP", "ABI", "ADT", "AET", "AIV", "ALK", "AMG", "AMTM", "AN",
    "ANDV", "ANF", "APC", "ATI", "AYI", "BBT", "BBWI", "BEAM", "BHF", "BIO",
    "BMC", "BMS", "BWA", "CA", "CAM", "CBE", "CCE", "CCEP", "CE", "CIEN",
    "CLF", "CNX", "COL", "COTY", "CPRI", "CSRA", "CVG", "CZR", "DDR", "DXC",
    "DYN", "EMN", "ENPH", "EQ", "ESRX", "ETSY", "EVHC", "FCPT", "FHN", "FLR",
    "FLS", "FMC", "FOSL", "FTI", "GENZ", "GHC", "GME", "GNW", "GR", "GT",
    "HAR", "HOG", "HOT", "HP", "HRB", "ILMN", "INFO", "IPGP", "ITT", "JAVA",
    "JEF", "KBH", "KG", "KMX", "KSS", "LB", "LEG", "LIFE", "LKQ", "LNC",
    "LUMN", "M", "MAC", "MAT", "MBI", "MEE", "MHS", "MI", "MKTX", "MMI",
    "MOLX", "MTW", "MUR", "NAVI", "NBR", "NCC", "NE", "NFX", "NKTR", "NOV",
    "NSM", "NYT", "OGN", "OI", "PBG", "PCL", "PENN", "POM", "PRGO", "PTV",
    "PVH", "QRVO", "R", "RHI", "RIG", "RRC", "RX", "S", "SCG", "SE",
    "SEDG", "SEE", "SGP", "SHLD", "SIG", "SII", "SLG", "SLM", "SOLS", "STI",
    "SUN", "SVU", "TDC", "TE", "TEG", "TFX", "TGNA", "THC", "TIE", "TLAB",
    "TRIP", "UA", "UAA", "UNM", "URBN", "UST", "VFC", "VIAV", "VNO", "VNT",
    "WHR", "WU", "XRAY", "XRX", "ZION"
])

const _SP500_CACHE = Ref{Union{Nothing,DataFrame}}(nothing)

# ---------------------------------------------------------------------------
# Universe subsets (Phase 1 - CLI experiment support)
# ---------------------------------------------------------------------------
# `universe="all"`    -> no filtering (full S&P 500 PIT membership).
# `universe="tech"`   -> restrict to the large-cap tech / growth tickers below.
# `universe="semis"`  -> restrict to semiconductor tickers (subset of tech).
# Filter is applied AFTER CSVs are loaded, so any ticker that does not have a
# data/CSV file is silently dropped by `load_all_data` first.
const TECH_TICKERS = Set{String}([
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "CSCO",
    "ORCL", "ADP", "TXN", "IBM", "HPQ", "FIS", "FISV", "PYPL",
])

const SEMIS_TICKERS = Set{String}([
    "AMD", "NVDA", "INTC", "MU", "TXN", "QCOM", "AVGO", "MRVL", "NXPI",
    "SWKS", "MCHP", "ON", "ADI", "LRCX", "AMAT", "KLAC", "ASML", "TSM",
    "STX", "WDC",
])

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

    # Raw close (NOT adj_close) -- used to derive the split/dividend adjustment factor
    # so that the open price can be brought onto the same adjusted basis as adj_close.
    raw_close_sym = _coalesce_numeric_column(df, ["close"])

    open_sym  = _coalesce_numeric_column(df, ["open"])
    high_sym  = _coalesce_numeric_column(df, ["high"])
    low_sym   = _coalesce_numeric_column(df, ["low"])
    vol_sym   = _coalesce_numeric_column(df, ["volume", "vol"])

    out = DataFrame(
        date = Date.(df[!, date_sym]),
        close = Float64.(coalesce.(df[!, close_sym], NaN))
    )

    # Adjusted open = raw_open * (adj_close / raw_close)
    # This keeps the open price on the same split/dividend-adjusted basis as adj_close,
    # which is required for the next-trading-day execution target (open[t+1] -> close[t+5]).
    if open_sym !== nothing
        raw_open = Float64.(coalesce.(df[!, open_sym], NaN))
        if raw_close_sym === nothing
            out.open = fill(NaN, nrow(out))
        elseif raw_close_sym != close_sym
            # adj_close is present and distinct from raw close -> apply adjustment factor.
            raw_close_vals = Float64.(coalesce.(df[!, raw_close_sym], NaN))
            adj_close_vals = out.close
            n = nrow(df)
            adj_factor = Vector{Float64}(undef, n)
            @inbounds for j in 1:n
                rc = raw_close_vals[j]
                ac = adj_close_vals[j]
                if !ismissing(rc) && isfinite(rc) && rc > 0.0 &&
                   !ismissing(ac) && isfinite(ac)
                    adj_factor[j] = ac / rc
                else
                    adj_factor[j] = NaN
                end
            end
            adj_open = Vector{Union{Missing, Float64}}(undef, n)
            @inbounds for j in 1:n
                ro = raw_open[j]
                adj_open[j] = ismissing(ro) ? missing : ro * adj_factor[j]
            end
            out.open = adj_open
        else
            # No adj_close (or raw close == adj close) -> already on the same basis.
            out.open = raw_open
        end
    else
        out.open = fill(NaN, nrow(out))
    end

    if high_sym !== nothing
        out.high = Float64.(coalesce.(df[!, high_sym], NaN))
    else
        out.high = copy(out.close)
    end

    if low_sym !== nothing
        out.low = Float64.(coalesce.(df[!, low_sym], NaN))
    else
        out.low = copy(out.close)
    end

    if vol_sym !== nothing
        out.volume = Float64.(coalesce.(df[!, vol_sym], NaN))
    else
        out.volume = fill(0.0, nrow(out))
    end

    sort!(out, :date)
    allunique(out.date) || error("Duplicate price dates require upstream resolution")
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
    # names(df) contains Strings; use hasproperty for Symbol column checks.
    colmap = Dict{Symbol, Symbol}()

    for nm in names(df)
        n = _normalize_name(String(nm))
        target = nothing
        if n in ("ticker", "symbol")
            target = :ticker
        elseif n in ("start", "start_date", "date_start", "from")
            target = :start_date
        elseif n in ("end", "end_date", "date_end", "to")
            target = :end_date
        end
        if target !== nothing && Symbol(nm) != target
            colmap[Symbol(nm)] = target
        end
    end

    if !isempty(colmap)
        rename!(df, colmap)
    end

    if !hasproperty(df, :ticker)
        error("SP500 constituents CSV must contain a ticker/symbol column.")
    end
    if !hasproperty(df, :start_date)
        error("SP500 constituents CSV must contain a start/start_date column.")
    end

    if !hasproperty(df, :end_date)
        df.end_date = fill(missing, nrow(df))
    end

    df.ticker = uppercase.(strip.(string.(df.ticker)))
    df.start_date = Date.(string.(df.start_date))
    # Preserve valid removal dates. Only blank cells mean still active.
    df.end_date = [_to_end_date(x) for x in df.end_date]

    all(df.start_date .<= df.end_date) || error("Membership starts after it ends")
    return df
end

"""
    _to_end_date(x) -> Date

Internal helper that coerces a single `end_date` cell into a `Date`,
treating `missing`, `nothing`, empty strings, and whitespace-only strings
as "still active" (returns `FUTURE_END_DATE` = 2999-12-31).
"""
function _to_end_date(x)
    if x === missing || x === nothing
        return FUTURE_END_DATE
    end
    s = strip(string(x))
    if isempty(s)
        return FUTURE_END_DATE
    end
    # Already a Date (defensive; if eltype is already Date this is a no-op).
    if x isa Date
        return x
    end
    return Date(s) # malformed dates must fail rather than extend membership

end

function get_sp500_constituents(date::Date)
    df = _normalize_sp500_columns(copy(_get_sp500_df()))
    # Phase 3 PIT filter: must be Vector{Date} on both sides of the comparison.
    # _normalize_sp500_columns guarantees this via _to_end_date above.
    # Type-assert defensively in case a future caller bypasses normalization.
    @assert eltype(df.start_date) <: Date "start_date column is not Vector{Date}"
    @assert eltype(df.end_date)   <: Date "end_date column is not Vector{Date}"

    mask = (df.start_date .<= date) .& (df.end_date .>= date)
    tickers = unique(df.ticker[mask])
    return tickers
end

function get_all_sp500_tickers()
    df = _normalize_sp500_columns(copy(_get_sp500_df()))
    tickers = unique(df.ticker)
    return unique(resolve_ticker.(tickers))
end

function load_ticker_data(ticker::AbstractString)
    t = resolve_ticker(ticker)

    for path in _candidate_price_paths(t)
        if isfile(path)
            raw = CSV.read(path, DataFrame)
            return _standardize_price_frame(raw)
        end
    end

    return nothing
end

function load_all_data(; universe::String="all",
                          exclude_tickers::Set{String}=Set{String}(),
                          require_complete::Bool=true)
    all_data = Dict{String, DataFrame}()
    for ticker in get_all_sp500_tickers()
        try
            df = load_ticker_data(ticker)
            if df !== nothing && nrow(df) > 0
                all_data[ticker] = df
            end
        catch e
            error("Failed loading $ticker: $(sprint(showerror, e))")
        end
    end

    missing_tickers = setdiff(get_all_sp500_tickers(), collect(keys(all_data)))
    require_complete && !isempty(missing_tickers) && error("Missing $(length(missing_tickers)) historical security files; refusing an incomplete universe. Run tools/audit_data.py for coverage.")
    isempty(missing_tickers) || @warn "Historical security files unavailable; universe is incomplete" count=length(missing_tickers)

    # Apply universe filter (after CSV loading, before feature computation).
    if universe == "tech"
        filter!(p -> p[1] in TECH_TICKERS, all_data)
    elseif universe == "semis"
        filter!(p -> p[1] in SEMIS_TICKERS, all_data)
    elseif universe == "all"
        # no filtering
    else
        @warn "Unknown universe=$(repr(universe)); expected 'all', 'tech', or 'semis'. Returning full universe."
    end

    # Apply per-ticker exclusion list (applies on top of universe filter).
    if !isempty(exclude_tickers)
        filter!(p -> !(p[1] in exclude_tickers), all_data)
    end

    return all_data
end

# ---------------------------------------------------------------------------
# Phase 3 - verify_pit_filter()
# ---------------------------------------------------------------------------
"""
    verify_pit_filter(; verbose::Bool=true) -> NamedTuple

Sanity check that the PIT membership filter actually excludes tickers past
their membership `end_date`. Verifies three known-bad tickers (CPWR, EP, SBNY)
are correctly excluded, and that a known-active ticker (AAPL) is included.

Returns a NamedTuple with:
  - all_excluded_correctly:  Bool
  - active_ticker_included:  Bool
  - test_results:            Vector{Tuple{String,Date,Bool}}
"""
function verify_pit_filter(; verbose::Bool=true)
    # Pick dates AFTER each ticker's official end_date (from the audit data).
    # All three must be excluded from the constituent set.
    test_cases = [
        ("CPWR", Date(2025,  9, 23), false),  # CPWR end_date 2012-01-03
        ("EP",   Date(2025,  9, 23), false),  # EP   end_date 2012-05-25
        ("SBNY", Date(2024,  6, 15), false),  # SBNY end_date 2023-03-15
        ("AAPL", Date(2025,  9, 23), true),   # AAPL still active (empty end_date)
    ]

    results = Tuple{String, Date, Bool}[]
    all_correct = true
    for (ticker, date, should_be_present) in test_cases
        constituents = Set(get_sp500_constituents(date))
        is_present = (ticker in constituents)
        passed = (is_present == should_be_present)
        all_correct &= passed
        push!(results, (ticker, date, passed))
        if verbose
            status = passed ? "PASS" : "FAIL"
            presence = is_present ? "present" : "absent"
            expected = should_be_present ? "present" : "absent"
            println("  [$status] $ticker on $date: $presence (expected $expected)")
        end
    end

    if verbose
        println(all_correct ? "\n  verify_pit_filter: ALL CHECKS PASSED" :
                              "\n  verify_pit_filter: ONE OR MORE CHECKS FAILED")
    end

    return (
        all_excluded_correctly  = all_correct,
        active_ticker_included  = any(r -> r[1] == "AAPL" && r[3], results),
        test_results            = results,
    )
end

# ---------------------------------------------------------------------------
# Phase 4 - Data Validation Layer
# ---------------------------------------------------------------------------
"""
    validate_ticker_data(ticker::String, df::DataFrame;
                         min_price::Float64=MIN_PRICE_THRESHOLD,
                         max_return::Float64=MAX_RETURN_THRESHOLD,
                         min_movement::Float64=MIN_PRICE_MOVEMENT_THRESHOLD,
                         post_delisting_only::Bool=true) -> NamedTuple

Validate a single ticker's price DataFrame against five corruption checks:
  1. **Low-price observations**: close or open below `min_price` (catches
     penny-stock fabrications like CPWR's \$0.0064 quotes).
  2. **Excessive 5-day returns**: |close[i] / close[i-5] - 1| > `max_return`
     (catches impossible price jumps; legitimate moves rarely exceed 50%).
  3. **Zero-volume with movement**: volume == 0 AND |return| > `min_movement`
     (catches stale-quote fabrications on halted tickers).
  4. **Post-membership observations**: any row with date > the ticker's
     membership `end_date` in `sp500_ticker_start_end.csv`.
  5. **Non-finite prices**: NaN/Inf in close/open (last-line defense).

Returns a NamedTuple with:
  - rejected_rows:  Vector{Int}  (1-based row indices)
  - reasons:        Vector{String}
  - summary:        Dict{String,Int} (issue_type -> count)

The function is **non-mutating**: it only inspects `df`. To actually drop
bad rows, call `df[setdiff(1:nrow(df), report.rejected_rows), :]` on the
result.
"""
function validate_ticker_data(ticker::String, df::DataFrame;
                              min_price::Float64=MIN_PRICE_THRESHOLD,
                              max_return::Float64=MAX_RETURN_THRESHOLD,
                              min_movement::Float64=MIN_PRICE_MOVEMENT_THRESHOLD,
                              post_delisting_only::Bool=true)
    rejected = Set{Int}()
    reasons  = String[]
    summary  = Dict{String, Int}()

    function _add!(idx::Int, reason::String, key::String)
        if !(idx in rejected)
            push!(rejected, idx)
            push!(reasons, reason)
        end
        summary[key] = get(summary, key, 0) + 1
    end

    n = nrow(df)
    n == 0 && return (rejected_rows=Int[], reasons=String[], summary=summary)

    has_close = hasproperty(df, :close)
    has_open  = hasproperty(df, :open)
    has_vol   = hasproperty(df, :volume)

    # Resolve the ticker's official end_date from the SP500 membership CSV.
    official_end = _official_end_date(ticker)

    @inbounds for i in 1:n
        # --- Check 5: Non-finite prices ---
        if has_close && (!isfinite(df.close[i]))
            _add!(i, "Non-finite close at row $i (date=$(df.date[i]))", "non_finite")
        end
        if has_open && (!isfinite(df.open[i]))
            _add!(i, "Non-finite open at row $i (date=$(df.date[i]))", "non_finite")
        end

        # --- Check 1: Low-price observations ---
        if has_close && isfinite(df.close[i]) && df.close[i] < min_price && df.close[i] > 0.0
            _add!(i, "Close \$$(round(df.close[i], digits=4)) < min_price \$$(min_price) at row $i (date=$(df.date[i]))", "low_price")
        end
        if has_open && isfinite(df.open[i]) && df.open[i] < min_price && df.open[i] > 0.0
            _add!(i, "Open \$$(round(df.open[i], digits=4)) < min_price \$$(min_price) at row $i (date=$(df.date[i]))", "low_price")
        end

        # --- Check 4: Post-delisting observation ---
        if official_end !== FUTURE_END_DATE && df.date[i] > official_end
            _add!(i, "Post-membership row $i (date=$(df.date[i])) > end_date $(official_end)", "post_membership")
        end
    end

    # --- Check 2: Excessive 5-day returns ---
    if has_close && n >= 6
        @inbounds for i in 6:n
            prev = df.close[i - 5]
            curr = df.close[i]
            if isfinite(prev) && isfinite(curr) && prev > 0.0
                ret = curr / prev - 1.0
                    if abs(ret) > max_return
                        _add!(i, "5-day return $(round(ret * 100, digits=2))% at row $i (date=$(df.date[i])) exceeds +/-$(round(max_return * 100, digits=1))%", "excessive_return")
                    end
            end
        end
    end

    # --- Check 3: Zero-volume with non-trivial movement ---
    if has_close && has_vol && n >= 2
        @inbounds for i in 2:n
            v = df.volume[i]
            if isfinite(v) && v == 0.0
                prev_c = df.close[i - 1]
                curr_c = df.close[i]
                if isfinite(prev_c) && isfinite(curr_c) && prev_c > 0.0
                    r = abs(curr_c / prev_c - 1.0)
                    if r > min_movement
                        _add!(i, "Zero volume with $(round(r * 100, digits=2))% movement at row $i (date=$(df.date[i]))", "zero_volume_movement")
                    end
                end
            end
        end
    end

    return (
        rejected_rows = sort(collect(rejected)),
        reasons       = reasons,
        summary       = summary,
    )
end

"""
    _official_end_date(ticker::String) -> Date

Look up the official SP500 end_date for `ticker` in the membership CSV.
Returns `FUTURE_END_DATE` (2999-12-31) if the ticker is still active or was
never in the index. The blacklist is intentionally NOT consulted here:
even if a ticker is blacklisted, the validation layer should still report
*why* the data is corrupt, so users can audit the blacklist itself.
"""
function _official_end_date(ticker::String)
    t = uppercase(strip(ticker))
    try
        df = _normalize_sp500_columns(copy(_get_sp500_df()))
        idxs = findall(==(t), df.ticker)
        isempty(idxs) && return FUTURE_END_DATE
        # Use the most recent end_date for tickers that appear multiple times
        # (e.g., AAL has 1996-2017 and 2015-2024 entries).
        return maximum(df.end_date[idxs])
    catch
        return FUTURE_END_DATE
    end
end

"""
    run_data_validation(all_data::Dict{String, DataFrame};
                        output_csv::Union{String,Nothing}=nothing) -> DataFrame

Run `validate_ticker_data` on every ticker in `all_data` and aggregate
findings into a summary DataFrame with columns:

    ticker, total_rows, rejected_rows, n_low_price, n_excessive_return,
    n_zero_volume_movement, n_post_membership, n_non_finite, worst_return

If `output_csv` is given, the summary is also written to that path.

Returns the summary DataFrame sorted by rejected_rows descending.
"""
function run_data_validation(all_data::Dict{String, DataFrame};
                              output_csv::Union{String,Nothing}=nothing)
    summary_rows = NamedTuple[]
    tickers = sort(collect(keys(all_data)))

    for ticker in tickers
        df = all_data[ticker]
        report = validate_ticker_data(ticker, df)
        s = report.summary
        push!(summary_rows, (
            ticker              = ticker,
            total_rows          = nrow(df),
            rejected_rows       = length(report.rejected_rows),
            n_low_price         = get(s, "low_price",         0),
            n_excessive_return  = get(s, "excessive_return",  0),
            n_zero_volume_movement = get(s, "zero_volume_movement", 0),
            n_post_membership    = get(s, "post_membership",    0),
            n_non_finite        = get(s, "non_finite",        0),
        ))
    end

    summary_df = DataFrame(summary_rows)
    sort!(summary_df, :rejected_rows, rev=true)

    if output_csv !== nothing
        # Make sure parent dir exists.
        outdir = dirname(output_csv)
        if !isempty(outdir) && !isdir(outdir)
            mkpath(outdir)
        end
        CSV.write(output_csv, summary_df)
    end

    return summary_df
end

end
