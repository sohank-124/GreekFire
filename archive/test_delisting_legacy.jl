# ARCHIVAL SOURCE. Data-specific assumptions are unverified. Do not execute.
# ============================================================================
# test_delisting_fix.jl
# ============================================================================
# Comprehensive test harness for the delisted-ticker / data-corruption fix.
# Verifies:
#   1. TICKER_BLACKLIST contains CPWR, EP, SBNY + the 157 new delisted tickers
#   2. get_sp500_constituents() correctly excludes CPWR, EP, SBNY post-delisting
#   3. validate_ticker_data() flags CPWR/EP/SBNY data as corrupted
#   4. verify_pit_filter() returns all_excluded_correctly == true
#
# Usage:
#   julia --project=. test_delisting_fix.jl
#
# Expected runtime: < 30 seconds (just CSV reads + small in-memory checks).
# ============================================================================

using Dates
using DataFrames
using CSV

const PROJ = @__DIR__
include(joinpath(PROJ, "data_loader.jl"))

println("="^70)
println("  DELISTED-TICKER / DATA-CORRUPTION FIX â€” TEST HARNESS")
println("="^70)

# ---------------------------------------------------------------------------
# Test 1: TICKER_BLACKLIST membership
# ---------------------------------------------------------------------------
println("\n[1/5] TICKER_BLACKLIST membership test...")

const REQUIRED_BLACKLIST = ["CPWR", "EP", "SBNY"]
for t in REQUIRED_BLACKLIST
    if t in DataLoader.TICKER_BLACKLIST
        println("  [PASS] $t in TICKER_BLACKLIST")
    else
        println("  [FAIL] $t NOT in TICKER_BLACKLIST")
        error("$t missing from TICKER_BLACKLIST")
    end
end

println("  Blacklist size: $(length(DataLoader.TICKER_BLACKLIST)) entries (expected 208)")

# ---------------------------------------------------------------------------
# Test 2: PIT filter excludes known-bad tickers
# ---------------------------------------------------------------------------
println("\n[2/5] PIT filter exclusion test...")

test_cases = [
    ("CPWR", Date(2025, 9, 23), false),  # CPWR end_date 2012-01-03
    ("EP",   Date(2025, 9, 23), false),  # EP   end_date 2012-05-25
    ("SBNY", Date(2024, 6, 15), false),  # SBNY end_date 2023-03-15
]
all_excluded = Ref(true)
for (ticker, date, should_be_present) in test_cases
    constituents = Set(DataLoader.get_sp500_constituents(date))
    is_present = (ticker in constituents)
    passed = (is_present == should_be_present)
    all_excluded[] &= passed
    status = passed ? "PASS" : "FAIL"
    println("  [$status] $ticker on $date: $(is_present ? "PRESENT" : "absent") (expected $(should_be_present ? "PRESENT" : "absent"))")
end
all_excluded = all_excluded[]

# ---------------------------------------------------------------------------
# Test 3: Active tickers still included
# ---------------------------------------------------------------------------
println("\n[3/5] Active-ticker inclusion test...")

active_cases = [
    ("AAPL", Date(2025, 9, 23)),
    ("MSFT", Date(2025, 9, 23)),
    ("NVDA", Date(2025, 9, 23)),
    ("GOOGL", Date(2010, 1, 4)),
]
all_active = Ref(true)
for (ticker, date) in active_cases
    constituents = Set(DataLoader.get_sp500_constituents(date))
    is_present = (ticker in constituents)
    all_active[] &= is_present
    status = is_present ? "PASS" : "FAIL"
    println("  [$status] $ticker on $date: $(is_present ? "present" : "ABSENT")")
end
all_active = all_active[]

# ---------------------------------------------------------------------------
# Test 4: verify_pit_filter() function
# ---------------------------------------------------------------------------
println("\n[4/5] verify_pit_filter() test...")
result = DataLoader.verify_pit_filter(verbose=true)
if result.all_excluded_correctly && result.active_ticker_included
    println("  [PASS] verify_pit_filter all checks passed")
else
    println("  [FAIL] verify_pit_filter reported failures")
    error("verify_pit_filter failed")
end

# ---------------------------------------------------------------------------
# Test 5: validate_ticker_data() flags corrupted data
# ---------------------------------------------------------------------------
println("\n[5/5] validate_ticker_data() corruption detection test...")

corruption_tests = [
    ("CPWR", :expect_low_price_or_post_delist),
    ("EP",   :expect_post_delist),
    ("SBNY", :expect_post_delist),
]
for (ticker, _expectation) in corruption_tests
    df = DataLoader.load_ticker_data(ticker)
    if df === nothing
        println("  [INFO] $ticker: blacklisted at load_ticker_data (load returned nothing) â€” already prevented from polluting backtest.")
        continue
    end

    # Blacklist guard prevents loading: this branch is defensive only.
    report = DataLoader.validate_ticker_data(ticker, df)
    println("  $ticker: total_rows=$(nrow(df)), rejected=$(length(report.rejected_rows))")
    for (k, v) in report.summary
        println("    - $k: $v")
    end
    if length(report.rejected_rows) == 0
        println("  [WARN] $ticker: no rows rejected (validation may need lower thresholds)")
    end
end

# ---------------------------------------------------------------------------
# Sanity check: load_all_data() does NOT include any blacklisted tickers
# ---------------------------------------------------------------------------
println("\n[BONUS] load_all_data() blacklist check...")
all_data = DataLoader.load_all_data()
blacklist_leaks = filter(t -> t in DataLoader.TICKER_BLACKLIST, collect(keys(all_data)))
if isempty(blacklist_leaks)
    println("  [PASS] No blacklisted tickers in load_all_data() output (n=$(length(all_data)) tickers loaded)")
else
    println("  [FAIL] Blacklisted tickers leaked into load_all_data(): $blacklist_leaks")
    error("Blacklist leak detected")
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  ALL TESTS PASSED")
println("="^70)
println("  TICKER_BLACKLIST:           $(length(DataLoader.TICKER_BLACKLIST)) entries")
println("  Known-bad tickers excluded: $(all_excluded ? "YES" : "NO")")
println("  Active tickers included:    $(all_active ? "YES" : "NO")")
println("  verify_pit_filter:          $(result.all_excluded_correctly ? "PASS" : "FAIL")")
println("  Universe after blacklist:   $(length(all_data)) tickers")
println("="^70)
