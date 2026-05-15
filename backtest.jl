module Backtest

using DataFrames, Dates, Statistics, ProgressMeter
using Base.Threads
include("model.jl")
include("features.jl")
include("data_loader.jl")

const TRAIN_WINDOW_DAYS = 252
const STEP_DAYS = 5
const TOP_N_STOCKS = 5
const BOTTOM_N_STOCKS = 5      # Number of stocks to short
const STOP_LOSS_THRESHOLD = 0.07 # 7% stop loss on individual short positions

function run_walk_forward(all_stock_data, spy_data)
    @info "Pre-computing feature matrices..."
    tickers = collect(keys(all_stock_data))
    all_features_vec = Vector{Pair{String, DataFrame}}(undef, length(tickers))

    @threads for i in 1:length(tickers)
        ticker = tickers[i]
        all_features_vec[i] = ticker => Features.calculate_features(all_stock_data[ticker], spy_data)
    end

    @info "Building master DataFrame..."
    all_dfs = DataFrame[]
    for (ticker, df) in all_features_vec
        if nrow(df) > 0
            df_copy = copy(df)
            df_copy.ticker .= ticker
            push!(all_dfs, df_copy)
        end
    end
    master_df = vcat(all_dfs...)

    trading_dates = sort(unique(master_df.date))

    start_date = DataLoader.START_DATE
    end_date = today()

    steps = []
    curr_idx = searchsortedfirst(trading_dates, start_date + Day(TRAIN_WINDOW_DAYS))
    while curr_idx + STEP_DAYS <= length(trading_dates)
        train_end   = trading_dates[curr_idx]
        test_start  = trading_dates[curr_idx + 1]
        test_end    = trading_dates[min(curr_idx + STEP_DAYS, length(trading_dates))]
        train_start = trading_dates[max(1, curr_idx - TRAIN_WINDOW_DAYS)]
        push!(steps, (
            train_start = train_start,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end
        ))
        curr_idx += STEP_DAYS
    end

    total_steps = length(steps)
    results = Vector{Any}(undef, total_steps)
    feature_importances_history = Vector{Any}(undef, total_steps)

    p = Progress(total_steps, desc="Walk-Forward Loop")

    feature_cols = String[]
    for col in names(master_df)
        if eltype(master_df[!, col]) <: Number && col ∉ ["date", "target", "ticker"]
            push!(feature_cols, String(col))
        end
    end

    # Load raw price data for sector and stop loss calculations
    # Build a ticker -> sector map (GICS sectors hardcoded for our universe)
    sector_map = Dict(
        "AAPL" => "Tech", "MSFT" => "Tech", "GOOGL" => "Tech",
        "AMZN" => "Tech", "NVDA" => "Tech", "META" => "Tech",
        "AMD"  => "Tech", "INTC" => "Tech", "ADBE" => "Tech",
        "CSCO" => "Tech", "TXN" => "Tech", "QCOM" => "Tech",
        "AVGO" => "Tech", "ACN" => "Tech", "IBM" => "Tech",
        "JPM"  => "Finance", "BAC" => "Finance", "GS" => "Finance",
        "AXP"  => "Finance", "BLK" => "Finance", "SCHW" => "Finance",
        "CME"  => "Finance", "ICE" => "Finance", "USB" => "Finance",
        "PNC"  => "Finance", "TFC" => "Finance", "COF" => "Finance",
        "MET"  => "Finance", "AIG" => "Finance", "TRV" => "Finance",
        "PGR"  => "Finance",
        "XOM"  => "Energy", "CVX" => "Energy", "COP" => "Energy",
        "EOG"  => "Energy", "SLB" => "Energy",
        "JNJ"  => "Health", "PFE" => "Health", "ABBV" => "Health",
        "MRK"  => "Health", "LLY" => "Health", "UNH" => "Health",
        "AMGN" => "Health", "GILD" => "Health", "ISRG" => "Health",
        "REGN" => "Health", "VRTX" => "Health", "CVS" => "Health",
        "TSLA" => "Auto",
        "DIS"  => "Media", "NFLX" => "Media",
        "KO"   => "Consumer", "WMT" => "Consumer", "PG" => "Consumer",
        "PEP"  => "Consumer", "COST" => "Consumer", "MCD" => "Consumer",
        "SBUX" => "Consumer", "NKE" => "Consumer", "TGT" => "Consumer",
        "LOW"  => "Consumer", "ROST" => "Consumer", "TJX" => "Consumer",
        "EL"   => "Consumer", "CL" => "Consumer", "KMB" => "Consumer",
        "GIS"  => "Consumer", "MDLZ" => "Consumer",
        "HD"   => "Industrial", "CAT" => "Industrial", "GE" => "Industrial",
        "BA"   => "Industrial", "MMM" => "Industrial", "HON" => "Industrial",
        "RTX"  => "Industrial", "EMR" => "Industrial", "ETN" => "Industrial",
        "ITW"  => "Industrial", "PH" => "Industrial", "ROK" => "Industrial",
        "UPS"  => "Industrial", "FDX" => "Industrial", "CSX" => "Industrial",
        "NSC"  => "Industrial", "DE" => "Industrial",
        "NEE"  => "Utilities",
        "SPG"  => "RealEstate", "PLD" => "RealEstate", "AMT" => "RealEstate",
        "CCI"  => "RealEstate", "EQIX" => "RealEstate", "DLR" => "RealEstate",
        "TMO"  => "Science", "DHR" => "Science",
        "BRK-B" => "Finance",
        "V"    => "Finance", "MA" => "Finance",
        "T"    => "Telecom", "VZ" => "Telecom"
    )

    for i in 1:total_steps
        step = steps[i]

        combined_train = master_df[
            (master_df.date .>= step.train_start) .&
            (master_df.date .<= step.train_end - Day(8)), :]

        combined_test = master_df[
            (master_df.date .>= step.test_start) .&
            (master_df.date .<= step.test_end), :]

        if isempty(feature_cols) || nrow(combined_train) == 0 || nrow(combined_test) == 0
            results[i] = nothing
            feature_importances_history[i] = nothing
            next!(p)
            continue
        end

        try
            pred_rf, pred_lr, pred_xgb, importances, mach_rf, mach_lr, mach_xgb =
                Model.train_and_predict(combined_train, combined_test, feature_cols)

            if mach_xgb === nothing
                results[i] = nothing
                feature_importances_history[i] = nothing
                next!(p)
                continue
            end

            feature_importances_history[i] = importances

            test_predictions = DataFrame(
                ticker        = combined_test.ticker,
                pred_ensemble = 0.5 .* pred_rf .+ 0.5 .* pred_xgb,
                actual        = combined_test.target
            )

            pred_scores = combine(
                groupby(test_predictions, :ticker),
                :pred_ensemble => mean => :avg_pred
            )

            first_date = minimum(combined_test.date)
            first_day = combined_test[combined_test.date .== first_date, :]
            actual_scores = combine(
                groupby(first_day, :ticker),
                :target => mean => :avg_actual
            )

            period_scores = innerjoin(pred_scores, actual_scores, on=:ticker)
            dropmissing!(period_scores, :avg_pred)
            filter!(row -> !isnan(row.avg_pred) && !isnan(row.avg_actual), period_scores)

            if nrow(period_scores) == 0
                results[i] = nothing
                next!(p)
                continue
            end

            # --- LONG candidates (rev=false, our working signal) ---
            sort!(period_scores, :avg_pred, rev=false)
            top_n_false       = min(TOP_N_STOCKS, nrow(period_scores))
            best_stocks_false = period_scores[1:top_n_false, :]
            actual_false      = mean(skipmissing(best_stocks_false.avg_actual))
            predicted_false   = mean(best_stocks_false.avg_pred)

            # --- SHORT candidates (rev=true, highest predicted = worst actual) ---
            # Pick bottom N stocks by our signal for shorting
            # Apply sector concentration filter: max 2 stocks per sector
            sort!(period_scores, :avg_pred, rev=true)
            short_candidates = DataFrame()
            sector_counts = Dict{String, Int}()
            for row in eachrow(period_scores)
                if nrow(short_candidates) >= BOTTOM_N_STOCKS
                    break
                end
                ticker = row.ticker
                sector = get(sector_map, ticker, "Unknown")
                count = get(sector_counts, sector, 0)
                if count < 2  # Max 2 per sector
                    push!(short_candidates, row)
                    sector_counts[sector] = count + 1
                end
            end

            # Compute short portfolio return
            # Short return = -actual_return - borrow_cost
            # Stop loss: if stock went up more than STOP_LOSS_THRESHOLD,
            # cap the loss at STOP_LOSS_THRESHOLD
            short_returns = Float64[]
            short_tickers = String[]
            for row in eachrow(short_candidates)
                if isnan(row.avg_actual)
                    continue
                end
                # Raw short return: negative of stock return minus borrow cost
                raw_short_ret = -row.avg_actual - 0.001
                # Apply stop loss: cap loss at STOP_LOSS_THRESHOLD
                # If stock went up (our short went wrong), loss is -avg_actual
                # Stop loss kicks in when -avg_actual < -STOP_LOSS_THRESHOLD
                short_ret_with_stop = max(raw_short_ret, -STOP_LOSS_THRESHOLD)
                push!(short_returns, short_ret_with_stop)
                push!(short_tickers, row.ticker)
            end

            actual_short = isempty(short_returns) ? 0.0 : mean(short_returns)

            # rev=true for compatibility with regime switcher
            sort!(period_scores, :avg_pred, rev=true)
            top_n_true       = min(TOP_N_STOCKS, nrow(period_scores))
            best_stocks_true = period_scores[1:top_n_true, :]
            actual_true      = mean(skipmissing(best_stocks_true.avg_actual))
            predicted_true   = mean(best_stocks_true.avg_pred)

            if isnan(actual_false) && isnan(actual_true)
                results[i] = nothing
                next!(p)
                continue
            end

            results[i] = (
                date               = step.test_start,
                tickers_false      = best_stocks_false.ticker,
                tickers_true       = best_stocks_true.ticker,
                tickers_short      = short_tickers,
                predicted_alpha    = predicted_false,
                actual_alpha_false = actual_false,
                actual_alpha_true  = actual_true,
                actual_alpha_short = actual_short,
                chosen_alpha       = isnan(actual_false) ? 0.0 : actual_false,
                regime_signal      = :base
            )

        catch e
            println("\nError at step $i: $e")
            results[i] = nothing
        end

        next!(p)
    end

    return results, feature_importances_history
end

end