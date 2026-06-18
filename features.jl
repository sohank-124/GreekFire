module Features

using DataFrames, Dates, Statistics

export calculate_features

const FORECAST_HORIZON = 5
const LOOKBACK_5 = 5
const LOOKBACK_10 = 10
const LOOKBACK_20 = 20
const LOOKBACK_60 = 60

function _standardize_price_df(df::DataFrame)
    if !hasproperty(df, :date)
        error("DataFrame must contain a date column.")
    end

    close_sym = hasproperty(df, :close) ? :close : (hasproperty(df, :adj_close) ? :adj_close : nothing)
    close_sym === nothing && error("Price frame must contain close or adj_close.")

    out = DataFrame(
        date = Date.(df.date),
        close = Float64.(coalesce.(df[!, close_sym], missing))
    )

    if hasproperty(df, :open)
        out.open = Float64.(coalesce.(df.open, missing))
    else
        out.open = copy(out.close)
    end

    if hasproperty(df, :high)
        out.high = Float64.(coalesce.(df.high, missing))
    else
        out.high = copy(out.close)
    end

    if hasproperty(df, :low)
        out.low = Float64.(coalesce.(df.low, missing))
    else
        out.low = copy(out.close)
    end

    if hasproperty(df, :volume)
        out.volume = Float64.(coalesce.(df.volume, missing))
    else
        out.volume = fill(0.0, nrow(out))
    end

    sort!(out, :date)
    unique!(out, :date)
    filter!(row -> isfinite(row.close), out)
    return out
end

function _align_on_common_dates(stock_df::DataFrame, spy_df::DataFrame)
    stock = _standardize_price_df(stock_df)
    spy   = _standardize_price_df(spy_df)

    common = intersect(stock.date, spy.date)
    sort!(common)

    stock = stock[in.(stock.date, Ref(common)), :]
    spy   = spy[in.(spy.date, Ref(common)), :]

    sort!(stock, :date)
    sort!(spy, :date)

    return stock, spy
end

function _rolling_mean(x::Vector{Float64}, i::Int, window::Int)
    lo = max(1, i - window + 1)
    return mean(@view x[lo:i])
end

function _rolling_std(x::Vector{Float64}, i::Int, window::Int)
    lo = max(1, i - window + 1)
    vals = @view x[lo:i]
    return length(vals) > 1 ? std(vals) : 0.0
end

function _pct_change(curr::Float64, prev::Float64)
    if !isfinite(curr) || !isfinite(prev) || prev == 0.0
        return NaN
    end
    return curr / prev - 1.0
end

function calculate_features(stock_df::DataFrame, spy_df::DataFrame; forecast_horizon::Int=FORECAST_HORIZON)
    stock, spy = _align_on_common_dates(stock_df, spy_df)

    n = nrow(stock)
    if n < LOOKBACK_60 + forecast_horizon + 1
        return DataFrame()
    end

    close = Float64.(stock.close)
    spy_close = Float64.(spy.close)
    dates = stock.date

    has_volume = hasproperty(stock, :volume)
    volume = has_volume ? Float64.(stock.volume) : fill(0.0, n)

    rows = DataFrame(
        date = Date[],
        ret_5 = Float64[],
        ret_20 = Float64[],
        sma_20 = Float64[],
        sma_60 = Float64[],
        vol_ratio = Float64[],
        ret_vol_ratio = Float64[],
        volume_momentum = Float64[],
        rel_mom_5 = Float64[],
        mom_quality_10 = Float64[],
        trend_r2_20 = Float64[],
        momentum_curve = Float64[],
        idio_vol_20 = Float64[],
        residual_momentum_20 = Float64[],
        volume_20 = Float64[],
        dollar_volume_ratio = Float64[],
        target = Float64[]
    )

    start_i = LOOKBACK_60 + 1

    for i in start_i:(n - forecast_horizon)
        c = close[i]
        c5  = close[max(1, i - LOOKBACK_5)]
        c20 = close[max(1, i - LOOKBACK_20)]

        s5  = spy_close[max(1, i - LOOKBACK_5)]

        spy_ret_5_val = _pct_change(spy_close[i], s5)

        stock_ret_5_val = _pct_change(c, c5)
        stock_ret_20_val = _pct_change(c, c20)

        past_rets = Float64[]
        for j in max(2, i - LOOKBACK_20 + 1):i
            push!(past_rets, _pct_change(close[j], close[j-1]))
        end

        vol_10 = _rolling_std(past_rets, length(past_rets), min(10, length(past_rets)))
        vol_60_window = _rolling_std(past_rets, length(past_rets), min(60, length(past_rets)))
        vol_ratio_val = (isfinite(vol_10) && isfinite(vol_60_window) && vol_60_window > 1e-10) ? vol_10 / vol_60_window : NaN

        vol_20_stock = _rolling_std(past_rets, length(past_rets), min(20, length(past_rets)))
        ret_vol_ratio_val = (isfinite(stock_ret_20_val) && isfinite(vol_20_stock) && vol_20_stock > 1e-10) ? stock_ret_20_val / vol_20_stock : NaN

        rel_mom_5_val = stock_ret_5_val - spy_ret_5_val

        mom_quality_10_val = mean(past_rets[max(1, end-9):end] .> 0.0)

        closes_20 = @view close[max(1, i-19):i]
        n_trend = length(closes_20)
        trend_r2_20_val = if n_trend >= 5
            x = 1.0:n_trend
            y = Float64.(closes_20)
            x_bar = mean(x)
            y_bar = mean(y)
            ss_xy = sum((x[j] - x_bar) * (y[j] - y_bar) for j in 1:n_trend)
            ss_xx = sum((x[j] - x_bar)^2 for j in 1:n_trend)
            if ss_xx > 1e-15
                b = ss_xy / ss_xx
                a = y_bar - b * x_bar
                ss_res = sum((y[j] - (a + b * x[j]))^2 for j in 1:n_trend)
                ss_tot = sum((y[j] - y_bar)^2 for j in 1:n_trend)
                ss_tot > 1e-15 ? 1.0 - ss_res / ss_tot : NaN
            else
                NaN
            end
        else
            NaN
        end

        momentum_curve_val = stock_ret_5_val - stock_ret_20_val

        vol_5_mean = has_volume ? mean(volume[max(1, i - LOOKBACK_5 + 1):i]) : NaN
        vol_20_mean = has_volume ? mean(volume[max(1, i - LOOKBACK_20 + 1):i]) : NaN
        volume_momentum_val = (isfinite(vol_5_mean) && isfinite(vol_20_mean) && vol_20_mean > 0) ? vol_5_mean / vol_20_mean : NaN

        # SPY returns for beta estimation
        spy_past_rets = Float64[]
        for j in max(2, i - LOOKBACK_20 + 1):i
            push!(spy_past_rets, _pct_change(spy_close[j], spy_close[j-1]))
        end

        # Beta regression: stock_ret = α + β * spy_ret + ε
        beta_20_val = NaN
        idio_vol_20_val = NaN
        residual_momentum_20_val = NaN
        if length(past_rets) >= 10
            valid = isfinite.(past_rets) .& isfinite.(spy_past_rets)
            x = spy_past_rets[valid]
            y = past_rets[valid]
            n_valid = length(x)
            if n_valid >= 10
                x_bar = mean(x)
                y_bar = mean(y)
                cov_xy = sum((x[j] - x_bar) * (y[j] - y_bar) for j in 1:n_valid) / n_valid
                var_x = sum((x[j] - x_bar)^2 for j in 1:n_valid) / n_valid
                if var_x > 1e-15
                    beta_20_val = cov_xy / var_x
                    alpha = y_bar - beta_20_val * x_bar
                    residuals = [y[j] - (alpha + beta_20_val * x[j]) for j in 1:n_valid]
                    idio_vol_20_val = std(residuals)
                    residual_momentum_20_val = sum(residuals)
                end
            end
        end

        # Dollar volume ratio
        dollar_volume_ratio_val = NaN
        if has_volume
            dv_5 = mean(volume[max(1, i - LOOKBACK_5 + 1):i] .* close[max(1, i - LOOKBACK_5 + 1):i])
            dv_20 = mean(volume[max(1, i - LOOKBACK_20 + 1):i] .* close[max(1, i - LOOKBACK_20 + 1):i])
            dollar_volume_ratio_val = (isfinite(dv_5) && isfinite(dv_20) && dv_20 > 0) ? dv_5 / dv_20 : NaN
        end

        future_price = close[i + forecast_horizon]
        target = _pct_change(future_price, c)

        row = (
            date = dates[i],
            ret_5 = stock_ret_5_val,
            ret_20 = stock_ret_20_val,
            sma_20 = c / max(_rolling_mean(close, i, 20), 1e-10) - 1.0,
            sma_60 = c / max(_rolling_mean(close, i, 60), 1e-10) - 1.0,
            vol_ratio = vol_ratio_val,
            ret_vol_ratio = ret_vol_ratio_val,
            volume_momentum = volume_momentum_val,
            rel_mom_5 = rel_mom_5_val,
            mom_quality_10 = mom_quality_10_val,
            trend_r2_20 = trend_r2_20_val,
            momentum_curve = momentum_curve_val,
            idio_vol_20 = idio_vol_20_val,
            residual_momentum_20 = residual_momentum_20_val,
            volume_20 = vol_20_mean,
            dollar_volume_ratio = dollar_volume_ratio_val,
            target = target
        )

        if all(isfinite, collect(values(row)))
            push!(rows, row)
        end
    end

    return rows
end

end
