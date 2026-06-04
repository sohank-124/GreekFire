module Features

using DataFrames, Statistics, LinearAlgebra, Distributions, Dates

function compute_rsi(prices; period=14)
    n = length(prices)
    rsi = fill(NaN, n)
    if n <= period
        return rsi
    end

    diffs = diff(prices)
    gains = max.(0.0, diffs)
    losses = max.(0.0, -diffs)

    avg_gain = sum(gains[1:period]) / period
    avg_loss = sum(losses[1:period]) / period

    for i in (period + 1):n
        if i == period + 1
        else
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        end
        
        if avg_loss == 0
            rsi[i] = 100.0
        else
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        end
    end
    return rsi
end

function compute_macd(prices; fast=12, slow=26, signal=9)
    n = length(prices)
    macd_line = fill(NaN, n)
    signal_line = fill(NaN, n)
    histogram = fill(NaN, n)

    function ema(data, period)
        res = fill(NaN, length(data))
        if length(data) < period return res end
        alpha = 2.0 / (period + 1)
        res[period] = sum(data[1:period]) / period
        for i in (period + 1):length(data)
            res[i] = data[i] * alpha + res[i-1] * (1.0 - alpha)
        end
        return res
    end

    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)

    for i in 1:n
        macd_line[i] = fast_ema[i] - slow_ema[i]
    end

    valid_macd = macd_line[(!isnan).(macd_line)]
    if length(valid_macd) >= signal
        sig_vals = ema(valid_macd, signal)
        valid_indices = findall(!isnan, macd_line)
        for i in 1:length(sig_vals)
            idx = valid_indices[i]
            signal_line[idx] = sig_vals[i]
        end
    end

    for i in 1:n
        histogram[i] = macd_line[i] - signal_line[i]
    end

    return macd_line, signal_line, histogram
end

function compute_bollinger_pos(prices; period=20, std_devs=2.0)
    n = length(prices)
    pos = fill(NaN, n)
    for i in (period):n
        window = prices[(i-period+1):i]
        mu = mean(window)
        sigma = std(window)
        upper = mu + std_devs * sigma
        lower = mu - std_devs * sigma
        if upper != lower
            pos[i] = (prices[i] - lower) / (upper - lower)
        end
    end
    return pos
end

function compute_sma_ratio(prices; period=20)
    n = length(prices)
    ratio = fill(NaN, n)
    for i in (period):n
        window = prices[(i-period+1):i]
        ratio[i] = prices[i] / mean(window)
    end
    return ratio
end

function compute_atr_norm(high, low, close; period=14)
    n = length(close)
    atr_norm = fill(NaN, n)
    tr = fill(NaN, n-1)
    for i in 1:(n-1)
        tr[i] = max(high[i+1] - low[i+1], 
                    max(abs(high[i+1] - close[i]), 
                        abs(low[i+1] - close[i])))
    end

    atr = fill(NaN, n)
    if n > period
        current_atr = sum(tr[1:period]) / period
        atr[period+1] = current_atr
        for i in (period+2):n
            current_atr = (current_atr * (period - 1) + tr[i-1]) / period
            atr[i] = current_atr
        end
    end

    for i in 1:n
        atr_norm[i] = atr[i] / close[i]
    end
    return atr_norm
end

function compute_momentum(prices; period=20)
    n = length(prices)
    mom = fill(NaN, n)
    for i in (period+1):n
        mom[i] = (prices[i] / prices[i-period]) - 1.0
    end
    return mom
end

function compute_52wk_pos(prices)
    n = length(prices)
    pos = fill(NaN, n)
    window = 252
    for i in (window):n
        slice = prices[(i-window+1):i]
        low = minimum(slice)
        high = maximum(slice)
        if high != low
            pos[i] = (prices[i] - low) / (high - low)
        end
    end
    return pos
end

function black_scholes_greeks(S, IV)
    T = 30.0 / 365.0
    r = 0.05
    K = S

    try
        if IV <= 0 || S <= 0
            return NaN, NaN, NaN, NaN
        end

        d1 = (log(S/K) + (r + 0.5 * IV^2) * T) / (IV * sqrt(T))
        d2 = d1 - IV * sqrt(T)

        phi = Normal(0, 1)
        
        delta = cdf(phi, d1)
        gamma = pdf(phi, d1) / (S * IV * sqrt(T))
        vega = S * pdf(phi, d1) * sqrt(T) / 100.0
        theta = (- (S * pdf(phi, d1) * IV) / (2 * sqrt(T)) - r * K * exp(-r * T) * cdf(phi, d2)) / 365.0

        return delta, gamma, theta, vega
    catch
        return NaN, NaN, NaN, NaN
    end
end

function compute_iv_rank(ivs)
    n = length(ivs)
    rank = fill(NaN, n)
    window = 252
    for i in (window):n
        slice = ivs[(i-window+1):i]
        valid_slice = slice[(!isnan).(slice)]
        if isempty(valid_slice)
            rank[i] = NaN
            continue
        end
        low = minimum(valid_slice)
        high = maximum(valid_slice)
        if high != low
            rank[i] = (ivs[i] - low) / (high - low)
        end
    end
    return rank
end

# NEW: Correlation feature function
function compute_rolling_correlation(rets1, rets2; period=20)
    n = length(rets1)
    corr = fill(NaN, n)
    if n < period return corr end
    
    for i in period:n
        w1 = rets1[(i-period+1):i]
        w2 = rets2[(i-period+1):i]
        if std(w1) > 0 && std(w2) > 0
            corr[i] = cor(w1, w2)
        end
    end
    return corr
end

function calculate_features(df, spy_df)
    # Align by date to prevent index mismatch between stock and SPY
    common_dates = intersect(df.date, spy_df.date)
    df = df[in.(df.date, Ref(common_dates)), :]
    spy_df = spy_df[in.(spy_df.date, Ref(common_dates)), :]
    sort!(df, :date)
    sort!(spy_df, :date)
    
    # Drop rows with missing price data (delisted tickers, weekends, etc.)
    valid_rows = .!ismissing.(df.close) .& .!isnothing.(df.close) .&
                 .!ismissing.(df.high) .& .!isnothing.(df.high) .&
                 .!ismissing.(df.low) .& .!isnothing.(df.low) .&
                 .!ismissing.(df.volume) .& .!isnothing.(df.volume)
    df = df[valid_rows, :]
    
    # Also drop any missing SPY data just in case
    spy_valid = .!ismissing.(spy_df.close) .& .!isnothing.(spy_df.close)
    spy_df = spy_df[spy_valid, :]
    
    # Re-align after independent filtering — dates may diverge
    common_dates = intersect(df.date, spy_df.date)
    df = df[in.(df.date, Ref(common_dates)), :]
    spy_df = spy_df[in.(spy_df.date, Ref(common_dates)), :]
    sort!(df, :date)
    sort!(spy_df, :date)
    
    # If too few rows remain, return empty DataFrame to avoid downstream errors
    if nrow(df) < 50 || nrow(spy_df) < 50
        return DataFrame()
    end
    
    # Force Float64 conversion to prevent Union type leakage
    prices  = Float64.(coalesce.(df.close, NaN))
    highs   = Float64.(coalesce.(df.high, NaN))
    lows    = Float64.(coalesce.(df.low, NaN))
    volumes = Float64.(coalesce.(df.volume, NaN))
    
    n = length(prices)
    
    rsi = compute_rsi(prices)
    macd_line, signal_line, macd_hist = compute_macd(prices)
    bb_pos = compute_bollinger_pos(prices)
    
    # SMAs
    sma20_ratio = compute_sma_ratio(prices; period=20)
    sma50_ratio = compute_sma_ratio(prices; period=50)
    sma200_ratio = compute_sma_ratio(prices; period=200) # ADDED
    
    atr_norm = compute_atr_norm(highs, lows, prices)
    
    vol_avg20 = fill(NaN, n)
    for i in 20:n
        vol_avg20[i] = mean(volumes[(i-19):i])
    end
    vol_ratio = volumes ./ vol_avg20
    
    mom = compute_momentum(prices)
    pos52 = compute_52wk_pos(prices)
    
    log_rets = diff(log.(prices))
    ivs = fill(NaN, n)
    for i in 31:n
        window = log_rets[(i-30):(i-1)]
        ivs[i] = std(window) * sqrt(252)
    end
    
    iv_rank = compute_iv_rank(ivs)
    
    deltas = fill(NaN, n)
    gammas = fill(NaN, n)
    thetas = fill(NaN, n)
    vegas = fill(NaN, n)
    
    for i in 1:n
        d, g, t, v = black_scholes_greeks(prices[i], ivs[i])
        deltas[i] = d
        gammas[i] = g
        thetas[i] = t
        vegas[i] = v
    end
    
    # NEW: Calculating returns for SPY correlation
    stock_rets = fill(NaN, n)
    spy_rets = fill(NaN, n)
    stock_rets[2:end] = diff(prices) ./ prices[1:end-1]
    spy_rets[2:end] = diff(spy_df.close) ./ spy_df.close[1:end-1]
    
    spy_corr = compute_rolling_correlation(stock_rets, spy_rets; period=20) # ADDED
    
    # Target: forward 1-week excess return vs SPY
    target = fill(NaN, n)
    spy_prices = spy_df.close
    
    for i in 1:(n-5)
        stock_ret = (prices[i+5] / prices[i]) - 1.0
        spy_ret = (spy_prices[i+5] / spy_prices[i]) - 1.0
        target[i] = stock_ret - spy_ret
    end

    feat_df = DataFrame(
        date = df.date,
        rsi = rsi,
        macd_line = macd_line,
        macd_signal = signal_line,
        macd_hist = macd_hist,
        bb_pos = bb_pos,
        sma20_ratio = sma20_ratio,
        sma50_ratio = sma50_ratio,
        sma200_ratio = sma200_ratio, # ADDED
        atr_norm = atr_norm,
        vol_ratio = vol_ratio,
        momentum = mom,
        pos52 = pos52,
        iv = ivs,
        iv_rank = iv_rank,
        delta = deltas,
        gamma = gammas,
        theta = thetas,
        vega = vegas,
        spy_corr = spy_corr, # ADDED
        target = target
    )
    
    return feat_df
end

end