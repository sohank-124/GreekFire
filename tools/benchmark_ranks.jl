# Compare identical percentile-rank calculations; no models or market data.
using DataFrames, Dates, Statistics, StatsBase
function rank_values(x)
    clamp.((tiedrank(x) .- 1) ./ (length(x)-1), 0.02, 0.98)
end
fixture = DataFrame(date=repeat(collect(Date(2020,1,1):Day(1):Date(2020,10,26)); inner=100),
                    value=sin.(collect(1:30000)))
function repeated_masks(df)
    out = copy(df)
    out.rank = zeros(nrow(out))
    for date in unique(out.date)
        mask = out.date .== date
        out[mask, :rank] = rank_values(out[mask, :value])
    end
    out.rank
end
function grouped(df)
    out = copy(df)
    out.rank = zeros(nrow(out))
    for group in groupby(out, :date)
        group[!, :rank] = rank_values(group.value)
    end
    out.rank
end
@assert repeated_masks(fixture) == grouped(fixture)
function contiguous(df)
    out=copy(df)
    out.rank=zeros(nrow(out))
    lo=1
    while lo <= nrow(out)
        hi=searchsortedlast(out.date,out.date[lo])
        out.rank[lo:hi] = rank_values(view(out.value,lo:hi))
        lo=hi+1
    end
    out.rank
end
@assert repeated_masks(fixture) == contiguous(fixture)
for (name, f) in (("repeated_masks", repeated_masks), ("grouped", grouped), ("contiguous", contiguous))
    times = [@elapsed f(fixture) for _ in 1:7]
    println(name, " seconds=", times, "; median=", median(times))
end
