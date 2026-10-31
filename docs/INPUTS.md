# Historical input contract

Market inputs are excluded from Git. Supply files in `data/` from a source you are permitted to use. No complete reproducible market-data feed is bundled or certified. The optional recovery tool and [provenance notes](DATA_AND_CREDITS.md) describe what can be recovered and what cannot be inferred.

`sp500_ticker_start_end.csv` requires `ticker,start_date,end_date`. Dates use ISO YYYY-MM-DD. Blank end means active; otherwise both boundaries are inclusive. Supply separate rows for repeat memberships. Confirm boundary semantics with the source and include share classes separately. Current constituents alone are insufficient. Malformed dates and reversed intervals fail. An index removal date is not necessarily a delisting date.

Each uppercase security symbol requires its own CSV, such as `data/ATVI.csv`. The engine never reads MSFT prices for ATVI. Genuine symbol changes need independently verified continuous security histories; until then a missing predecessor file is an error. Avoid reusing a ticker's later company's history.

Required usable price schema:

```csv
date,open,close,adjclose,volume
2020-01-02,100.0,102.0,51.0,1000
```

This row is an invented schema example. `adj_close` and `adjusted_close` are aliases. Dates may be `timestamp` or `time`. Raw open is adjusted by `adjclose/close`. If adjusted close is omitted, open and close are assumed to share a coherent raw basis; results spanning splits or dividends then need special care. If raw close or open is absent, usable execution labels cannot be inferred. Optional high/low columns are not used by current predictors. Volume must be meaningful for volume features.

The benchmark calendar is the sorted SPY date vector. Missing stock sessions remain missing on that calendar; labels never extend a holding period to the next available stock quote. Invalid future outcomes do not remove signal rows. Evaluation stops when any signal candidate lacks its scheduled outcome. This conservative coverage rule can block a run even if the missing candidate would not have been selected. No delisting recovery value is invented.

Record vendor, retrieval timestamp, source URL, file hash, security identity and adjustment policy for each acquired file. Validate duplicate dates, positivity, corporate actions, halts, delisting payoffs and calendar coverage before interpreting a run. The loader rejects duplicate dates for upstream resolution. The diagnostic helper `validate_ticker_data` flags unusual returns and post-membership observations for investigation, not automatic removal. An unusually large loss may be real and must not be deleted merely for being extreme.

Run `python tools/audit_data.py` for structural coverage. Its output cannot certify completeness or economic correctness. On October 30, 2026, the local table had 1,246 intervals and 1,194 symbols; 529 symbol files were absent and ABS had an invalid header. Sample active counts were 492 (2000-01-03), 499 (2010-01-04), 505 (2020-01-02) and 503 (2025-01-02). Counts near 500 are a sanity check, not proof of correct membership.

For public reproduction use `examples/synthetic_backtest.jl`, which generates deterministic prices without downloads. It exercises real feature/model/portfolio code and does not represent market returns.
