# GreekFire: project overview

> Historical preparation record. Superseded for current methodology by the [October 30 audit](AUDIT_2026-10-30.md). Numbers and unresolved-code descriptions below refer to the earlier implementation.

## Question and outcome

Could an ensemble trained on stock-level price and volume features rank historical S&P 500 stocks well enough to outperform a market benchmark after assumed trading costs?

The archived experiment did not establish that result. In the documented 315-step comparison, net strategy CAGR was 7.47% versus 12.56% for aligned SPY, with a larger drawdown. These are conditional historical simulations: unavailable future outcomes can affect which stocks reach selection, and the same period was used in earlier research decisions. See the [dated results](../CHANGELOG/RESEARCH_RESULTS_2026-09-30.md) before interpreting the numbers.

## Engineering work

The implementation combines Julia data processing and modeling with Python analysis:

| Component | Purpose | Source |
| --- | --- | --- |
| Price ingestion | Normalize local CSVs and reconstruct adjusted opens | [data_loader.jl](../data_loader.jl) |
| Optional recovery | Stage and validate upstream archive downloads before explicit import | [recover_prices.py](../tools/recover_prices.py) |
| Features and labels | Calculate momentum, volatility, trend, and volume features; record actual label end dates | [features.jl](../features.jl) |
| Models | Fit seeded Random Forest, XGBoost, and a linear reference | [model.jl](../model.jl) |
| Walk-forward engine | Train on prior available labels, rank stocks, construct portfolios, and deduct assumed costs | [backtest.jl](../backtest.jl) |
| Evaluation | Export matching benchmark windows and check completeness | [validation.jl](../validation.jl) |
| Reporting | Compare gross/net returns, drawdowns, and completed sensitivity runs | [research_report.py](../research_report.py) |

The [synthetic example](../examples/synthetic_pipeline.jl) demonstrates the feature and timing mechanics without training or market data.

## What the investigation taught me

- A label is available when its exit observation occurs, not after an approximate calendar buffer. The training selector checks each row's actual `target_end_date`.
- A benchmark comparison needs matching entry and exit assumptions. Close-to-close benchmark returns do not match next-open strategy entries.
- Costs can dominate a high-turnover strategy. Gross returns alone do not answer the research question.
- Historical membership is only one part of avoiding selection bias. Missing delisting outcomes, a global blacklist, and merger mappings still compromise the universe.
- Repeatedly inspecting a test period uses up its value as independent confirmation. A rerun cannot make it untouched again.

## Scope and authorship

This is an independent project developed while learning Julia and scientific computing with tutorials. It uses established libraries and methods; it does not claim a new ML algorithm. The original tutorial links were not retained. Later repository preparation covered review, documentation, missing-value handling, and verification tooling.

The project is archived as a scientific-computing investigation. There is no live trading record, independently verified alpha, hardware implementation, or claimed external adoption. Further performance research would require independently checked security histories and missing-outcome handling, followed by new evaluation.
