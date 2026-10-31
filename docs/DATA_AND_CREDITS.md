# Data provenance and acknowledgments

## Reported data source

The author reports using [Acelogic/WayBackMachineStockScraper](https://github.com/Acelogic/WayBackMachineStockScraper) for project data. That repository describes recovery of delisted-stock prices from archived Yahoo Finance CSVs and historical pages through the Internet Archive's Wayback Machine.

The upstream documentation describes partial and variable coverage, stitched snapshots, and adjusted prices reflecting the archived snapshot date. Recovery of prices does not independently establish complete corporate-action histories, executable delisting payoffs, or a historical S&P 500 membership universe.

For the original market cache, the exact scraper commit, retrieval dates, per-file lineage, and source of the historical membership CSV were not recorded in the supplied documentation. Attribution is an author-reported source, not certification that every original file came from it or is accurate.

The new [optional recovery workflow](DATA_RECOVERY.md) invokes a separate upstream checkout pinned at `154db4fff923f3b3c92618ad79aa30bdc5efabf2`. Future acquisitions record run times, hashes, and coverage in staging and import receipts. This does not reconstruct the original cache's missing provenance. Upstream source and its data are not bundled into the public repository; the Julia loader continues reading local CSVs.

## Distributed and local inputs

- Raw market prices and membership CSVs remain local and excluded from Git.
- The two `octave/momentum_baseline_*.csv` files are legacy derived experiment exports. They predate data remediation and are retained only for the archived Python checks, not current performance claims.
- `CHANGELOG/` contains a dated written result snapshot and figures, not a newly reproduced result.
- The example under `examples/` generates synthetic prices in memory; it contains no market observations.

Public availability of a scraper does not establish redistribution rights to its recovered data. No new data license or blanket ownership claim is added here.

## Learning resources and software

The author learned Julia and scientific computing using tutorials whose links were not retained. No specific tutorial citation or claim of writing every component from first principles is made.

The Julia dependencies are declared in `Project.toml` and resolved in `Manifest.toml`. Core tools include DataFrames.jl, CSV.jl, MLJ, DecisionTree.jl, MLJLinearModels.jl, and XGBoost.jl. Python reporting uses the standard library and optional matplotlib. These packages retain their respective licenses.

Research motivation includes [Asness, Moskowitz and Pedersen, Value and Momentum Everywhere](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere) and [Moreira and Muir, Volatility-Managed Portfolios](https://www.nber.org/papers/w22208). Neither paper validates this implementation.
