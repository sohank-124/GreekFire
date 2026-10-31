# Optional Wayback price recovery

GreekFire can invoke [Acelogic/WayBackMachineStockScraper](https://github.com/Acelogic/WayBackMachineStockScraper) as a separately checked-out optional tool. The integration pins upstream commit `154db4fff923f3b3c92618ad79aa30bdc5efabf2`; upstream code and recovered market observations are not bundled in the public project.

This acquisition workflow is never called by `main.jl`, normal tests, or CI. Recovery does not train models, run backtests, update membership, remove blacklist entries, or change ticker mappings.

## Install once

Use Python 3.12 and Git. In PowerShell, from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-recovery.txt
.\.venv\Scripts\python.exe tools/recover_prices.py setup
```

On macOS/Linux, use `.venv/bin/python` instead. Setup clones upstream into ignored `.local-tools/wayback-scraper/` and checks out the pinned commit. Fetch refuses a different revision or modified tracked upstream files. Setup does not overwrite an existing checkout or automatically install packages.

## Download and inspect

```powershell
.\.venv\Scripts\python.exe tools/recover_prices.py fetch YHOO --timeout 180
```

Fetch prints a run ID such as `20261025T120000Z-1234abcd`. Each invocation gets a new directory under ignored `recovered-data/`; it never writes into `data/`. Provide specific tickers: upstream's built-in Nasdaq-100 list is not a historical S&P 500 universe. Upstream rate limits requests; the wrapper bounds each subprocess to 180 seconds by default (configurable up to 900). Failed or timed-out runs cannot be imported.

```powershell
.\.venv\Scripts\python.exe tools/recover_prices.py inspect RUN_ID
```

Replace `RUN_ID` with the actual printed ID. Inspection checks required columns, finite positive prices, raw OHLC consistency, nonnegative integer volume, and unique ascending ISO dates. It reports coverage and SHA-256 hashes. Missing or invalid observations cause failure rather than imputation. Inspection and import need only the Python standard library.

Review date coverage, gaps, security identity, and adjusted-price basis. Structural checks do not certify economically correct history or executable delisting payoffs. Upstream can stitch observations from different archive snapshots; this integration does not reconcile their adjustment bases.

## Import one reviewed ticker

```powershell
.\.venv\Scripts\python.exe tools/recover_prices.py import RUN_ID YHOO
```

Import revalidates the CSV and requires its hash to match the completed retrieval record. It exclusively creates `data/YHOO.csv` and `data/YHOO.recovery.json`. An existing price file or receipt causes refusal; there is no force/overwrite or automatic merge option. The receipt records upstream commit, retrieval times, coverage, hash, and run ID. The upstream manifest stays in the staging directory.

This is acquisition-run provenance, not exact source snapshot URLs or per-row lineage: the pinned upstream does not export those. Historical membership provenance and corporate actions remain separate requirements.

Import does not enable a ticker in the strategy. YHOO, for example, remains blacklisted; acquisition-successor mappings also remain in effect. Safely revising those rules requires independently verified history and new evaluation.

## Offline verification

```text
python -m unittest discover -s tests -p "test_*.py" -v
julia --project=. test_research_validity.jl
```

Python tests use synthetic CSVs and simulated subprocesses for download, timeout, failed-process, validation, provenance, tampering, and no-overwrite behavior. The Julia suite reads the same synthetic upstream-format fixture and checks date parsing and adjusted-close/open calculations through the real loader. Tests make no archive requests and do not train models.

Archive availability, rate limiting, and network access are external dependencies. Successful setup or offline checks cannot establish that a particular ticker is currently recoverable. Recovery does not retroactively validate the archived strategy returns.
