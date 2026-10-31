# Repository preparation record

> Historical preparation record. Superseded for current methodology by the [October 30 audit](AUDIT_2026-10-30.md). Numbers and unresolved-code descriptions below refer to the earlier implementation.

Prepared on 2026-10-25. Existing worktree changes and archived experiment results were retained. No historical backtest, new model fit, parameter search, data download, or performance rerun was authorized for this preparation.

## Changes

- Added a short engineering overview, data-source attribution, authorship context, and a synthetic Julia mechanics demonstration.
- Fixed conversion of missing numeric CSV/frame cells: missing values become `NaN` before conversion to `Float64`, allowing existing finite-value checks to handle them rather than throwing a conversion error. Added regression coverage for missing close/open/high/low/volume cells.
- Added repository checks for Python syntax, local Markdown links, required publication artifacts, accidental raw-data inclusion, and selected credential patterns. This is not an exhaustive security or Git-history audit.
- Added lightweight CI for synthetic and arithmetic checks. CI does not invoke the backtest entry points. Its dependency installation may take longer than the checks themselves.
- Excluded local exploratory experiment orchestrators from new publication inputs. These scripts remain on disk; they are outside the supported quick-check path. The already tracked `EXPERIMENT_LOG.md` remains historical exploratory material and is not current performance evidence.

## Boundaries

The historical engine's future-outcome eligibility filter, blacklist, and merger mappings remain unresolved. Fixing the complete research methodology would require verified source data and new evaluation. This preparation does not claim those repairs, validate old returns, or convert the previously inspected period into an independent test.

The dated snapshot remains dated 2026-09-30; it is not a result from this preparation. The synthetic example explicitly demonstrates the missing-future-row limitation. No tutorial references were invented, and no license for third-party data was assumed.

## Verification commands

These commands do not train a model or run a backtest:

```text
python tools/check_repository.py
python research_report.py --self-test
python robustness_matrix.py --self-test
python risk_overlay_study.py --self-test
julia --project=. examples/synthetic_pipeline.jl
julia --project=. test_research_validity.jl
```

Julia dependencies must already be installed, or installed separately with `Pkg.instantiate()`. No package installation is needed for the Python checks. `test_delisting_fix.jl` depends on private historical inputs and is deliberately outside the public quick-check and CI path.

### Local verification results (2026-10-25)

- Synthetic Julia demo: 8 assertions passed.
- Julia research-validity suite: 40 assertions passed, including the new missing-cell regression checks.
- Python report arithmetic, matrix archive-completeness, and legacy overlay self-tests: passed.
- Repository inventory, Python syntax, local links, required files, selected credential patterns, and file-size checks: passed.
- `git diff --check`: passed; Git emitted line-ending normalization warnings only.

The installed Julia 1.12.4 executable and bundled Python runtime were used because the Windows app-execution aliases were inaccessible in the agent environment. The matrix self-test used an ignored workspace temporary directory after the sandbox's default temporary directory rejected writes. GitHub Actions has been configured but has not yet run remotely. These checks establish mechanics and packaging, not historical performance or complete data validity.

Git commits, pushes, and changes to repository visibility are separate from these checks. Repository checks inspect the current publication inventory, including non-ignored new files; they do not stage it. Run them again after staging or cleanup if the inventory changes.

## Commit and push

The preparation leaves the Git index and remote untouched. Review the complete diff: the worktree already contained substantial uncommitted changes before preparation, and those are part of the files you will submit. New documentation and test files will appear under `git status` until staged.

From the project root, after reviewing the files you want to publish:

```text
git status --short
git diff
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "Archive GreekFire research with reproducible checks and methodology notes"
git push -u origin HEAD
```

`git add -A` stages all non-ignored changes, including existing work; inspect the staged changes before committing. The raw market cache and local exploratory runners are ignored. Already tracked files, including the historical experiment log, remain tracked despite ignore rules. No force push is needed. Keep the repository private until you decide to share it; these commands do not change visibility.

## Subsequent optional recovery integration (2026-10-25)

After the original preparation, the author separately authorized integration and verification of the recommended optional scraper workflow. This added `tools/recover_prices.py`, optional requirements, recovery documentation, a synthetic CSV fixture, offline Python integration tests, and a Julia loader compatibility test. CI runs only offline tests; setup and fetch remain explicit commands. Upstream commit `154db4fff923f3b3c92618ad79aa30bdc5efabf2` was independently obtained from the remote and verified in the separate local checkout.

Verification completed:

- 14 offline Python tests passed, including invalid input, missing results despite a successful process exit, timeouts, provenance checks, modified downloads, existing file/receipt refusal, and exclusive-create race protection.
- 45 Julia research-validity assertions passed, including five checks on upstream CSV format, adjusted-close selection, and adjusted-open reconstruction.
- A bounded live YHOO download succeeded: 5,142 rows from 1996-04-12 to 2016-09-14, staged in run `20261025T215239Z-8160d4a7`. Reinspection verified the recorded SHA-256 hash. Julia's actual CSV loader read all 5,142 rows and produced finite adjusted opens and closes.
- Optional dependencies were installed in ignored `.local-tools/recovery-venv/` for the live check. Neither that environment nor the upstream checkout or recovered market observations is included in the publication inventory.
- Existing `data/` inputs were not changed. Import was exercised against synthetic files in temporary test directories, not the live market cache. No backtest or model training was run.

The live result verifies this acquisition path for one ticker, not general archive availability, economic data accuracy, or a complete delisting history. Exact archive snapshot lineage is not exported by the pinned upstream tool.
