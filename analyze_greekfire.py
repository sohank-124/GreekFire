#!/usr/bin/env python3
"""
analyze_greekfire.py
====================

Standalone post-hoc analysis of the GreekFire backtest outputs.

Inputs
------
octave/holdings.csv     - per-step, per-ticker portfolio holdings
                          (columns: step_index, date, ticker, rank,
                           predicted_score, sector, weight, side,
                           actual_return, contribution)
octave/model_health.csv - per-step model-health metrics
                          (columns: step_index, date, ic, rank_ic,
                           portfolio_return, spy_return,
                           pred_spread, decile_spread)

Outputs (created under octave/analysis/)
----------------------------------------
yearly_metrics.csv           - per-year IC / Rank IC / portfolio / SPY
stock_frequency.csv          - per-ticker count, avg weight
sector_exposure.csv          - average sector weight (all dates + drawdown)
top_contributors.csv         - top 20 stocks by sum(contribution)
worst_contributors.csv       - bottom 20 stocks by sum(contribution)
drawdown_attribution.csv     - worst 20 stocks over Sep 2023 - Jun 2025
ic_timeseries.png            - IC with 20-step rolling-mean overlay
rank_ic_timeseries.png       - Rank IC with 20-step rolling-mean overlay
rolling_ic.png               - 20-step rolling IC, alone
rolling_rank_ic.png          - 20-step rolling Rank IC, alone
sector_exposure.png          - stacked area chart of sector weights
analysis_summary.md          - human-readable findings + recommendations

Usage
-----
    python analyze_greekfire.py

Notes
-----
* All aggregations use Polars.  Charts use matplotlib with Polars -> numpy
  conversion at the plot boundary.
* The script is robust to missing columns: every optional column is gated
  with an `in df.columns` check.
* Dates in the source CSVs are ISO strings (YYYY-MM-DD), so lexicographic
  string comparison matches chronological order and no parsing is needed
  for filtering.  A parallel `date_dt` column is added for clean x-axes.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOLDINGS_PATH = Path("octave/holdings.csv")
HEALTH_PATH = Path("octave/model_health.csv")
OUTPUT_DIR = Path("octave/analysis")

# Drawdown investigation window per spec.
DRAWDOWN_START = "2023-09-01"
DRAWDOWN_END = "2025-06-30"

# Chart styling
FIG_SIZE = (12, 6)
DPI = 150
DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _banner(title: str) -> None:
    """Print a section header to stdout."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _to_dates(series: pl.Series) -> list[datetime]:
    """Convert an ISO-date string series to a list of Python datetimes."""
    return [datetime.strptime(s, DATE_FMT) for s in series.to_list()]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the two CSVs and attach a parsed datetime column for plotting."""
    if not HOLDINGS_PATH.exists():
        sys.exit(f"Missing input file: {HOLDINGS_PATH}")
    if not HEALTH_PATH.exists():
        sys.exit(f"Missing input file: {HEALTH_PATH}")

    holdings = pl.read_csv(HOLDINGS_PATH)
    health = pl.read_csv(HEALTH_PATH)

    # Drop fully-null rows that occasionally appear at the end of CSVs.
    if "step_index" in holdings.columns and "date" in holdings.columns:
        holdings = holdings.drop_nulls(subset=["step_index", "date"])
    if "step_index" in health.columns and "date" in health.columns:
        health = health.drop_nulls(subset=["step_index", "date"])

    # Add a Python-datetime view of "date" for clean matplotlib axes.
    # Original "date" column is kept (string) for lexicographic filtering.
    if "date" in health.columns and health["date"].dtype == pl.Utf8:
        health = health.with_columns(
            pl.col("date").str.strptime(pl.Datetime, DATE_FMT).alias("date_dt")
        )
    if "date" in holdings.columns and holdings["date"].dtype == pl.Utf8:
        holdings = holdings.with_columns(
            pl.col("date").str.strptime(pl.Datetime, DATE_FMT).alias("date_dt")
        )

    return holdings, health


# ---------------------------------------------------------------------------
# Analysis 1: Model health
# ---------------------------------------------------------------------------
def analyze_model_health(health: pl.DataFrame, output_dir: Path) -> dict:
    """Run all model-health analyses; return key stats for the report."""
    _banner("1. Model Health")

    # ---- 1.1 Overall statistics -------------------------------------------
    print("\n[1.1] Overall statistics")
    stats: dict = {}
    if "ic" in health.columns:
        stats["mean_ic"] = float(health["ic"].mean())
        stats["std_ic"] = float(health["ic"].std())
        print(f"  Mean IC     : {stats['mean_ic']:.4f}   (std {stats['std_ic']:.4f})")
    if "rank_ic" in health.columns:
        stats["mean_rank_ic"] = float(health["rank_ic"].mean())
        stats["std_rank_ic"] = float(health["rank_ic"].std())
        print(f"  Mean Rank IC: {stats['mean_rank_ic']:.4f}   (std {stats['std_rank_ic']:.4f})")
    if "portfolio_return" in health.columns:
        stats["mean_port_return"] = float(health["portfolio_return"].mean())
        stats["std_port_return"] = float(health["portfolio_return"].std())
        print(f"  Mean PortRet: {stats['mean_port_return']:.4f}   (std {stats['std_port_return']:.4f})")
    if "spy_return" in health.columns:
        stats["mean_spy_return"] = float(health["spy_return"].mean())
        print(f"  Mean SPYRet : {stats['mean_spy_return']:.4f}")

    # ---- 1.2 Rolling metrics (20-step trailing window) --------------------
    # `pl.col(...).rolling_mean(window_size=20)` produces a trailing rolling
    # mean of the most-recent 20 rows.  This is the canonical Polars API; the
    # method-style `df.rolling_mean(...)` shown in the spec is a hint, not a
    # real DataFrame method.
    print("\n[1.2] 20-step rolling metrics")
    rolling_exprs = []
    if "ic" in health.columns:
        rolling_exprs.append(
            pl.col("ic").rolling_mean(window_size=20).alias("rolling_ic")
        )
    if "rank_ic" in health.columns:
        rolling_exprs.append(
            pl.col("rank_ic").rolling_mean(window_size=20).alias("rolling_rank_ic")
        )
    if rolling_exprs:
        health = health.with_columns(rolling_exprs)

    # ---- 1.3 Annual breakdown --------------------------------------------
    print("\n[1.3] Annual breakdown -> yearly_metrics.csv")
    if "date" in health.columns:
        # Year is the first 4 chars of the ISO date string; cast to Int32 so
        # the group_by produces a tidy integer-typed year column.
        agg_exprs = []
        if "ic" in health.columns:
            agg_exprs.append(pl.col("ic").mean().alias("mean_ic"))
        if "rank_ic" in health.columns:
            agg_exprs.append(pl.col("rank_ic").mean().alias("mean_rank_ic"))
        if "portfolio_return" in health.columns:
            agg_exprs.append(
                pl.col("portfolio_return").mean().alias("mean_portfolio_return")
            )
        if "spy_return" in health.columns:
            agg_exprs.append(pl.col("spy_return").mean().alias("mean_spy_return"))

        yearly = (
            health.with_columns(
                pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("year")
            )
            .group_by("year")
            .agg(agg_exprs)
            .sort("year")
        )
        yearly.write_csv(output_dir / "yearly_metrics.csv")
        print(yearly)

    # ---- 1.4 Drawdown investigation --------------------------------------
    print(f"\n[1.4] Drawdown investigation ({DRAWDOWN_START} - {DRAWDOWN_END})")
    dd = health.filter(
        (pl.col("date") >= DRAWDOWN_START) & (pl.col("date") <= DRAWDOWN_END)
    )
    n_dd = dd.height
    n_total = health.height
    print(f"  Steps in window: {n_dd} / {n_total} total")

    dd_stats: dict = {"window_steps": n_dd, "total_steps": n_total}
    for col in ("ic", "rank_ic"):
        if col not in dd.columns:
            continue
        s = dd[col].drop_nulls()
        neg = int((s < 0).sum())
        pos = int((s > 0).sum())
        neg_pct = neg / s.len() if s.len() else float("nan")
        mean_window = float(dd[col].mean()) if s.len() else float("nan")
        mean_overall = float(health[col].mean()) if col in health.columns else float("nan")
        dd_stats[f"{col}_neg_count"] = neg
        dd_stats[f"{col}_pos_count"] = pos
        dd_stats[f"{col}_neg_pct"] = neg_pct
        dd_stats[f"{col}_mean_window"] = mean_window
        dd_stats[f"{col}_mean_overall"] = mean_overall
        print(
            f"  {col:<8}: window mean {mean_window:+.4f}  "
            f"(overall {mean_overall:+.4f})  "
            f"neg-steps {neg}/{s.len()} ({neg_pct:.1%})"
        )

    dd_stats["ic_deteriorated"] = bool(
        "ic" in dd.columns
        and "ic" in health.columns
        and dd_stats.get("ic_mean_window", 0.0) < dd_stats.get("ic_mean_overall", 0.0)
    )
    dd_stats["rank_ic_deteriorated"] = bool(
        "rank_ic" in dd.columns
        and "rank_ic" in health.columns
        and dd_stats.get("rank_ic_mean_window", 0.0)
        < dd_stats.get("rank_ic_mean_overall", 0.0)
    )
    print(
        f"  IC deteriorated? {dd_stats['ic_deteriorated']}; "
        f"Rank IC deteriorated? {dd_stats['rank_ic_deteriorated']}"
    )

    # ---- 1.5 Charts ------------------------------------------------------
    print("\n[1.5] Writing charts")
    _plot_ic_timeseries(health, output_dir)
    _plot_rolling(health, output_dir)

    stats["dd"] = dd_stats
    return stats


def _plot_ic_timeseries(health: pl.DataFrame, output_dir: Path) -> None:
    """IC and Rank IC time series with 20-step rolling-mean overlays."""
    if "date_dt" not in health.columns:
        return

    x = health["date_dt"].to_list()

    if "ic" in health.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(x, health["ic"].to_list(), alpha=0.35, label="IC", color="C0")
        if "rolling_ic" in health.columns:
            ax.plot(
                x, health["rolling_ic"].to_list(), linewidth=1.6,
                label="20-step rolling IC", color="C0",
            )
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("Information Coefficient (IC) over time")
        ax.set_xlabel("Date")
        ax.set_ylabel("IC")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(output_dir / "ic_timeseries.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    if "rank_ic" in health.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(x, health["rank_ic"].to_list(), alpha=0.35, label="Rank IC", color="C1")
        if "rolling_rank_ic" in health.columns:
            ax.plot(
                x, health["rolling_rank_ic"].to_list(), linewidth=1.6,
                label="20-step rolling Rank IC", color="C1",
            )
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("Rank Information Coefficient over time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rank IC")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(output_dir / "rank_ic_timeseries.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def _plot_rolling(health: pl.DataFrame, output_dir: Path) -> None:
    """Standalone 20-step rolling IC / Rank IC charts."""
    if "date_dt" not in health.columns:
        return
    x = health["date_dt"].to_list()

    if "rolling_ic" in health.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(x, health["rolling_ic"].to_list(), color="C0")
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("20-step Rolling IC")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling IC")
        ax.grid(True)
        fig.autofmt_xdate()
        fig.savefig(output_dir / "rolling_ic.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    if "rolling_rank_ic" in health.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(x, health["rolling_rank_ic"].to_list(), color="C1")
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("20-step Rolling Rank IC")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Rank IC")
        ax.grid(True)
        fig.autofmt_xdate()
        fig.savefig(output_dir / "rolling_rank_ic.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis 2: Holdings
# ---------------------------------------------------------------------------
def analyze_holdings(holdings: pl.DataFrame, output_dir: Path) -> dict:
    """Run all holdings analyses; return key stats for the report."""
    _banner("2. Holdings")

    # ---- 2.1 Holdings concentration --------------------------------------
    print("\n[2.1] Holdings concentration -> stock_frequency.csv")
    freq_stats: dict = {}
    if "ticker" in holdings.columns and "weight" in holdings.columns:
        per_ticker = (
            holdings.group_by("ticker")
            .agg(
                pl.len().alias("count"),
                pl.col("weight").mean().alias("avg_weight"),
            )
            .sort("count", descending=True)
        )
        per_ticker.write_csv(output_dir / "stock_frequency.csv")
        print("  Top 20 most-held tickers:")
        print(per_ticker.head(20))

        freq_stats["n_tickers"] = per_ticker.height
        freq_stats["mean_count"] = float(per_ticker["count"].mean())
        freq_stats["max_count"] = int(per_ticker["count"].max())

        # Herfindahl index per step:
        #   H_t = sum_i (w_i / sum_j w_j)^2
        # averaged across all steps.  Higher = more concentrated.
        if "step_index" in holdings.columns:
            per_step_total = holdings.group_by("step_index").agg(
                pl.col("weight").sum().alias("step_total")
            )
            joined = holdings.join(per_step_total, on="step_index", how="left")
            hh = (
                joined.with_columns(
                    (pl.col("weight") / pl.col("step_total")).alias("norm_w")
                )
                .group_by("step_index")
                .agg((pl.col("norm_w") ** 2).sum().alias("hh"))
            )
            mean_hh = float(hh["hh"].mean())
            freq_stats["herfindahl"] = mean_hh
            print(f"  Mean Herfindahl index (concentration): {mean_hh:.4f}")
            print(f"  Distinct tickers held at any time       : {freq_stats['n_tickers']}")
            print(f"  Mean holding count per ticker           : "
                  f"{freq_stats['mean_count']:.1f}")

    # ---- 2.2 Sector exposure ---------------------------------------------
    print("\n[2.2] Sector exposure -> sector_exposure.csv / sector_exposure.png")
    sector_stats: dict = {}
    if {"date", "sector", "weight"}.issubset(holdings.columns):
        # Per (date, sector) total weight (sum of all tickers in that sector
        # on that date), then average across dates for an "average sector
        # exposure" number.
        per_ds = (
            holdings.group_by(["date", "sector"])
            .agg(pl.col("weight").sum().alias("sector_weight"))
        )

        avg_sector = (
            per_ds.group_by("sector")
            .agg(pl.col("sector_weight").mean().alias("avg_weight"))
            .sort("avg_weight", descending=True)
        )
        avg_sector.write_csv(output_dir / "sector_exposure.csv")
        print("  Average sector exposure (all dates):")
        print(avg_sector)
        sector_stats["avg_sector"] = avg_sector

        dd_per_ds = per_ds.filter(
            (pl.col("date") >= DRAWDOWN_START) & (pl.col("date") <= DRAWDOWN_END)
        )
        dd_sector = (
            dd_per_ds.group_by("sector")
            .agg(pl.col("sector_weight").mean().alias("avg_weight"))
            .sort("avg_weight", descending=True)
        )
        sector_stats["dd_sector"] = dd_sector
        print(f"\n  Average sector exposure (drawdown window):")
        print(dd_sector)

        # Pivot to wide form for the stacked area chart; sort by date so
        # matplotlib's categorical x-axis is chronological.
        pivot = (
            per_ds.sort("date")
            .pivot(
                index="date",
                on="sector",
                values="sector_weight",
                aggregate_function="sum",
            )
            .fill_null(0.0)
        )
        _plot_sector_area(pivot, output_dir)

    # ---- 2.3 Attribution --------------------------------------------------
    print("\n[2.3] Attribution -> top_contributors.csv / worst_contributors.csv")
    attribution: dict = {}
    if {"ticker", "contribution"}.issubset(holdings.columns):
        per_ticker_attr = (
            holdings.group_by("ticker")
            .agg(pl.col("contribution").sum().alias("total_contribution"))
        )
        top = per_ticker_attr.sort("total_contribution", descending=True).head(20)
        worst = per_ticker_attr.sort("total_contribution").head(20)
        top.write_csv(output_dir / "top_contributors.csv")
        worst.write_csv(output_dir / "worst_contributors.csv")
        attribution["top"] = top
        attribution["worst"] = worst
        print("  Top 20 contributors:")
        print(top)
        print("\n  Worst 20 contributors:")
        print(worst)

    if {"sector", "contribution"}.issubset(holdings.columns):
        sector_attr = (
            holdings.group_by("sector")
            .agg(pl.col("contribution").sum().alias("total_contribution"))
            .sort("total_contribution", descending=True)
        )
        attribution["sector_attr"] = sector_attr
        print("\n  Contribution by sector:")
        print(sector_attr)

    # ---- 2.4 Drawdown attribution -----------------------------------------
    print(f"\n[2.4] Drawdown attribution -> drawdown_attribution.csv")
    dd_attr: dict = {"cause": "no_aggregate_loss"}

    dd_holdings = holdings.filter(
        (pl.col("date") >= DRAWDOWN_START) & (pl.col("date") <= DRAWDOWN_END)
    )
    if dd_holdings.height > 0 and {"ticker", "contribution"}.issubset(
        dd_holdings.columns
    ):
        # Worst 20 stocks over the window -- saved to drawdown_attribution.csv
        all_ticker_contrib = (
            dd_holdings.group_by("ticker")
            .agg(pl.col("contribution").sum().alias("total_contribution"))
        )
        worst_20 = all_ticker_contrib.sort("total_contribution").head(20)
        worst_20.write_csv(output_dir / "drawdown_attribution.csv")

        # Sector-level contribution over the window
        worst_dd_sector: Optional[pl.DataFrame] = None
        if "sector" in dd_holdings.columns:
            worst_dd_sector = (
                dd_holdings.group_by("sector")
                .agg(pl.col("contribution").sum().alias("total_contribution"))
                .sort("total_contribution")
            )

        # Sum of negative contributions across all stocks = total realized
        # loss in the window.  Used as the denominator for the share metrics.
        neg_total = float(
            all_ticker_contrib.filter(pl.col("total_contribution") < 0)
            .select(pl.col("total_contribution").sum())
            .item()
        )

        if neg_total < 0:
            # Top-5 worst stock share of total loss
            top5_loss = float(worst_20.head(5)["total_contribution"].sum())
            top5_share = abs(top5_loss / neg_total) if neg_total else 0.0

            # Top-2 worst sector share of total loss
            top2_sec_share = 0.0
            if worst_dd_sector is not None:
                neg_sectors = worst_dd_sector.filter(
                    pl.col("total_contribution") < 0
                )
                if neg_sectors.height >= 2:
                    top2_loss = float(
                        neg_sectors.head(2)["total_contribution"].sum()
                    )
                    top2_sec_share = abs(top2_loss / neg_total) if neg_total else 0.0
                elif neg_sectors.height == 1:
                    top2_sec_share = 1.0

            if top5_share > 0.5:
                cause = "few_stocks"
            elif top2_sec_share > 0.5:
                cause = "sector_concentration"
            else:
                cause = "broad_weakness"

            dd_attr.update(
                cause=cause,
                top5_share=top5_share,
                top2_sec_share=top2_sec_share,
                total_loss=neg_total,
            )
            print(f"  Total loss in window     : {neg_total:+.4f}")
            print(f"  Top-5 worst stock share  : {top5_share:.1%}")
            print(f"  Top-2 worst sector share : {top2_sec_share:.1%}")
            print(f"  Implied cause            : {cause}")
        else:
            print("  Window did not produce aggregate losses; no classification.")

        # Always remember the tables for the report renderer.
        dd_attr["worst_dd"] = worst_20
        dd_attr["worst_dd_sector"] = worst_dd_sector
    else:
        print("  Drawdown window has no holdings rows; skipping attribution.")

    return {
        "frequency": freq_stats,
        "sector": sector_stats,
        "attribution": attribution,
        "drawdown": dd_attr,
    }


def _plot_sector_area(pivot: pl.DataFrame, output_dir: Path) -> None:
    """Stacked area chart of sector weights over time."""
    if pivot.height == 0:
        return
    sector_cols = [c for c in pivot.columns if c != "date"]
    if not sector_cols:
        return
    # Use the parsed datetime axis if we have it (cleaner x-labels).
    if "date_dt" in pivot.columns:
        x = pivot["date_dt"].to_list()
    else:
        x = pivot["date"].to_list()
    weights = np.column_stack([pivot[c].to_numpy() for c in sector_cols])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.stackplot(x, weights.T, labels=sector_cols, alpha=0.85)
    ax.set_title("Sector exposure over time (stacked weights)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total weight")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    fig.autofmt_xdate()
    fig.savefig(output_dir / "sector_exposure.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
def _md_table(df: Optional[pl.DataFrame], float_fmt: str = "{:.4f}") -> str:
    """Render a small Polars DataFrame as a Markdown pipe table."""
    if df is None or df.height == 0:
        return "_(no data)_"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * df.width) + " |"
    rows = []
    for row in df.iter_rows(named=True):
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def generate_report(
    health_stats: dict,
    holdings_stats: dict,
    output_dir: Path,
) -> None:
    """Emit analysis_summary.md with the key findings and recommendations."""
    _banner("3. Generating report")

    # --- Pull stats --------------------------------------------------------
    overall_ic = health_stats.get("mean_ic", float("nan"))
    overall_rank_ic = health_stats.get("mean_rank_ic", float("nan"))
    ic_positive = overall_ic > 0
    rank_ic_positive = overall_rank_ic > 0

    dd = health_stats.get("dd", {})
    dd_ic = dd.get("ic_mean_window", float("nan"))
    dd_rank_ic = dd.get("rank_ic_mean_window", float("nan"))
    dd_ic_neg_pct = dd.get("ic_neg_pct", float("nan"))
    dd_rank_ic_neg_pct = dd.get("rank_ic_neg_pct", float("nan"))
    ic_deteriorated = dd.get("ic_deteriorated", False)
    rank_ic_deteriorated = dd.get("rank_ic_deteriorated", False)

    # --- Drawdown cause ----------------------------------------------------
    dd_attr = holdings_stats.get("drawdown", {})
    cause = dd_attr.get("cause", "undetermined")
    cause_text = {
        "few_stocks": (
            "losses concentrated in a small number of stocks "
            "(top-5 worst stocks account for >50% of losses)"
        ),
        "sector_concentration": (
            "losses concentrated in a small number of sectors "
            "(top-2 worst sectors account for >50% of losses)"
        ),
        "broad_weakness": (
            "losses spread broadly across many stocks and sectors"
        ),
        "no_aggregate_loss": (
            "aggregate contributions were non-negative during the window"
        ),
        "undetermined": "insufficient evidence to classify",
    }.get(cause, cause)

    worst_dd = dd_attr.get("worst_dd")
    top5_md = (
        _md_table(worst_dd.head(5)) if worst_dd is not None else "_(insufficient data)_"
    )

    # --- Sector exposure comparison (all vs drawdown) ---------------------
    avg_sector = holdings_stats.get("sector", {}).get("avg_sector")
    dd_sector = holdings_stats.get("sector", {}).get("dd_sector")

    # Energy emphasis check (case-insensitive substring match).
    energy_line = "_No sector label contained the substring 'energy'._"
    if (
        avg_sector is not None
        and "sector" in avg_sector.columns
        and avg_sector.height > 0
    ):
        energy_rows_all = avg_sector.filter(
            pl.col("sector").cast(pl.Utf8).str.contains("(?i)energy")
        )
        if energy_rows_all.height > 0:
            w_all = float(energy_rows_all["avg_weight"].sum())
            if (
                dd_sector is not None
                and "sector" in dd_sector.columns
                and dd_sector.height > 0
            ):
                energy_rows_dd = dd_sector.filter(
                    pl.col("sector").cast(pl.Utf8).str.contains("(?i)energy")
                )
                if energy_rows_dd.height > 0:
                    w_dd = float(energy_rows_dd["avg_weight"].sum())
                    direction = (
                        "Overweighted" if w_dd > w_all else "Underweighted"
                    )
                    energy_line = (
                        f"Energy sector average weight: **{w_all:.2%}** "
                        f"(all dates) vs **{w_dd:.2%}** (drawdown window). "
                        f"**{direction}** during the drawdown."
                    )
                else:
                    energy_line = (
                        f"Energy sector average weight (all dates): "
                        f"**{w_all:.2%}**. No energy exposure during the drawdown window."
                    )
            else:
                energy_line = (
                    f"Energy sector average weight (all dates): **{w_all:.2%}**."
                )

    # --- Recommendations ---------------------------------------------------
    recs: list[str] = []
    if ic_positive and not ic_deteriorated:
        recs.append(
            "- Model IC is positive overall and stable; no global remediation required."
        )
    if not ic_positive:
        recs.append(
            "- Overall mean IC is non-positive; consider re-training or feature review."
        )
    if ic_deteriorated or rank_ic_deteriorated:
        recs.append(
            "- IC/Rank IC deteriorated during "
            f"{DRAWDOWN_START} - {DRAWDOWN_END}: add an IC health gate. "
            "Concrete mitigation: if the 20-step rolling IC is < 0 for 10 "
            "consecutive steps, halve position size; if < 0 for 20 steps, "
            "pause the strategy until rolling IC recovers."
        )
    if cause == "few_stocks":
        recs.append(
            "- Losses concentrated in a few stocks: tighten per-name caps and "
            "add stock-level stop-loss / max-contribution guardrails."
        )
    if cause == "sector_concentration":
        recs.append(
            "- Losses concentrated in a few sectors: enforce stricter sector "
            "caps (e.g., max 25% per sector) and add a sector-neutralisation overlay."
        )
    if cause == "broad_weakness":
        recs.append(
            "- Losses are broad-based: signal is likely mis-calibrated. Re-fit "
            "the model on more recent data and consider regime conditioning."
        )
    if cause == "no_aggregate_loss":
        recs.append(
            "- Aggregate contributions were non-negative in the window; the "
            "drawdown narrative is more about *underperformance vs SPY* than "
            "absolute losses -- revisit the benchmark comparison."
        )
    if not recs:
        recs.append("- No automatic recommendations triggered; inspect charts manually.")

    # --- Build the markdown ------------------------------------------------
    lines: list[str] = []
    lines.append("# GreekFire Backtest Analysis Summary")
    lines.append("")
    lines.append("_Generated by `analyze_greekfire.py`._")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("### Overall model health")
    lines.append("")
    lines.append(
        f"- Mean IC: **{overall_ic:+.4f}**  "
        f"({'positive' if ic_positive else 'non-positive'})"
    )
    lines.append(
        f"- Mean Rank IC: **{overall_rank_ic:+.4f}**  "
        f"({'positive' if rank_ic_positive else 'non-positive'})"
    )
    lines.append(
        f"- Mean portfolio return: **{health_stats.get('mean_port_return', float('nan')):+.4f}**"
    )
    lines.append(
        f"- Mean SPY return: **{health_stats.get('mean_spy_return', float('nan')):+.4f}**"
    )
    if "std_ic" in health_stats:
        lines.append(f"- Std IC: **{health_stats['std_ic']:.4f}**")
    if "std_rank_ic" in health_stats:
        lines.append(f"- Std Rank IC: **{health_stats['std_rank_ic']:.4f}**")
    lines.append("")

    lines.append(
        f"### Model health during drawdown window ({DRAWDOWN_START} - {DRAWDOWN_END})"
    )
    lines.append("")
    lines.append(
        f"- Window steps: **{dd.get('window_steps', 0)}** "
        f"of {dd.get('total_steps', 0)} total"
    )
    lines.append(
        f"- Mean IC in window: **{dd_ic:+.4f}**  (overall {overall_ic:+.4f})"
    )
    lines.append(
        f"- Mean Rank IC in window: **{dd_rank_ic:+.4f}**  "
        f"(overall {overall_rank_ic:+.4f})"
    )
    lines.append(f"- IC < 0 in {dd_ic_neg_pct:.1%} of window steps")
    lines.append(f"- Rank IC < 0 in {dd_rank_ic_neg_pct:.1%} of window steps")
    lines.append(
        f"- IC deteriorated: **{ic_deteriorated}** | "
        f"Rank IC deteriorated: **{rank_ic_deteriorated}**"
    )
    lines.append("")

    lines.append("## Drawdown Causes")
    lines.append("")
    lines.append("### Sector emphasis")
    lines.append("")
    lines.append(energy_line)
    lines.append("")

    lines.append("### Ranking failure")
    lines.append("")
    if not (dd_ic != dd_ic) and not (dd_rank_ic != dd_rank_ic):
        if dd_ic < 0 or dd_rank_ic < 0:
            lines.append(
                f"- **Yes**, ranking failure is supported by the data: the "
                f"model's IC and/or Rank IC was negative in the drawdown "
                f"window (IC {dd_ic:+.4f}, Rank IC {dd_rank_ic:+.4f})."
            )
        else:
            lines.append(
                f"- Ranking did **not** clearly fail on average "
                f"(IC {dd_ic:+.4f}, Rank IC {dd_rank_ic:+.4f}), so losses "
                f"are unlikely to be due to a pure ranking collapse."
            )
    else:
        lines.append("- Insufficient data to assess ranking failure.")
    lines.append("")

    lines.append("### Loss concentration")
    lines.append("")
    lines.append(f"- Implied cause: **{cause_text}**")
    if "top5_share" in dd_attr:
        lines.append(
            f"- Top-5 worst stock share of loss: **{dd_attr['top5_share']:.1%}**"
        )
    if "top2_sec_share" in dd_attr:
        lines.append(
            f"- Top-2 worst sector share of loss: **{dd_attr['top2_sec_share']:.1%}**"
        )
    if "total_loss" in dd_attr:
        lines.append(f"- Total realized loss in window: **{dd_attr['total_loss']:+.4f}**")
    lines.append("")

    lines.append("### Top-5 worst contributors during drawdown")
    lines.append("")
    lines.append(top5_md)
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.extend(recs)
    lines.append("")

    (output_dir / "analysis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"  Wrote {output_dir / 'analysis_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    holdings, health = load_data()
    health_stats = analyze_model_health(health, output_dir)
    holdings_stats = analyze_holdings(holdings, output_dir)
    generate_report(health_stats, holdings_stats, output_dir)

    print()
    _banner("Done")
    print(f"All artefacts saved under: {output_dir}/")


if __name__ == "__main__":
    main()
