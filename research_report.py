"""Generate an exploratory research report from a CURRENT GreekFire validation export.

Run: python research_report.py --steps results/corrected/validation_holdout_A1_steps.csv
Check arithmetic: python research_report.py --self-test
"""

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


PERIODS_PER_YEAR = 252 / 5
REQUIRED = {"methodology", "date", "actual_alpha_long", "net_alpha_long", "long_cost",
            "turnover", "spy_return", "long_tickers"}


def read_steps(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Old export: rerun Julia validation to get net, cost, and aligned SPY fields")
        rows = list(reader)
    if not rows or [row["date"] for row in rows] != sorted({row["date"] for row in rows}):
        raise ValueError("Steps must have unique, sorted dates")
    for row in rows:
        if row["methodology"] != "corrected-v2":
            raise ValueError("Historical export is not corrected-v2")
        for field in ("actual_alpha_long", "net_alpha_long", "long_cost", "turnover"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite {field} on {row['date']}")
        if not math.isclose(row["actual_alpha_long"] - row["long_cost"],
                            row["net_alpha_long"], abs_tol=1e-8):
            raise ValueError(f"Gross/net accounting mismatch on {row['date']}")
        row["spy_return"] = float(row["spy_return"]) if row["spy_return"] else None
    rates = [row["long_cost"] / row["turnover"] for row in rows if row["turnover"] > 0]
    if not rates or any(not math.isclose(rate, rates[0], abs_tol=1e-10) for rate in rates):
        raise ValueError("Missing or inconsistent portfolio cost rate in validation export")
    return rows


def statistics_for(returns):
    if len(returns) < 2 or any(not math.isfinite(value) or value <= -1 for value in returns):
        raise ValueError("Need at least two finite step returns above -100%")
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    curve = [wealth]
    drawdowns = [0.0]
    for step_return in returns:
        wealth *= 1 + step_return
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1 - wealth / peak)
        curve.append(wealth)
        drawdowns.append(1 - wealth / peak)
    cagr = wealth ** (PERIODS_PER_YEAR / len(returns)) - 1
    volatility = statistics.stdev(returns) * math.sqrt(PERIODS_PER_YEAR)
    downside = math.sqrt(sum(min(value, 0) ** 2 for value in returns) / len(returns))
    return {
        "cagr": cagr, "volatility": volatility,
        "sharpe": statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(PERIODS_PER_YEAR)
                  if volatility else 0.0,
        "sortino": statistics.mean(returns) / downside * math.sqrt(PERIODS_PER_YEAR)
                   if downside else float("nan"),
        "max_drawdown": max_drawdown,
        "calmar": cagr / max_drawdown if max_drawdown else float("nan"),
        "cumulative": wealth - 1, "curve": curve, "drawdowns": drawdowns,
    }


def block_interval(differences, block_size=12, iterations=500):
    generator = random.Random(42)
    length = len(differences)
    if length < 5 * block_size:
        return None
    sample_means = []
    for _ in range(iterations):
        sample = []
        while len(sample) < length:
            start = generator.randrange(length - block_size + 1)
            sample.extend(differences[start:start + block_size])
        sample_means.append(statistics.mean(sample[:length]))
    sample_means.sort()
    return sample_means[int(0.025 * iterations)], sample_means[int(0.975 * iterations)]


def plot(curves, dates, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    for key, field, title, filename in (
        ("curve", "Equity (initial = 1)", "Aligned equity curves", "research_equity.png"),
        ("drawdowns", "Drawdown", "Peak-to-trough drawdown", "research_drawdown.png"),
    ):
        figure, axis = pyplot.subplots(figsize=(10, 4))
        for name, stats in curves.items():
            axis.plot(range(len(dates) + 1), stats[key], label=name, linewidth=1.5)
        axis.set(xlabel="Five-day steps", ylabel=field, title=title)
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=140)
        pyplot.close(figure)


def generate(rows, output_path, robustness_dir):
    aligned = [row for row in rows if row["spy_return"] is not None and
               math.isfinite(row["spy_return"])]
    if len(aligned) < 12:
        raise ValueError("Need at least 12 matching SPY observations; no zero-filled benchmark")
    total_cost_bps = next(row["long_cost"] / row["turnover"] * 10_000
                          for row in rows if row["turnover"] > 0)
    gross = [row["actual_alpha_long"] for row in aligned]
    net = [row["net_alpha_long"] for row in aligned]
    spy = [row["spy_return"] for row in aligned]
    results = {"GreekFire gross": statistics_for(gross),
               "GreekFire net": statistics_for(net),
               "SPY (no ETF costs)": statistics_for(spy)}
    differences = [strategy - benchmark for strategy, benchmark in zip(net, spy)]
    interval = block_interval(differences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot(results, [row["date"] for row in aligned], output_path.parent)
    lines = ["# GreekFire research audit (generated)", "",
             f"Input: `{len(rows)}` steps; `{len(aligned)}` aligned SPY observations from "
             f"{aligned[0]['date']} to {aligned[-1]['date']}.", "",
             "## Method", "",
             "Local historical adjusted price CSVs, a date-ranged S&P 500 membership file, and a "
             "explicit membership filtering supply the universe. No lifetime ticker blacklist or "
             "acquirer remapping is applied. Source CSVs are not distributed. Features include "
             "momentum, trend, volatility, volume, and market-relative measures. A seeded Random "
             "Forest and seeded XGBoost provide forecasts. The reference configuration blends them "
             "50/50, retrains over a rolling 252-session window every five trading days, and holds "
             "ten sector-capped, equal-weight long stocks. Fixed matrix variants alter one named "
             "parameter; check the run command for the input CSV before applying these settings. "
             "Shorts and HMM overlay are disabled.", "",
             "The signal uses information through a stock's close; the target buys at the next "
             "adjusted open and exits at the fifth adjusted close. Training labels must finish by "
             "train_end. Results below use only dates with a matching SPY execution window. "
             "Gross and cost-adjusted net returns use the same holdings, with "
             f"{total_cost_bps:.1f} bps total assumed transaction cost plus slippage per unit "
             "of executed entry-plus-exit notional stock turnover, including the initial entry. "
             "SPY uses the same entry/exit dates without ETF costs. "
             "SPY turnover and ETF trading costs are not modeled; it is an uncosted reference, "
             "not a net-of-cost investable replica. No live fills are modeled.", "",
             "## Aligned performance", "",
             "| Series | CAGR | Annual vol | Sharpe (rf=0) | Sortino (rf=0) | Max DD | Calmar | Cumulative |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        lines.append("| " + name + " | " + " | ".join([
            f"{result['cagr']:.2%}", f"{result['volatility']:.2%}",
            f"{result['sharpe']:.2f}", f"{result['sortino']:.2f}",
            f"{result['max_drawdown']:.2%}", f"{result['calmar']:.2f}",
            f"{result['cumulative']:.2%}"]) + " |")
    lines.extend(["", f"Mean executed entry-plus-exit notional stock turnover/step: {statistics.mean(row['turnover'] for row in aligned):.1%}.",
                  f"Mean charged cost/step: {statistics.mean(row['long_cost'] for row in aligned):.3%}.",
                  "", "![Aligned equity](research_equity.png)",
                  "", "![Drawdown](research_drawdown.png)",
                  "", "## Cost sensitivity (same positions, no model retraining)", "",
                  "| Total cost per unit executed entry-plus-exit notional turnover | CAGR | Sharpe (rf=0) | Max DD |",
                  "| ---: | ---: | ---: | ---: |"])
    for bps in (0, 15, 30):
        result = statistics_for([row["actual_alpha_long"] - bps / 10_000 * row["turnover"]
                                 for row in aligned])
        lines.append(f"| {bps} bps | {result['cagr']:.2%} | {result['sharpe']:.2f} | {result['max_drawdown']:.2%} |")
    names = ("reference", "shorter_training", "fewer_longs", "rank_weights", "model_blend")
    archived = [robustness_dir / f"{name}_validation_holdout_A1_steps.csv" for name in names]
    completed = []
    incomplete = []
    reference_dates = [row["date"] for row in aligned]
    for name, path in zip(names, archived):
        if not path.exists():
            incomplete.append(name)
            continue
        variant = read_steps(path)
        if [row["date"] for row in variant] == reference_dates:
            completed.append((name, variant))
        else:
            incomplete.append(name)
    lines.extend(["", "## Fixed-parameter robustness (incomplete matrix)", "",
                  "Only variants with the same observed dates as the reference are tabulated. "
                  "No winner is selected from this historical period.", "",
                  "| Configuration | Net CAGR | Sharpe (rf=0) | Max DD |",
                  "| --- | ---: | ---: | ---: |"])
    for name, variant in completed:
        result = statistics_for([row["net_alpha_long"] for row in variant])
        lines.append(f"| {name} | {result['cagr']:.2%} | {result['sharpe']:.2f} | {result['max_drawdown']:.2%} |")
    if incomplete:
        lines.append(f"\nNot completed on matching dates: {', '.join(incomplete)}. Short smoke runs are not full-period evidence.")
    lines.append("No short-count, stop-loss, or holding-period claim is supported by the disabled overlay.")
    sorted_net = sorted(net)
    lines.extend(["", "## Distribution and uncertainty", "",
                  f"Net five-day returns: 5th percentile {sorted_net[int(.05 * len(net))]:.2%}; "
                  f"median {sorted_net[len(net)//2]:.2%}; 95th percentile {sorted_net[int(.95 * len(net))]:.2%}.",
                  f"Observed average net minus SPY per five-day step: {statistics.mean(differences):.3%}."])
    lines.extend(["", "## Descriptive market conditions", "",
                  "These buckets use each interval's **realized** SPY return; they are not available "
                  "at the signal date and must not be used for portfolio selection."])
    for label, predicate in (("SPY up", lambda value: value > 0),
                             ("SPY flat/down", lambda value: value <= 0)):
        matching = [row["net_alpha_long"] for row in aligned if predicate(row["spy_return"])]
        if matching:
            lines.append(f"{label}: {len(matching)} intervals, mean GreekFire net step return "
                         f"{statistics.mean(matching):.2%}.")
    if interval:
        lines.append(f"Exploratory 95% moving-block bootstrap interval (12 steps, seed 42, 500 draws) "
                     f"for that mean difference: [{interval[0]:.3%}, {interval[1]:.3%}]. "
                     "This is not an alpha test and does not correct for past parameter selection.")
    else:
        lines.append("No bootstrap interval: fewer than 60 aligned steps; uncertainty is not estimable by this procedure.")
    lines.extend(["", "## Limitations and decision", "",
                  "Historical dates here have been repeatedly inspected and tuned; they are NOT a "
                  "fresh independent out-of-sample test. Corrected-v2 retains signals independently "
                  "of future targets, applies membership before ranks, preserves security identity "
                  "and refuses incomplete outcome coverage. Membership provenance, missing delisting "
                  "outcomes, corporate actions, liquidity and assumed fills still need verification. "
                  "Costs charge entry and liquidation notional. Cash between windows earns zero. "
                  "Short overlay and capped-loss proxy are disabled. A new untouched period and "
                  "independently verified data are required before making an alpha claim.", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path} and two plots ({len(aligned)} aligned steps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, default=Path("results/corrected/validation_holdout_A1_steps.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/corrected/research_report.md"))
    parser.add_argument("--robustness-dir", type=Path, default=Path("results/corrected/robustness"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = statistics_for([0.10, -0.10])
        assert math.isclose(result["cumulative"], -0.01)
        assert math.isclose(result["max_drawdown"], 0.10)
        assert block_interval([0.01] * 24) is None
        assert block_interval([0.01] * 60) == (0.01, 0.01)
        print("Report arithmetic and deterministic bootstrap checks passed.")
    else:
        generate(read_steps(args.steps), args.output, args.robustness_dir)


if __name__ == "__main__":
    main()