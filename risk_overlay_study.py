"""Small, no-training risk overlay experiment on the saved momentum holdout.

Run: python risk_overlay_study.py
Check timing and input assumptions: python risk_overlay_study.py --self-test
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS_PER_YEAR = 252 / 5
WINDOW = 20
TARGET_VOL = 0.20
MIN_VOL = 0.05
COST_PER_TURNOVER = 0.002


def load_steps(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < WINDOW + 2 or any(not row["date"] for row in rows):
        raise ValueError("Expected a dated holdout with at least 22 steps")
    dates = [row["date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Holdout dates must be unique and ordered")
    for row in rows:
        row["return"] = float(row["actual_alpha_long"])
        tickers = row["long_tickers"].split(";")
        weights = [float(weight) for weight in row["weights"].split(";")]
        if (len(tickers) != len(weights) or len(set(tickers)) != len(tickers)
                or not math.isfinite(row["return"]) or row["return"] <= -1
                or not all(math.isfinite(weight) and weight >= 0 for weight in weights)
                or not math.isclose(sum(weights), 1, abs_tol=1e-5)):
            raise ValueError(f"Invalid holdings or return on {row['date']}")
        row["holdings"] = dict(zip(tickers, weights))
    return rows


def exposure(prior_returns):
    if len(prior_returns) < WINDOW:
        return 1.0
    realized_vol = statistics.stdev(prior_returns[-WINDOW:]) * math.sqrt(STEPS_PER_YEAR)
    return min(1.0, TARGET_VOL / max(MIN_VOL, realized_vol))


def evaluate(rows, managed):
    returns = []
    exposures = []
    old_holdings = {}
    for index, row in enumerate(rows):
        prior_returns = [item["return"] for item in rows[:index]]
        scale = exposure(prior_returns) if managed else 1.0
        holdings = {ticker: weight * scale for ticker, weight in row["holdings"].items()}
        # Include the cash leg; moving from 100% cash to 100% stock costs one unit.
        turnover = 0.5 * (
            sum(abs(holdings.get(ticker, 0) - old_holdings.get(ticker, 0))
                for ticker in holdings.keys() | old_holdings.keys())
            + abs((1 - scale) - (1 - sum(old_holdings.values())))
        )
        returns.append(scale * row["return"] - COST_PER_TURNOVER * turnover)
        exposures.append(scale)
        old_holdings = holdings
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for step_return in returns:
        wealth *= 1 + step_return
        peak = max(peak, wealth)
        drawdown = max(drawdown, 1 - wealth / peak)
    return {
        "cagr": wealth ** (STEPS_PER_YEAR / len(returns)) - 1,
        "sharpe": statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(STEPS_PER_YEAR),
        "max_drawdown": drawdown,
        "mean_exposure": statistics.mean(exposures),
    }


def self_test():
    history = [0.01] * WINDOW
    assert exposure(history[:-1]) == 1.0
    assert exposure(history) == 1.0
    assert exposure(history + [0.5]) < 1.0
    rows = load_steps(ROOT / "octave" / "momentum_baseline_steps.csv")
    with (ROOT / "octave" / "momentum_baseline_summary.csv").open(newline="", encoding="utf-8") as source:
        expected = next(csv.DictReader(source))
    assert expected["config"] == "MOMENTUM"
    wealth = math.prod(1 + row["return"] for row in rows)
    assert math.isclose(wealth ** (STEPS_PER_YEAR / len(rows)) - 1, float(expected["cagr"]), abs_tol=1e-10)
    print("Timing and baseline reproduction checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows = load_steps(ROOT / "octave" / "momentum_baseline_steps.csv")
    print(f"Saved holdout: {rows[0]['date']} to {rows[-1]['date']} ({len(rows)} five-day steps)")
    print("Rule: 20-step trailing volatility, 20% annual target, long-only, max 100% exposure")
    print("Assumptions: zero cash yield, 20 bps per unit one-way turnover; no new stock backtest")
    for name, managed in (("Momentum baseline", False), ("Volatility overlay", True)):
        result = evaluate(rows, managed)
        print(f"{name}: CAGR {result['cagr']:.2%}, Sharpe (rf=0) {result['sharpe']:.2f}, "
              f"max drawdown {result['max_drawdown']:.2%}, mean exposure {result['mean_exposure']:.1%}")


if __name__ == "__main__":
    main()