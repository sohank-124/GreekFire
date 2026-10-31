#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GreekFire Turnover / Transaction-Cost / Slippage / Ranking-Stability Audit

This script is the core evidence generator for the audit. It reads:
  - octave/holdings.csv       (PRIMARY source for A1 and F1 baskets)
  - octave/model_health.csv   (cross-validation source)

It computes:
  PHASE 1 Data load & validation
  PHASE 2 Turnover (membership + weight-based, A1 vs F1)
  PHASE 3 Ranking stability (persistence decay, holding period, rank correlation)
  PHASE 4 Gross return reconstruction (A1 + F1) and cross-validation
  PHASE 5 Transaction cost analysis (S1..S4)
  PHASE 6 Adversarial assessment
  PHASE 7 Structured stdout output + results CSV
"""

import os
import sys
import math
import warnings
from collections import defaultdict

# Force UTF-8 stdout (Windows console defaults to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
HOLDINGS_CSV = os.path.join(HERE, "holdings.csv")
MODEL_HEALTH_CSV = os.path.join(HERE, "model_health.csv")
RESULTS_CSV = os.path.join(HERE, "turnover_audit_results.csv")

STEPS_PER_YEAR = 50.4
RISK_FREE_PER_STEP = 0.05 / STEPS_PER_YEAR
SAR_SIZE = 10  # long positions per step

COST_SCENARIOS = {
    "S1 (5bps/side)":  0.001,
    "S2 (10bps/side)": 0.002,
    "S3 (15bps/side)": 0.003,
    "S4 (20bps/side)": 0.004,
}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def banner(title):
    line = "=" * 100
    print()
    print(line)
    print(title)
    print(line)


def section(title):
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


# ---------------------------------------------------------------------------
# PHASE 1 - Load & Validate
# ---------------------------------------------------------------------------
def load_and_validate():
    section("PHASE 1: DATA LOAD & VALIDATION")
    df = pd.read_csv(HOLDINGS_CSV)
    print(f"Loaded holdings.csv: {len(df)} rows, {df['step_index'].nunique()} unique steps")
    print(f"Columns: {list(df.columns)}")
    print(f"Side counts:\n{df['side'].value_counts().to_string()}")

    long_df = df[df["side"] == "long"].copy().sort_values(["step_index", "rank"]).reset_index(drop=True)
    print(f"\nFiltered to long side: {len(long_df)} rows")

    n_steps = long_df["step_index"].nunique()
    step_min, step_max = long_df["step_index"].min(), long_df["step_index"].max()
    print(f"Steps: count={n_steps}, min={step_min}, max={step_max}")

    issues = []

    # 10 longs per step
    counts = long_df.groupby("step_index").size()
    bad_steps = counts[counts != SAR_SIZE]
    if len(bad_steps) > 0:
        issues.append(f"Positions-per-step violations: {len(bad_steps)} steps != {SAR_SIZE}. "
                       f"First few: {bad_steps.head(5).to_dict()}")
    else:
        print(f"[OK] All {n_steps} steps have exactly {SAR_SIZE} long positions")

    # Contiguous steps
    expected_steps = set(range(step_min, step_max + 1))
    missing = expected_steps - set(long_df["step_index"].unique())
    if missing:
        issues.append(f"Missing step_indices: {sorted(missing)}")
    else:
        print(f"[OK] All step_indices contiguous from {step_min} to {step_max}")

    # equal_weight sums to 1
    ew_sum = long_df.groupby("step_index")["equal_weight"].sum()
    ew_viol = ew_sum[ew_sum.abs() - 1.0 > 0.001]
    if len(ew_viol) > 0:
        issues.append(f"equal_weight sum violation at {len(ew_viol)} steps. "
                       f"Example: {ew_viol.head().to_dict()}")
    else:
        print(f"[OK] Sum of equal_weight ≈ 1.0 for all {len(ew_sum)} steps "
              f"(min={ew_sum.min():.6f}, max={ew_sum.max():.6f})")

    # inv_vol_weight sums to 1
    iv_sum = long_df.groupby("step_index")["inv_vol_weight"].sum()
    iv_viol = iv_sum[iv_sum.abs() - 1.0 > 0.001]
    if len(iv_viol) > 0:
        issues.append(f"inv_vol_weight sum violation at {len(iv_viol)} steps. "
                       f"Example: {iv_viol.head().to_dict()}")
    else:
        print(f"[OK] Sum of inv_vol_weight ≈ 1.0 for all {len(iv_sum)} steps "
              f"(min={iv_sum.min():.6f}, max={iv_sum.max():.6f})")

    # Load model_health
    mh = pd.read_csv(MODEL_HEALTH_CSV)
    print(f"\nLoaded model_health.csv: {len(mh)} rows, {mh['step_index'].nunique()} steps")
    print(f"Columns: {list(mh.columns)}")

    # Cross-validate F1 reconstruction against model_health portfolio_return
    f1_per_step = long_df.groupby("step_index")["contribution"].sum()
    mh_pr = mh.set_index("step_index")["portfolio_return"]
    common = f1_per_step.index.intersection(mh_pr.index)
    diff = (f1_per_step.loc[common] - mh_pr.loc[common]).abs()
    max_diff = diff.max()
    if max_diff > 1e-6:
        issues.append(f"F1 reconstructed return vs model_health portfolio_return: "
                       f"max abs diff = {max_diff:.10f} (> 1e-6). "
                       f"Worst step: {diff.idxmax()}")
    else:
        print(f"[OK] F1 reconstructed return matches model_health portfolio_return "
              f"(max abs diff = {max_diff:.3e}, mean diff = {diff.mean():.3e})")

    if issues:
        print("\nVALIDATION ISSUES:")
        for i, msg in enumerate(issues, 1):
            print(f"  [{i}] {msg}")
    else:
        print("\n[OK] No validation issues found")

    return long_df, mh, issues


# ---------------------------------------------------------------------------
# PHASE 2 - Turnover
# ---------------------------------------------------------------------------
def compute_turnover(long_df):
    section("PHASE 2: TURNOVER COMPUTATION")

    # Build per-step structures
    grouped = long_df.groupby("step_index")
    tickers_by_step = {s: set(g["ticker"]) for s, g in grouped}
    ew_by_step = {s: dict(zip(g["ticker"], g["equal_weight"])) for s, g in grouped}
    iv_by_step = {s: dict(zip(g["ticker"], g["inv_vol_weight"])) for s, g in grouped}

    steps_sorted = sorted(tickers_by_step.keys())
    n_trans = len(steps_sorted) - 1
    print(f"Steps sorted: {len(steps_sorted)}, transitions to compute: {n_trans}")

    membership_turnover = []
    positions_replaced = []
    turnover_A1 = []
    turnover_F1 = []
    f1_entry_exit = []
    f1_rebalance = []

    for i in range(n_trans):
        s_t = steps_sorted[i]
        s_tp1 = steps_sorted[i + 1]
        S_t = tickers_by_step[s_t]
        S_tp1 = tickers_by_step[s_tp1]
        inter = S_t & S_tp1
        mto = 1.0 - len(inter) / max(len(S_t), 1)
        membership_turnover.append(mto)
        positions_replaced.append(len(S_t) - len(inter))

        all_t = S_t | S_tp1
        # A1 weight-based turnover
        d_A1 = 0.0
        d_F1 = 0.0
        for tkr in all_t:
            w_A1_t = ew_by_step[s_t].get(tkr, 0.0)
            w_A1_tp1 = ew_by_step[s_tp1].get(tkr, 0.0)
            w_F1_t = iv_by_step[s_t].get(tkr, 0.0)
            w_F1_tp1 = iv_by_step[s_tp1].get(tkr, 0.0)
            d_A1 += abs(w_A1_t - w_A1_tp1)
            d_F1 += abs(w_F1_t - w_F1_tp1)
        turnover_A1.append(0.5 * d_A1)
        turnover_F1.append(0.5 * d_F1)

        # F1 decomposition
        leaving = S_t - S_tp1
        entering = S_tp1 - S_t
        ee = 0.5 * (sum(iv_by_step[s_t][tkr] for tkr in leaving)
                    + sum(iv_by_step[s_tp1][tkr] for tkr in entering))
        reb = 0.5 * sum(abs(iv_by_step[s_t][tkr] - iv_by_step[s_tp1][tkr]) for tkr in inter)
        f1_entry_exit.append(ee)
        f1_rebalance.append(reb)

    membership_turnover = np.array(membership_turnover)
    positions_replaced = np.array(positions_replaced, dtype=float)
    turnover_A1 = np.array(turnover_A1)
    turnover_F1 = np.array(turnover_F1)
    f1_entry_exit = np.array(f1_entry_exit)
    f1_rebalance = np.array(f1_rebalance)

    # Sanity: F1 turnover should equal entry_exit + rebalance
    f1_check = f1_entry_exit + f1_rebalance
    f1_decomp_err = np.max(np.abs(f1_check - turnover_F1))
    print(f"[CHECK] F1 turnover |entry_exit + rebalance - turnover| max err = {f1_decomp_err:.3e}")

    # Sanity: A1 weight turnover should equal membership turnover * 0.1 * 2 * ... wait no
    # For equal weights, turnover_A1 per transition = positions_replaced * 0.1
    #   since each replaced stock adds 0.5*(0.1 + 0.1) = 0.1 to turnover
    a1_check = positions_replaced * 0.1
    a1_check_err = np.max(np.abs(a1_check - turnover_A1))
    print(f"[CHECK] A1 turnover |positions_replaced*0.1 - turnover| max err = {a1_check_err:.3e}")

    def stats(arr, label):
        return {
            "label": label,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    summary = {
        "membership": stats(membership_turnover, "membership"),
        "A1": stats(turnover_A1, "A1 (equal_weight)"),
        "F1": stats(turnover_F1, "F1 (inv_vol_weight)"),
        "positions_replaced": stats(positions_replaced, "positions_replaced"),
    }

    # Annualized (turnover per step * steps per year)
    summary["A1"]["annualized"] = summary["A1"]["mean"] * STEPS_PER_YEAR
    summary["F1"]["annualized"] = summary["F1"]["mean"] * STEPS_PER_YEAR
    summary["membership"]["annualized"] = summary["membership"]["mean"] * STEPS_PER_YEAR

    # Distribution table (by A1 weight turnover)
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001)]
    print(f"\nMembership turnover: mean={summary['membership']['mean']:.6f} "
          f"median={summary['membership']['median']:.6f} "
          f"max={summary['membership']['max']:.6f}")
    print(f"Positions replaced per rebalance: mean={summary['positions_replaced']['mean']:.4f} "
          f"median={summary['positions_replaced']['median']:.1f} "
          f"max={summary['positions_replaced']['max']:.0f}")
    print(f"Turnover A1: mean={summary['A1']['mean']:.6f} median={summary['A1']['median']:.6f} "
          f"std={summary['A1']['std']:.6f} min={summary['A1']['min']:.6f} max={summary['A1']['max']:.6f}")
    print(f"Turnover F1: mean={summary['F1']['mean']:.6f} median={summary['F1']['median']:.6f} "
          f"std={summary['F1']['std']:.6f} min={summary['F1']['min']:.6f} max={summary['F1']['max']:.6f}")
    print(f"Annualized turnover A1 = {summary['A1']['annualized']:.4f}")
    print(f"Annualized turnover F1 = {summary['F1']['annualized']:.4f}")
    print(f"Annualized membership turnover = {summary['membership']['annualized']:.4f}")
    print(f"% steps with membership turnover > 0.8 = "
          f"{100.0*np.mean(membership_turnover > 0.8):.2f}%")
    print(f"% steps with membership turnover = 1.0 (full replacement) = "
          f"{100.0*np.mean(membership_turnover >= 1.0 - 1e-9):.2f}%")

    # Distribution buckets — by membership turnover
    print("\nTurnover Distribution (by membership turnover %):")
    print(f"  {'Bucket':<14} {'Count':>8} {'Pct':>10}")
    for lo, hi in buckets:
        if hi > 1.0:
            mask = (membership_turnover >= lo) & (membership_turnover <= hi)
        else:
            mask = (membership_turnover >= lo) & (membership_turnover < hi)
        cnt = int(mask.sum())
        pct = 100.0 * cnt / len(membership_turnover)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:>8} {pct:>9.2f}%")

    # Same distribution by A1 weight turnover
    print("\nTurnover Distribution (by A1 weight turnover):")
    print(f"  {'Bucket':<14} {'Count':>8} {'Pct':>10}")
    for lo, hi in buckets:
        if hi > 1.0:
            mask = (turnover_A1 >= lo) & (turnover_A1 <= hi)
        else:
            mask = (turnover_A1 >= lo) & (turnover_A1 < hi)
        cnt = int(mask.sum())
        pct = 100.0 * cnt / len(turnover_A1)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:>8} {pct:>9.2f}%")

    # Same distribution by F1 weight turnover
    print("\nTurnover Distribution (by F1 weight turnover):")
    print(f"  {'Bucket':<14} {'Count':>8} {'Pct':>10}")
    for lo, hi in buckets:
        if hi > 1.0:
            mask = (turnover_F1 >= lo) & (turnover_F1 <= hi)
        else:
            mask = (turnover_F1 >= lo) & (turnover_F1 < hi)
        cnt = int(mask.sum())
        pct = 100.0 * cnt / len(turnover_F1)
        print(f"  [{lo:.1f}, {hi:.1f})  {cnt:>8} {pct:>9.2f}%")

    # F1 decomposition summary
    print("\nF1 Turnover Decomposition:")
    print(f"  mean entry/exit component = {np.mean(f1_entry_exit):.6f} "
          f"({100.0*np.mean(f1_entry_exit)/np.mean(turnover_F1):.1f}% of F1 turnover)")
    print(f"  mean rebalance component = {np.mean(f1_rebalance):.6f} "
          f"({100.0*np.mean(f1_rebalance)/np.mean(turnover_F1):.1f}% of F1 turnover)")
    print(f"  When there is membership change, rebalance component is small "
          f"(because weights simply redistribute).")

    return {
        "steps_sorted": steps_sorted,
        "tickers_by_step": tickers_by_step,
        "ew_by_step": ew_by_step,
        "iv_by_step": iv_by_step,
        "membership_turnover": membership_turnover,
        "positions_replaced": positions_replaced,
        "turnover_A1": turnover_A1,
        "turnover_F1": turnover_F1,
        "f1_entry_exit": f1_entry_exit,
        "f1_rebalance": f1_rebalance,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# PHASE 3 - Ranking Stability
# ---------------------------------------------------------------------------
def compute_ranking_stability(long_df, turnover_data):
    section("PHASE 3: RANKING STABILITY")

    tickers_by_step = turnover_data["tickers_by_step"]
    steps_sorted = turnover_data["steps_sorted"]
    n_steps = len(steps_sorted)
    step_idx = {s: i for i, s in enumerate(steps_sorted)}

    grouped = long_df.groupby("step_index")
    rank_by_step = {s: dict(zip(g["ticker"], g["rank"])) for s, g in grouped}

    # Basket persistence for k in [1,2,4,8,16]
    ks = [1, 2, 4, 8, 16]
    print("\nBasket Persistence Decay Curve (mean overlap with t+k):")
    print(f"  {'k (steps)':<10} {'k (weeks)':<10} {'Valid pairs':<12} {'Mean overlap':<14} {'Persistence %':<14}")
    persistence = {}
    for k in ks:
        overlaps = []
        for i in range(n_steps - k):
            S_t = tickers_by_step[steps_sorted[i]]
            S_tpk = tickers_by_step[steps_sorted[i + k]]
            overlaps.append(len(S_t & S_tpk) / SAR_SIZE)
        if overlaps:
            m = float(np.mean(overlaps))
            persistence[k] = m
            print(f"  {k:<10} {k:<10} {len(overlaps):<12} "
                  f"{m*10:<14.3f} {100.0*m:>10.2f}%")
        else:
            persistence[k] = float("nan")
            print(f"  {k:<10} {k:<10} {0:<12} {'n/a':<14} {'n/a':<14}")

    # Holding period distribution
    # Identify entry events: ticker present at step t but not at step t-1 (or t=0)
    holdings_timeline = defaultdict(list)  # ticker -> list of step_indices where held
    for s in steps_sorted:
        for tkr in tickers_by_step[s]:
            holdings_timeline[tkr].append(s)

    holding_periods = []  # all periods (completed + still-censored flag)
    censored_count = 0
    completed_periods = []
    last_step = steps_sorted[-1]
    for tkr, s_list in holdings_timeline.items():
        s_sorted = sorted(s_list)
        # break into contiguous runs
        run_start = s_sorted[0]
        prev = s_sorted[0]
        for s in s_sorted[1:]:
            if s != prev + 1:
                run_len = prev - run_start + 1
                if run_start == 0 and False:  # treat t=0 entries as new entries per spec
                    pass
                holding_periods.append((run_len, False))
                completed_periods.append(run_len)
                run_start = s
            prev = s
        # close final run
        run_len = prev - run_start + 1
        is_censored = (prev == last_step)  # still held at last step
        holding_periods.append((run_len, is_censored))
        if is_censored:
            censored_count += 1
        else:
            completed_periods.append(run_len)

    completed = np.array(completed_periods, dtype=float)
    print("\nHolding Period Distribution (1 step = 1 week = 5 trading days):")
    print(f"  Total holding periods identified: {len(holding_periods)}")
    print(f"  Right-censored (still held at step {last_step}): {censored_count}")
    print(f"  Completed (uncensored) periods: {len(completed)}")
    if len(completed) > 0:
        print(f"  Mean holding period (completed only): {np.mean(completed):.3f} weeks")
        print(f"  Median holding period: {np.median(completed):.3f} weeks")
        print(f"  P25 holding period: {np.percentile(completed, 25):.3f} weeks")
        print(f"  P75 holding period: {np.percentile(completed, 75):.3f} weeks")
        print(f"  Min holding period: {np.min(completed):.3f} weeks")
        print(f"  Max holding period: {np.max(completed):.3f} weeks")
        all_hp = np.array([p[0] for p in holding_periods], dtype=float)
        print(f"  Max holding period (incl. censored): {np.max(all_hp):.3f} weeks")

    # Rank correlation between consecutive steps for persistent tickers
    rhos = []
    for i in range(n_steps - 1):
        s_t = steps_sorted[i]
        s_tp1 = steps_sorted[i + 1]
        S_t = tickers_by_step[s_t]
        S_tp1 = tickers_by_step[s_tp1]
        inter = S_t & S_tp1
        if len(inter) >= 3:
            r_t = [rank_by_step[s_t][tkr] for tkr in inter]
            r_tp1 = [rank_by_step[s_tp1][tkr] for tkr in inter]
            rho, _ = spearmanr(r_t, r_tp1)
            if not math.isnan(rho):
                rhos.append(rho)
    rhos = np.array(rhos)
    print(f"\nRank Correlation (Spearman) between consecutive rebalances (persistent tickers only, n>=3):")
    print(f"  Valid transitions: {len(rhos)} of {n_steps - 1}")
    print(f"  Mean Spearman rho = {np.mean(rhos):.4f}")
    print(f"  Median rho = {np.median(rhos):.4f}")
    print(f"  Std rho = {np.std(rhos, ddof=1):.4f}")
    print(f"  % steps with rho > 0.5 = {100.0*np.mean(rhos > 0.5):.2f}%")
    print(f"  % steps with rho < 0   = {100.0*np.mean(rhos < 0):.2f}%")
    print(f"  % steps with rho > 0   = {100.0*np.mean(rhos > 0):.2f}%")

    # Churn assessment
    mto = turnover_data["membership_turnover"]
    print(f"\nChurn Assessment:")
    print(f"  % steps with membership turnover > 50%: "
          f"{100.0*np.mean(mto > 0.5):.2f}%")
    print(f"  % steps with membership turnover = 100%: "
          f"{100.0*np.mean(mto >= 1.0 - 1e-9):.2f}%")
    print(f"  Mean positions replaced per rebalance: {np.mean(turnover_data['positions_replaced']):.4f}")
    print(f"  Typical weekly momentum strategies: 20-40% turnover/rebalance")
    print(f"  Measured: mean membership turnover = {np.mean(mto)*100:.2f}% per rebalance")
    verdict = "HIGH CHURN" if np.mean(mto) > 0.4 else ("ELEVATED" if np.mean(mto) > 0.2 else "NORMAL")
    print(f"  Verdict: {verdict}")

    return {
        "persistence": persistence,
        "holding_periods": holding_periods,
        "completed_periods": completed,
        "censored_count": censored_count,
        "rhos": rhos,
    }


# ---------------------------------------------------------------------------
# PHASE 4 - Gross return reconstruction
# ---------------------------------------------------------------------------
def gross_metrics(returns):
    """Compute CAGR, Sharpe, Max DD, Calmar for a per-step return series."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return dict(cagr=float("nan"), sharpe=float("nan"), max_dd=float("nan"), calmar=float("nan"),
                    cum=float("nan"), mean=float("nan"), std=float("nan"))
    n_steps = len(r)
    cum = np.cumprod(1.0 + r)[-1]
    cagr = cum ** (STEPS_PER_YEAR / n_steps) - 1.0 if cum > 0 else float("nan")
    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    sharpe = (mean_r - RISK_FREE_PER_STEP) / std_r * math.sqrt(STEPS_PER_YEAR) if std_r > 0 else float("nan")
    # Max DD
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else float("nan")
    return dict(cagr=float(cagr), sharpe=float(sharpe), max_dd=float(max_dd),
                calmar=float(calmar), cum=float(cum), mean=mean_r, std=float(std_r))


def compute_gross_returns(long_df, mh):
    section("PHASE 4: GROSS RETURN RECONSTRUCTION")

    grouped = long_df.groupby("step_index")
    r_A1 = grouped.apply(lambda g: float((g["equal_weight"] * g["actual_return"]).sum()))
    r_F1 = grouped.apply(lambda g: float((g["inv_vol_weight"] * g["actual_return"]).sum()))
    r_F1_contrib = grouped["contribution"].sum()

    # Should match
    diff = (r_F1 - r_F1_contrib).abs().max()
    print(f"R_F1 (recomputed from inv_vol_weight * actual_return) vs sum(contribution) max abs diff: {diff:.3e}")

    mh_pr = mh.set_index("step_index")["portfolio_return"]
    common = r_F1.index.intersection(mh_pr.index)
    diff_mh = (r_F1.loc[common] - mh_pr.loc[common]).abs()
    print(f"R_F1 vs model_health.portfolio_return: max abs diff = {diff_mh.max():.3e}, "
          f"mean abs diff = {diff_mh.mean():.3e}")
    if diff_mh.max() > 1e-6:
        print(f"  [WARN] Mismatch above 1e-6. Worst step: {diff_mh.idxmax()}")
    else:
        print(f"  [OK] R_F1 matches model_health portfolio_return (max diff {diff_mh.max():.3e} <= 1e-6)")

    a1_m = gross_metrics(r_A1.values)
    f1_m = gross_metrics(r_F1.values)

    n_steps = len(r_A1)
    print(f"\nNumber of steps used: {n_steps}")
    print(f"\nMean per-step return:  A1 = {np.mean(r_A1):.8f}   F1 = {np.mean(r_F1):.8f}")
    print(f"Std  per-step return:  A1 = {np.std(r_A1, ddof=1):.8f}   F1 = {np.std(r_F1, ddof=1):.8f}")
    print(f"Sum  per-step return:  A1 = {np.sum(r_A1):.8f}   F1 = {np.sum(r_F1):.8f}")
    print(f"Total cumulative return (simple prefix product):   A1 = {a1_m['cum']:.6f}   F1 = {f1_m['cum']:.6f}")

    print(f"\nGross Metrics Table (A1 vs F1):")
    print(f"  {'Metric':<20} {'A1 (equal_weight)':<22} {'F1 (inv_vol_weight)':<22}")
    print(f"  {'CAGR':<20} {a1_m['cagr']:<22.6f} {f1_m['cagr']:<22.6f}")
    print(f"  {'Sharpe':<20} {a1_m['sharpe']:<22.6f} {f1_m['sharpe']:<22.6f}")
    print(f"  {'Max DD':<20} {a1_m['max_dd']:<22.6f} {f1_m['max_dd']:<22.6f}")
    print(f"  {'Calmar':<20} {a1_m['calmar']:<22.6f} {f1_m['calmar']:<22.6f}")

    return r_A1, r_F1, a1_m, f1_m


# ---------------------------------------------------------------------------
# PHASE 5 - Transaction cost analysis
# ---------------------------------------------------------------------------
def compute_costs(r_A1, r_F1, turnover_A1, turnover_F1, a1_gross, f1_gross):
    section("PHASE 5: TRANSACTION COST ANALYSIS")

    def net_metrics(returns, turnover, cost_rt):
        # turnover has n-1 transitions (one per t->t+1), returns has n steps.
        # Convention: cost of transition t->t+1 is charged to step t+1.
        # Step 0 carries no rebalance cost (initial setup assumed free).
        r = np.asarray(returns, dtype=float)
        n = len(r)
        nt = len(turnover)
        cost_per_step = np.zeros(n, dtype=float)
        # Align: turnover[i] is transition step_i -> step_{i+1}, charged to step_{i+1}
        # If length matches n == nt+1 (standard), this maps cleanly.
        if nt == n - 1:
            cost_per_step[1:] = turnover * cost_rt
        else:
            # Fallback: assume already aligned
            cost_per_step[:nt] = turnover[:n] * cost_rt
        net = r - cost_per_step
        m = gross_metrics(net)
        m["annual_drag"] = float(np.mean(turnover) * cost_rt * STEPS_PER_YEAR)
        m["mean_turnover"] = float(np.mean(turnover))
        m["mean_cost_per_step"] = float(np.mean(cost_per_step))
        # mean net cost per REBALANCE step (the actual cost events)
        m["mean_cost_per_rebalance"] = float(np.mean(turnover) * cost_rt)
        return m, cost_per_step

    results = {}
    for cost_label, cost_rt in COST_SCENARIOS.items():
        m_A1, c_A1 = net_metrics(r_A1.values, turnover_A1, cost_rt)
        m_A1["cagr_drag"] = a1_gross["cagr"] - m_A1["cagr"]
        m_F1, c_F1 = net_metrics(r_F1.values, turnover_F1, cost_rt)
        m_F1["cagr_drag"] = f1_gross["cagr"] - m_F1["cagr"]
        results[cost_label] = {"A1": m_A1, "F1": m_F1}

    # Print Sensitivity Tables
    def print_table(metric_key, metric_label, fmt="{:.6f}"):
        print(f"\nCost Sensitivity Table — {metric_label}")
        print(f"  {'Scenario':<18} {'A1':<22} {'F1':<22}")
        for cost_label in COST_SCENARIOS:
            a1v = results[cost_label]["A1"][metric_key]
            f1v = results[cost_label]["F1"][metric_key]
            print(f"  {cost_label:<18} {fmt.format(a1v):<22} {fmt.format(f1v):<22}")

    print_table("cagr", "Net CAGR")
    print_table("sharpe", "Net Sharpe")
    print_table("max_dd", "Net Max DD")
    print_table("calmar", "Net Calmar")
    print_table("cagr_drag", "CAGR Drag (Gross-Net)")

    print(f"\nAnnual Cost Drag Table (mean_turnover × round_trip_cost × 50.4):")
    print(f"  {'Scenario':<18} {'A1 annual drag':<22} {'F1 annual drag':<22}")
    for cost_label in COST_SCENARIOS:
        ad_a1 = results[cost_label]["A1"]["annual_drag"]
        ad_f1 = results[cost_label]["F1"]["annual_drag"]
        print(f"  {cost_label:<18} {ad_a1:<22.6f} {ad_f1:<22.6f}")

    # Break-even cost analysis
    print(f"\nBreak-even Cost Analysis (Net CAGR = 0):")
    print(f"  Use Gross mean per-step return / mean weight-based turnover:")
    print(f"  A1: gross mean return = {a1_gross['mean']:.8f}, mean turnover = {np.mean(turnover_A1):.8f}")
    print(f"  F1: gross mean return = {f1_gross['mean']:.8f}, mean turnover = {np.mean(turnover_F1):.8f}")
    be_A1_rt = a1_gross["mean"] / np.mean(turnover_A1) if np.mean(turnover_A1) > 0 else float("nan")
    be_F1_rt = f1_gross["mean"] / np.mean(turnover_F1) if np.mean(turnover_F1) > 0 else float("nan")
    print(f"  Break-even round-trip cost: A1 = {be_A1_rt:.6f}  F1 = {be_F1_rt:.6f}")
    print(f"  Break-even per side (bps):  A1 = {be_A1_rt*10000/2:.2f}  F1 = {be_F1_rt*10000/2:.2f}")
    print(f"  Break-even round-trip (bps): A1 = {be_A1_rt*10000:.2f}  F1 = {be_F1_rt*10000:.2f}")

    # A1 vs F1 post-cost comparison
    print(f"\nA1 vs F1 Post-Cost Comparison (Net CAGR + Net Sharpe by scenario):")
    print(f"  {'Scenario':<18} {'A1 Net CAGR':<14} {'F1 Net CAGR':<14} {'A1 Net Sharpe':<16} {'F1 Net Sharpe':<16} {'A1-F1 CAGR':<12}")
    for cost_label in COST_SCENARIOS:
        a1_cagr = results[cost_label]["A1"]["cagr"]
        f1_cagr = results[cost_label]["F1"]["cagr"]
        a1_sharpe = results[cost_label]["A1"]["sharpe"]
        f1_sharpe = results[cost_label]["F1"]["sharpe"]
        diff_cagr = a1_cagr - f1_cagr
        print(f"  {cost_label:<18} {a1_cagr:<14.6f} {f1_cagr:<14.6f} "
              f"{a1_sharpe:<16.6f} {f1_sharpe:<16.6f} {diff_cagr:<12.6f}")

    return results, be_A1_rt, be_F1_rt


# ---------------------------------------------------------------------------
# PHASE 6 - Adversarial assessment
# ---------------------------------------------------------------------------
def adversarial_assessment(turnover_data, a1_gross, f1_gross, r_A1, r_F1, cost_results):
    section("PHASE 6: ADVERSARIAL ASSESSMENT")

    mean_mto = float(np.mean(turnover_data["membership_turnover"]))
    mean_A1_to = float(np.mean(turnover_data["turnover_A1"]))
    mean_F1_to = float(np.mean(turnover_data["turnover_F1"]))
    pct_full = 100.0 * float(np.mean(turnover_data["membership_turnover"] >= 1.0 - 1e-9))
    pct_high_churn = 100.0 * float(np.mean(turnover_data["membership_turnover"] > 0.8))

    cagr_drag_S2_A1 = cost_results["S2 (10bps/side)"]["A1"]["cagr_drag"]
    cagr_drag_S2_F1 = cost_results["S2 (10bps/side)"]["F1"]["cagr_drag"]
    annual_drag_S2_A1 = cost_results["S2 (10bps/side)"]["A1"]["annual_drag"]
    annual_drag_S2_F1 = cost_results["S2 (10bps/side)"]["F1"]["annual_drag"]

    # Incremental drag from S2 to S4
    cagr_drag_inc_A1 = cost_results["S4 (20bps/side)"]["A1"]["cagr_drag"] - \
                       cost_results["S2 (10bps/side)"]["A1"]["cagr_drag"]
    cagr_drag_inc_F1 = cost_results["S4 (20bps/side)"]["F1"]["cagr_drag"] - \
                       cost_results["S2 (10bps/side)"]["F1"]["cagr_drag"]

    rows = []

    # Claim 1: "100% turnover per rebalance"
    verdict = "DISPROVEN" if mean_mto < 0.95 else ("CONFIRMED" if mean_mto >= 0.999 else "PARTIALLY")
    rows.append({
        "claim": "100% turnover per rebalance",
        "measured": f"mean membership turnover = {mean_mto*100:.2f}%/rebalance\n"
                    f"mean A1 weight turnover = {mean_A1_to*100:.2f}%\n"
                    f"mean F1 weight turnover = {mean_F1_to*100:.2f}%\n"
                    f"% steps with 100% turnover = {pct_full:.2f}%\n"
                    f"% steps with turnover > 80% = {pct_high_churn:.2f}%",
        "claimed": "100% per rebalance",
        "verdict": verdict,
    })

    # Claim 2: "+10-20% CAGR from missing costs"
    verdict2 = "PARTIALLY CONFIRMED" if (0.10 <= cagr_drag_S2_F1 <= 0.20) else (
        "CONFIRMED" if cagr_drag_S2_F1 > 0.20 else "DISPROVEN (too small)")
    rows.append({
        "claim": "+10-20% CAGR from missing costs (at 10bps/side, S2)",
        "measured": f"F1 CAGR drag @ S2 = {cagr_drag_S2_F1*100:.2f}%\n"
                    f"A1 CAGR drag @ S2 = {cagr_drag_S2_A1*100:.2f}%",
        "claimed": "+10% to +20% CAGR",
        "verdict": verdict2,
    })

    # Claim 3: "+3-5% from missing slippage (S2→S4 incremental)"
    verdict3 = "CONFIRMED" if (0.03 <= cagr_drag_inc_F1 <= 0.05) else (
        "PARTIALLY" if 0.02 <= cagr_drag_inc_F1 <= 0.06 else "DISPROVEN")
    rows.append({
        "claim": "+3-5% CAGR from missing slippage (incremental S2→S4)",
        "measured": f"F1 incremental drag (S2→S4) = {cagr_drag_inc_F1*100:.2f}%\n"
                    f"A1 incremental drag (S2→S4) = {cagr_drag_inc_A1*100:.2f}%",
        "claimed": "+3% to +5% CAGR",
        "verdict": verdict3,
    })

    # Claim 4: "~10% annual drag at 10bps x 50 rebalances"
    verdict4 = "CONFIRMED" if (0.08 <= annual_drag_S2_F1 <= 0.12) else (
        "PARTIALLY" if 0.05 <= annual_drag_S2_F1 <= 0.15 else "DISPROVEN")
    rows.append({
        "claim": "~10% annual drag at 10bps × 50 rebalances",
        "measured": f"F1 annual drag @ S2 = {annual_drag_S2_F1*100:.2f}%\n"
                    f"A1 annual drag @ S2 = {annual_drag_S2_A1*100:.2f}%\n"
                    f"F1 mean turnover/rebalance = {mean_F1_to*100:.2f}%\n"
                    f"Annualized turnover = {mean_F1_to*50.4*100:.2f}%",
        "claimed": "~10% annual drag",
        "verdict": verdict4,
    })

    for r in rows:
        print(f"\n  CLAIM: {r['claim']}")
        print(f"  Claimed value:\n    {r['claimed']}")
        print(f"  Measured value:\n    {r['measured']}")
        print(f"  VERDICT: {r['verdict']}")

    return rows


# ---------------------------------------------------------------------------
# PHASE 7 - Save results CSV
# ---------------------------------------------------------------------------
def save_results_csv(turnover_summary, a1_gross, f1_gross, cost_results, adv_rows):
    rows = []
    # Top-level summary
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "membership_turnover_mean", "A1": turnover_summary["membership"]["mean"],
                 "F1": turnover_summary["membership"]["mean"], "notes": "same selection for A1 & F1"})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "membership_turnover_median", "A1": turnover_summary["membership"]["median"],
                 "F1": turnover_summary["membership"]["median"], "notes": ""})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "weight_turnover_mean", "A1": turnover_summary["A1"]["mean"],
                 "F1": turnover_summary["F1"]["mean"], "notes": ""})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "weight_turnover_median", "A1": turnover_summary["A1"]["median"],
                 "F1": turnover_summary["F1"]["median"], "notes": ""})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "weight_turnover_std", "A1": turnover_summary["A1"]["std"],
                 "F1": turnover_summary["F1"]["std"], "notes": ""})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "annualized_weight_turnover", "A1": turnover_summary["A1"]["annualized"],
                 "F1": turnover_summary["F1"]["annualized"], "notes": "mean × 50.4"})
    rows.append({"section": "TURNOVER_SUMMARY",
                 "metric": "mean_positions_replaced", "A1": turnover_summary["positions_replaced"]["mean"],
                 "F1": turnover_summary["positions_replaced"]["mean"], "notes": ""})

    # Gross metrics
    for k in ["cagr", "sharpe", "max_dd", "calmar"]:
        rows.append({"section": "GROSS_METRICS", "metric": k, "A1": a1_gross[k], "F1": f1_gross[k],
                     "notes": ""})

    # Cost scenarios
    for cost_label, rd in cost_results.items():
        for metric_key in ["cagr", "sharpe", "max_dd", "calmar", "cagr_drag", "annual_drag",
                            "mean_cost_per_step"]:
            rows.append({"section": f"COST_{cost_label}", "metric": metric_key,
                         "A1": rd["A1"][metric_key], "F1": rd["F1"][metric_key], "notes": ""})

    # Adversarial
    for r in adv_rows:
        rows.append({"section": "ADVERSARIAL", "metric": r["claim"],
                     "A1": "", "F1": "", "notes": f"{r['measured']} || VERDICT: {r['verdict']}"})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results CSV: {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    banner("GreekFire Turnover / Transaction-Cost / Slippage / Ranking-Stability Audit")

    long_df, mh, issues = load_and_validate()
    turnover_data = compute_turnover(long_df)
    ranking_data = compute_ranking_stability(long_df, turnover_data)
    r_A1, r_F1, a1_gross, f1_gross = compute_gross_returns(long_df, mh)
    cost_results, be_A1, be_F1 = compute_costs(r_A1, r_F1,
                                                turnover_data["turnover_A1"],
                                                turnover_data["turnover_F1"],
                                                a1_gross, f1_gross)
    adv_rows = adversarial_assessment(turnover_data, a1_gross, f1_gross, r_A1, r_F1, cost_results)

    save_results_csv(turnover_data["summary"], a1_gross, f1_gross, cost_results, adv_rows)

    # Summary recap
    section("FINAL RECAP")
    print(f"Mean membership turnover: {turnover_data['summary']['membership']['mean']*100:.2f}%/rebalance")
    print(f"Mean A1 weight turnover:  {turnover_data['summary']['A1']['mean']*100:.2f}%/rebalance")
    print(f"Mean F1 weight turnover:  {turnover_data['summary']['F1']['mean']*100:.2f}%/rebalance")
    print(f"Annualized turnover F1:   {turnover_data['summary']['F1']['annualized']*100:.2f}%")
    print(f"Gross CAGR:               A1={a1_gross['cagr']*100:.2f}%   F1={f1_gross['cagr']*100:.2f}%")
    print(f"Gross Sharpe:             A1={a1_gross['sharpe']:.4f}   F1={f1_gross['sharpe']:.4f}")
    print(f"Gross Max DD:             A1={a1_gross['max_dd']*100:.2f}%   F1={f1_gross['max_dd']*100:.2f}%")
    print(f"Break-even round-trip bps: A1={be_A1*10000:.2f}   F1={be_F1*10000:.2f}")
    print(f"Break-even per-side bps:   A1={be_A1*10000/2:.2f}   F1={be_F1*10000/2:.2f}")
    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()