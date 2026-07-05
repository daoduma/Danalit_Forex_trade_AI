"""Backtest performance metrics: the numbers a go/no-go decision is made on."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def max_drawdown(series: pd.Series) -> float:
    """Max peak-to-trough drawdown as a fraction of the peak (0..1)."""
    if series.empty:
        return 0.0
    running_max = series.cummax()
    dd = (series - running_max) / running_max.replace(0, np.nan)
    return float(-dd.min()) if dd.notna().any() else 0.0


def consecutive_losses(pnls: list[float]) -> int:
    worst = cur = 0
    for p in pnls:
        cur = cur + 1 if p < 0 else 0
        worst = max(worst, cur)
    return worst


def summarize(
    trades: list,
    equity_curve: pd.DataFrame,
    initial_balance: float,
    bar_minutes: int = 15,
) -> dict:
    """Full metric set from an engine run."""
    out: dict = {"n_trades": len(trades), "initial_balance": initial_balance}
    if equity_curve.empty:
        return out
    eq = equity_curve["equity"]
    bal = equity_curve["balance"]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)

    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)

    out.update({
        "final_equity": float(eq.iloc[-1]),
        "net_profit": float(eq.iloc[-1] - initial_balance),
        "return_pct": float(eq.iloc[-1] / initial_balance - 1),
        "cagr": float((eq.iloc[-1] / initial_balance) ** (1 / years) - 1)
        if eq.iloc[-1] > 0 else -1.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy": float(np.mean(pnls)) if pnls else None,
        "win_rate": float(len(wins) / len(pnls)) if pnls else None,
        "max_drawdown_equity": max_drawdown(eq),
        "max_drawdown_balance": max_drawdown(bal),
        "consecutive_losses": consecutive_losses(pnls),
        "total_commission": float(sum(t.commission for t in trades)),
        "total_swap": float(sum(t.swap for t in trades)),
        "total_spread_cost": float(sum(getattr(t, "spread_cost", 0.0) for t in trades)),
        "gross_pnl": float(sum(t.gross_pnl for t in trades)),
    })

    # bar-return risk stats
    rets = eq.pct_change().dropna()
    if len(rets) > 2 and rets.std() > 0:
        bars_per_year = len(rets) / years
        ann = np.sqrt(bars_per_year)
        out["sharpe"] = float(rets.mean() / rets.std() * ann)
        downside = rets[rets < 0]
        out["sortino"] = float(rets.mean() / downside.std() * ann) if len(downside) > 2 and downside.std() > 0 else None
    else:
        out["sharpe"] = out["sortino"] = None

    if trades:
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        out["avg_duration_h"] = float(np.mean(durations))
        out["median_duration_h"] = float(np.median(durations))
        df = pd.DataFrame({
            "instrument": [t.instrument for t in trades],
            "year": [t.exit_time.year for t in trades],
            "net": pnls,
        })
        out["by_instrument"] = df.groupby("instrument")["net"].agg(["count", "sum"]).round(4).to_dict("index")
        out["by_year"] = df.groupby("year")["net"].agg(["count", "sum"]).round(4).to_dict("index")
    return out


def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Month-end equity returns table (year rows x month columns, %)."""
    if equity_curve.empty:
        return pd.DataFrame()
    eq = equity_curve["equity"].resample("ME").last()
    rets = eq.pct_change().dropna() * 100
    if rets.empty:
        return pd.DataFrame()
    return (
        pd.DataFrame({"year": rets.index.year, "month": rets.index.month, "ret": rets.values})
        .pivot(index="year", columns="month", values="ret")
        .round(2)
    )


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam rational approximation — good to ~1e-9, no scipy dependency
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe_ratio(
    sharpe_annual: Optional[float],
    n_trials: int,
    n_obs: int,
    bars_per_year: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> Optional[float]:
    """P(true Sharpe > 0) after deflating for multiple testing (n_trials)."""
    if sharpe_annual is None or n_obs < 30 or n_trials < 1:
        return None
    sr = sharpe_annual / np.sqrt(bars_per_year)  # per-bar Sharpe
    # expected max Sharpe of n_trials pure-noise strategies (per bar)
    emc = 0.5772156649
    if n_trials > 1:
        z1 = _norm_ppf(1 - 1.0 / n_trials)
        z2 = _norm_ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(1.0 / max(n_obs - 1, 1)) * ((1 - emc) * z1 + emc * z2)
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12) / max(n_obs - 1, 1))
    return float(_norm_cdf((sr - sr0) / denom))


def bootstrap_drawdowns(
    trade_pnls: list[float],
    initial_balance: float,
    n_paths: int = 1000,
    seed: int = 7,
) -> dict:
    """Bootstrap-resample the trade sequence -> drawdown distribution + P(ruin)."""
    if not trade_pnls:
        return {}
    rng = np.random.default_rng(seed)
    pnls = np.asarray(trade_pnls)
    dds, ruins = [], 0
    for _ in range(n_paths):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        eq = initial_balance + np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate([[initial_balance], eq]))[1:]
        dd = np.max((peak - eq) / peak) if len(eq) else 0.0
        dds.append(dd)
        if np.min(eq) <= initial_balance * 0.5:  # 'ruin' = losing half the account
            ruins += 1
    dds = np.sort(dds)
    return {
        "dd_p50": float(np.percentile(dds, 50)),
        "dd_p95": float(np.percentile(dds, 95)),
        "dd_p99": float(np.percentile(dds, 99)),
        "p_ruin_half": ruins / n_paths,
        "n_paths": n_paths,
    }
