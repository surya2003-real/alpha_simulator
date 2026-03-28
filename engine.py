"""
engine.py
Vectorized backtest engine for the Alpha Simulator.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from data_loader import generate_universe, get_benchmark_returns

import math

def _sanitize(val):
    """Recursively replaces NaN and Infinity with 0.0 to prevent JSON crashes."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize(v) for v in val]
    return val

# ─────────────────────────────────────────────
# Config (Updated for Technical Factors)
# ─────────────────────────────────────────────

FACTOR_COLS = {
    "momentum_1m":    "1M Momentum",
    "momentum_6m":    "6M Momentum",
    "momentum_12m":   "12M Momentum",
    "rsi_14":         "RSI-14",
    "volatility_20d": "20D Volatility",
}

HIGHER_IS_BETTER = {
    "momentum_1m":    True,
    "momentum_6m":    True,
    "momentum_12m":   True,
    "rsi_14":         True,
    "volatility_20d": False, # Typically, lower volatility is preferred for defensive factors
}


@dataclass
class FactorConfig:
    name: str              
    weight: float              
    higher_is_better: bool = True
    winsorize: bool = True
    zscore: bool = True


@dataclass
class BacktestConfig:
    start_date: str = "2004-01-01"
    end_date:   str = "2018-12-31"
    
    custom_alpha: Optional[str] = None
    factors: List[FactorConfig] = field(default_factory=list)

    sectors: Optional[List[str]] = None
    market_caps: Optional[List[str]] = None
    min_adv: float = 0.0

    max_positions: int = 50
    max_weight: float = 0.05
    weighting: str = "equal"        
    long_short: bool = False
    short_fraction: float = 0.2

    rebalance_freq: str = "M"       

    slippage_bps: float = 10.0
    commission_bps: float = 5.0
    max_drawdown_stop: float = 0.20
    benchmark: str = "SPY"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0:
        return s * 0
    return (s - s.mean()) / std


def _compute_composite_score(
    snapshot: pd.DataFrame,
    factors: List[FactorConfig],
    custom_alpha: Optional[str] = None
) -> pd.Series:
    
    if custom_alpha:
        clean_snap = snapshot.copy()
        for col in clean_snap.columns:
            if clean_snap[col].dtype in [np.float64, np.int64]:
                clean_snap[col] = clean_snap[col].fillna(clean_snap[col].median())
        try:
            if custom_alpha and "returns" in custom_alpha and "lag" not in custom_alpha:
                raise ValueError("Use lagged returns (returns_lag1) to avoid lookahead bias.")
            return clean_snap.eval(custom_alpha)
        except Exception as e:
            raise ValueError(f"Failed to evaluate custom alpha '{custom_alpha}': {str(e)}")

    scores = pd.DataFrame(index=snapshot.index)
    total_w = sum(f.weight for f in factors)

    for fc in factors:
        col = fc.name
        if col not in snapshot.columns:
            continue
        s = snapshot[col].copy()
        if s.isna().all():
            continue
        s = s.fillna(s.median())
        if fc.winsorize:
            s = _winsorize(s)
        if fc.zscore:
            s = _zscore(s)
        if not fc.higher_is_better:
            s = -s
        scores[col] = s * (fc.weight / total_w)

    return scores.sum(axis=1)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def _safe_sharpe(rets: pd.Series) -> float:
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))


def _sortino(rets: pd.Series, mar: float = 0.0) -> float:
    downside = rets[rets < mar]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    dd_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    ann_ret = (1 + rets.mean()) ** 252 - 1
    return float(ann_ret / dd_std)


def _max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    mdd = drawdown.min()
    end_idx = drawdown.idxmin()
    peak_idx = equity[:end_idx].idxmax()
    recovery = equity[end_idx:]
    rec_idx = recovery[recovery >= equity[peak_idx]].index
    rec_months = len(rec_idx) // 21 if len(rec_idx) > 0 else None
    return float(mdd), drawdown, rec_months


def _var_cvar(rets: pd.Series, conf: float = 0.95):
    var = float(rets.quantile(1 - conf))
    cvar = float(rets[rets <= var].mean())
    return var, cvar


def _calmar(rets: pd.Series, equity: pd.Series) -> float:
    cagr = (1 + rets.mean()) ** 252 - 1
    mdd, _, _ = _max_drawdown(equity)
    return float(cagr / abs(mdd)) if mdd != 0 else 0.0


def _omega(rets: pd.Series, mar: float = 0.0) -> float:
    gains = (rets[rets > mar] - mar).sum()
    losses = (mar - rets[rets < mar]).sum()
    return float(gains / losses) if losses > 0 else np.inf


def _information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    active = strat - bench
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(252))


def _treynor(rets: pd.Series, bench: pd.Series, rf: float = 0.02) -> float:
    beta = np.cov(rets, bench)[0, 1] / np.var(bench)
    ann = (1 + rets.mean()) ** 252 - 1
    return float((ann - rf) / beta) if beta != 0 else 0.0


def _rolling_sharpe(rets: pd.Series, window: int = 252) -> pd.Series:
    roll_mean = rets.rolling(window).mean()
    roll_std  = rets.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).fillna(0)


def _annual_returns(rets: pd.Series) -> Dict[int, float]:
    return (
        (1 + rets).groupby(rets.index.year).prod() - 1
    ).to_dict()


def _monthly_returns(rets: pd.Series) -> Dict[str, float]:
    monthly = (1 + rets).resample("M").prod() - 1
    return {str(d.date()): round(float(v), 6) for d, v in monthly.items()}


def _factor_exposures(strat_rets: pd.Series, universe_df: pd.DataFrame) -> Dict[str, List]:
    dates = strat_rets.index
    bench = get_benchmark_returns(dates)
    bench = bench.reindex(dates).fillna(0)
    window = 252
    betas = []
    roll_dates = []
    for i in range(window, len(dates), 21):
        s = strat_rets.iloc[i-window:i]
        b = bench.iloc[i-window:i]
        if b.std() > 0:
            beta = np.cov(s, b)[0, 1] / np.var(b)
        else:
            beta = 1.0
        betas.append(round(float(beta), 3))
        roll_dates.append(str(dates[i].date()))
    return {"dates": roll_dates, "market_beta": betas}


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self._universe: Optional[pd.DataFrame] = None
        self._dates: Optional[pd.DatetimeIndex] = None
        self._rebal_dates: Optional[pd.DatetimeIndex] = None

    def _load_data(self):
        df = generate_universe()
        df = df[
            (df["date"] >= pd.Timestamp(self.cfg.start_date)) &
            (df["date"] <= pd.Timestamp(self.cfg.end_date))
        ]
        
        # --- DISABLE THESE FOR YFINANCE DATA ---
        # if self.cfg.sectors:
        #     df = df[df["sector"].isin(self.cfg.sectors)]
        # if self.cfg.market_caps:
        #     df = df[df["market_cap_tier"].isin(self.cfg.market_caps)]
        
        if self.cfg.min_adv > 0:
            df = df[df["adv"] >= self.cfg.min_adv]
            
        self._universe = df
        self._dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        freq_map = {"M": "BME", "W": "W-FRI", "Q": "BQE"}
        freq = freq_map.get(self.cfg.rebalance_freq, "BME")
        start = pd.Timestamp(self.cfg.start_date)
        end   = pd.Timestamp(self.cfg.end_date)
        return pd.bdate_range(start, end, freq=freq)

    def _build_portfolio(self, rebal_date: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
        snap = self._universe[self._universe["date"] == rebal_date].set_index("ticker")
        if snap.empty:
            prior = self._universe[self._universe["date"] <= rebal_date]
            if prior.empty:
                return pd.Series(dtype=float), pd.Series(dtype=float)
            rebal_date = prior["date"].max()
            snap = self._universe[self._universe["date"] == rebal_date].set_index("ticker")

        snap = snap.dropna(subset=["returns_lag1"])
        scores = _compute_composite_score(snap, self.cfg.factors, self.cfg.custom_alpha)
        snap["score"] = scores

        n_long = self.cfg.max_positions
        n_short = int(n_long * self.cfg.short_fraction) if self.cfg.long_short else 0

        long_stocks = snap.nlargest(n_long, "score")
        weights = pd.Series(0.0, index=snap.index)

        if self.cfg.weighting == "equal":
            w = pd.Series(1.0 / n_long, index=long_stocks.index)
        elif self.cfg.weighting == "signal":
            raw = long_stocks["score"].clip(lower=0)
            w = raw / raw.sum() if raw.sum() > 0 else pd.Series(1/n_long, index=long_stocks.index)
        else:  
            w = pd.Series(1.0 / n_long, index=long_stocks.index)

        w = w.clip(upper=self.cfg.max_weight)
        w = w / w.sum()
        weights[long_stocks.index] = w

        if self.cfg.long_short and n_short > 0:
            short_stocks = snap.nsmallest(n_short, "score")
            sw = pd.Series(-self.cfg.short_fraction / n_short, index=short_stocks.index)
            weights[short_stocks.index] = sw

        return weights[weights != 0], snap["score"]

    def run(self) -> Dict[str, Any]:
        self._load_data()
        rebal_dates = self._get_rebalance_dates()

        all_dates = self._dates.values
        portfolio_value = 1.0
        equity_curve: Dict[str, float] = {}
        current_weights: pd.Series = pd.Series(dtype=float)
        trade_log: List[Dict] = []
        turnover_log: List[float] = []

        price_pivot = self._universe.pivot(index="date", columns="ticker", values="price")
        price_pivot = price_pivot.sort_index()

        stopped = False
        peak = 1.0
        first_rebal = True

        for date in pd.DatetimeIndex(all_dates):
            if not stopped and (date in rebal_dates or first_rebal):
                first_rebal = False
                
                # 1. Unpack the new tuple returned by _build_portfolio
                prev_idx = np.searchsorted(self._dates, date) - 1
                if prev_idx < 0:
                    continue

                signal_date = self._dates.iloc[prev_idx]
                new_weights, current_scores = self._build_portfolio(signal_date)

                if not new_weights.empty:
                    all_tickers = current_weights.index.union(new_weights.index)
                    old_w = current_weights.reindex(all_tickers).fillna(0)
                    new_w = new_weights.reindex(all_tickers).fillna(0)
                    turnover = (new_w - old_w).abs().sum() / 2
                    turnover_log.append(float(turnover))

                    cost_bps = (self.cfg.slippage_bps + self.cfg.commission_bps)
                    cost = turnover * cost_bps / 10000
                    portfolio_value *= (1 - cost)

                    buys  = new_w[new_w > old_w + 0.001].index
                    sells = new_w[new_w < old_w - 0.001].index
                    
                    for t in buys:
                        price_row = price_pivot.loc[price_pivot.index < date].iloc[-1]
                        px = float(price_row.get(t, 0))
                        trade_log.append({
                            "date": str(date.date()),
                            "ticker": t,
                            "action": "BUY",
                            "weight_new": round(float(new_w[t]), 4),
                            "weight_old": round(float(old_w.get(t, 0)), 4),
                            "price": round(px, 2),
                            # 2. Use the fast O(1) lookup on current_scores instead of querying the main dataframe
                            "score": round(float(current_scores.get(t, 0.0)), 4),
                        })
                        
                    for t in sells:
                        price_row = price_pivot.loc[price_pivot.index < date].iloc[-1]
                        px = float(price_row.get(t, 0))
                        trade_log.append({
                            "date": str(date.date()),
                            "ticker": t,
                            "action": "SELL",
                            "weight_new": round(float(new_w[t]), 4),
                            "weight_old": round(float(old_w.get(t, 0)), 4),
                            "price": round(px, 2),
                            "score": 0.0,
                        })

                    current_weights = new_weights

            if not current_weights.empty and not stopped:
                prev_date_idx = np.searchsorted(price_pivot.index, date)
                if prev_date_idx > 0:
                    prev_date = price_pivot.index[prev_date_idx - 1]
                    prev_prices = price_pivot.loc[prev_date]
                    curr_prices = price_pivot.loc[date] if date in price_pivot.index else prev_prices

                    held = current_weights.index.intersection(prev_prices.dropna().index)
                    if len(held) > 0:
                        p0 = prev_prices[held].fillna(method="ffill")
                        p1 = curr_prices[held].fillna(method="ffill")
                        daily_rets = (p1 - p0) / p0.replace(0, np.nan)
                        daily_rets = daily_rets.fillna(0)
                        w = current_weights[held]
                        w = w / w.abs().sum()
                        portfolio_ret = (daily_rets * w).sum()
                        portfolio_value *= (1 + portfolio_ret)

            if portfolio_value > peak:
                peak = portfolio_value
            dd = (portfolio_value - peak) / peak
            if dd < -self.cfg.max_drawdown_stop:
                stopped = True
                current_weights = pd.Series(dtype=float)

            equity_curve[str(date.date())] = round(float(portfolio_value), 6)

        return self._compute_results(equity_curve, trade_log, turnover_log)

    def _compute_results(
        self,
        equity_curve: Dict[str, float],
        trade_log: List[Dict],
        turnover_log: List[float],
    ) -> Dict[str, Any]:
        eq = pd.Series(equity_curve)
        eq.index = pd.to_datetime(eq.index)
        eq = eq.sort_index()

        rets = eq.pct_change().dropna()

        bench_rets = get_benchmark_returns(eq.index)
        bench_rets = bench_rets.reindex(eq.index).fillna(0)
        bench_eq   = (1 + bench_rets).cumprod()

        mdd, dd_series, rec_months = _max_drawdown(eq)
        var95, cvar95 = _var_cvar(rets, 0.95)
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (252 / len(rets)) - 1)
        ann_vol = float(rets.std() * np.sqrt(252))
        
        bench_total = float((bench_eq.iloc[-1] / bench_eq.iloc[0]) ** (252 / len(bench_rets)) - 1) if len(bench_rets) > 0 else 0
        alpha = cagr - bench_total
        
        beta_val = 1.0
        if np.var(bench_rets.reindex(rets.index).fillna(0)) > 0:
            beta_val = float(np.cov(rets, bench_rets.reindex(rets.index).fillna(0))[0, 1] /
                         np.var(bench_rets.reindex(rets.index).fillna(0)))

        active_share = float(np.clip(abs(alpha) / max(abs(cagr), 0.01), 0, 1))

        metrics = {
            "cagr":              round(cagr, 4),
            "total_return":      round(float(eq.iloc[-1] - 1), 4),
            "ann_vol":           round(ann_vol, 4),
            "sharpe":            round(_safe_sharpe(rets), 3),
            "sortino":           round(_sortino(rets), 3),
            "calmar":            round(_calmar(rets, eq), 3),
            "omega":             round(_omega(rets), 3),
            "information_ratio": round(_information_ratio(rets, bench_rets.reindex(rets.index).fillna(0)), 3),
            "treynor":           round(_treynor(rets, bench_rets.reindex(rets.index).fillna(0)), 4),
            "max_drawdown":      round(mdd, 4),
            "recovery_months":   rec_months,
            "var_95":            round(var95, 4),
            "cvar_95":           round(cvar95, 4),
            "alpha":             round(alpha, 4),
            "beta":              round(beta_val, 3),
            "active_share":      round(active_share, 3),
            "avg_turnover":      round(float(np.mean(turnover_log)) if turnover_log else 0, 3),
        }

        ann = _annual_returns(rets)
        if ann:
            metrics["best_year"]  = max(ann.values())
            metrics["worst_year"] = min(ann.values())

        rolling_sh = _rolling_sharpe(rets)

        raw_results = {
            "metrics":          metrics,
            "equity_curve":     {str(k.date()): round(float(v), 6) for k, v in eq.items()},
            "benchmark_curve":  {str(k.date()): round(float(v), 6) for k, v in bench_eq.items()},
            "drawdown_series":  {str(k.date()): round(float(v), 6) for k, v in dd_series.items()},
            "rolling_sharpe":   {str(k.date()): round(float(v), 3) for k, v in rolling_sh.items()},
            "annual_returns":   {str(k): round(float(v), 4) for k, v in ann.items()},
            "monthly_returns":  _monthly_returns(rets),
            "factor_exposure":  _factor_exposures(rets, self._universe),
            "trade_log":        trade_log[-500:],  
            "n_trades":         len(trade_log),
            "factor_cols":      FACTOR_COLS,
        }
        
        return _sanitize(raw_results)