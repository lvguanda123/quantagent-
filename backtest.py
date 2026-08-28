"""
Simple signal-driven backtest for the academic (高校版) variant.

Given a K-line DataFrame and a signal mode, walks the bars, opens a
position when the signal fires, and force-closes after ``hold_bars``
candles. No transaction costs, no leverage, no partial fills — this is
intentionally the simplest possible v1 so users can sanity-check that
the platform wires an end-to-end analysis → backtest loop.

Signal modes
------------
* ``decision``     : fast in-process strategy — short SMA crosses long SMA
                     ⇒ long; the inverse ⇒ short; otherwise flat. This
                     stands in for "follow the AI decision" without
                     paying an LLM round-trip per bar.
* ``always_long``  : buy on the first bar, hold for ``hold_bars``, repeat.
* ``always_short`` : mirror of ``always_long``.

Returns
-------
(metrics, equity_chart_b64) : tuple
    ``metrics`` is a dict with total_return, annual_return, max_drawdown,
    win_rate, trade_count. ``equity_chart_b64`` is a base64-encoded PNG
    of the equity curve, ready to embed in the web UI / PDF.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # Reuse the project's font defaults so the chart matches the rest of
    # the UI. matplotlib's CJK fallback to DejaVu Sans is acceptable
    # here; Chinese labels are minimal.
    plt.rcParams["axes.unicode_minus"] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


def _close_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D close-price Series, handling common column names."""
    if "close" in df.columns:
        return df["close"].astype(float)
    # Fall back to any case-variant of close
    for col in df.columns:
        if str(col).lower() == "close":
            return df[col].astype(float)
    raise ValueError("K线数据缺少 close 列")


def _generate_signals(
    df: pd.DataFrame,
    signal_mode: str,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.Series:
    """Produce a per-bar signal series.

    Values:
      +1 = open long
      -1 = open short
       0 = flat
    """
    close = _close_series(df)
    idx = close.index
    if signal_mode == "always_long":
        # Open long on every bar, but treat consecutive longs as one trade
        # by emitting 0 between positions. The runner will dedupe.
        return pd.Series([1] + [0] * (len(close) - 1), index=idx)
    if signal_mode == "always_short":
        return pd.Series([-1] + [0] * (len(close) - 1), index=idx)

    # default: SMA cross
    short = close.rolling(window=short_window, min_periods=1).mean()
    long_ = close.rolling(window=long_window, min_periods=1).mean()
    sig = (short > long_).astype(int) - (short < long_).astype(int)
    # Hold signal until reverse — emit 0 in between so the runner
    # dedupes to a single open at the crossover.
    sig = sig.ffill().fillna(0)
    # The first row is always 0 (no prior signal); force a +1/-1 only on
    # actual cross events:
    cross = sig.diff().fillna(sig)
    # cross will be +2 on long entry, -2 on short entry, 0 otherwise
    cross = cross.clip(-1, 1)
    return cross.astype(int)


def _run_trades(
    close: pd.Series,
    signals: pd.Series,
    hold_bars: int,
    initial_capital: float,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Walk the bars, open and close positions, return trades + equity."""
    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = [initial_capital]
    in_pos = 0  # +1 long / -1 short / 0 flat
    entry_price: Optional[float] = None
    entry_bar = -1

    for i, (idx, px) in enumerate(close.items()):
        sig = int(signals.iloc[i]) if i < len(signals) else 0

        # If we're in a position, check whether to close (hold elapsed)
        if in_pos != 0 and entry_price is not None and (i - entry_bar) >= hold_bars:
            if in_pos == 1:
                pnl = (px - entry_price) / entry_price
            else:
                pnl = (entry_price - px) / entry_price
            trades.append({
                "entry": entry_price,
                "exit": px,
                "side": "long" if in_pos == 1 else "short",
                "pnl": pnl,
            })
            in_pos = 0
            entry_price = None

        # If a new signal arrives and we're flat, open
        if in_pos == 0 and sig != 0:
            in_pos = sig
            entry_price = px
            entry_bar = i

        # Update mark-to-market equity
        if in_pos == 1 and entry_price is not None:
            equity = equity_curve[-1] * (1 + (px - entry_price) / entry_price * (1 / max(1, i - entry_bar + 1)))
            # Use simpler compounded equity instead:
            equity = initial_capital * (1 + (px - entry_price) / entry_price)
        elif in_pos == -1 and entry_price is not None:
            equity = initial_capital * (1 + (entry_price - px) / entry_price)
        else:
            equity = equity_curve[-1]
        equity_curve.append(equity)

    # Force-close any open position at the last bar
    if in_pos != 0 and entry_price is not None:
        last_px = float(close.iloc[-1])
        if in_pos == 1:
            pnl = (last_px - entry_price) / entry_price
        else:
            pnl = (entry_price - last_px) / entry_price
        trades.append({
            "entry": entry_price,
            "exit": last_px,
            "side": "long" if in_pos == 1 else "short",
            "pnl": pnl,
        })
        equity_curve[-1] = initial_capital * (1 + pnl)

    return trades, equity_curve


def _compute_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: List[float],
    n_bars: int,
    initial_capital: float,
) -> Dict[str, Any]:
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital if initial_capital else 0.0

    # Annualised return — assume 252 daily bars or scale linearly by bars
    if n_bars > 1:
        annual_return = (1 + total_return) ** (252 / max(n_bars, 1)) - 1
    else:
        annual_return = 0.0

    # Max drawdown
    peak = equity_curve[0] if equity_curve else initial_capital
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    # Win rate
    if trades:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "trade_count": len(trades),
    }


def _render_equity_png(equity_curve: List[float], dates: Optional[pd.DatetimeIndex]) -> str:
    """Render the equity curve as a base64-encoded PNG."""
    if not MATPLOTLIB_AVAILABLE:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=110)
    try:
        x = dates if dates is not None and len(dates) == len(equity_curve) else range(len(equity_curve))
        ax.plot(x, equity_curve, color="#C8A044", linewidth=1.8, label="Equity")
        ax.fill_between(x, equity_curve, alpha=0.12, color="#C8A044")
        ax.set_title("Backtest Equity Curve", fontsize=11)
        ax.set_xlabel("bar" if dates is None else "")
        ax.set_ylabel("Equity (CNY)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        plt.close(fig)
        return ""


def run_simple_backtest(
    df: pd.DataFrame,
    *,
    signal_mode: str = "decision",
    hold_bars: int = 5,
    initial_capital: float = 100000.0,
) -> Tuple[Dict[str, Any], str]:
    """Run the backtest and return ``(metrics, equity_chart_b64)``.

    Parameters
    ----------
    df : pd.DataFrame
        K-line data. Must contain a ``close`` column.
    signal_mode : str
        One of ``"decision"``, ``"always_long"``, ``"always_short"``.
    hold_bars : int
        How many bars to hold each open position before force-closing.
    initial_capital : float
        Starting equity in account currency.
    """
    if df is None or len(df) == 0:
        raise ValueError("回测数据为空")
    close = _close_series(df)
    if len(close) < 2:
        raise ValueError("回测数据少于 2 根 K 线，无法生成信号")

    signals = _generate_signals(df, signal_mode)
    trades, equity_curve = _run_trades(close, signals, hold_bars, initial_capital)
    metrics = _compute_metrics(trades, equity_curve, len(close), initial_capital)

    dates = None
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
    except Exception:
        dates = None

    equity_b64 = _render_equity_png(equity_curve, dates)
    return metrics, equity_b64
