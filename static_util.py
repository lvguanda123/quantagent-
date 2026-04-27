import base64
import io
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

import color_style as color
from graph_util import (
    fit_trendlines_high_low,
    fit_trendlines_single,
    get_line_points,
    split_line_into_segments,
    _make_record_dir,
)

matplotlib.use("Agg")


def generate_kline_image(kline_data, stock_name: str = "default", timeframe: str = "1d", date_range: str = "", run_ts: str = None) -> dict:
    """
    Generate a candlestick (K-line) chart from OHLCV data, save it locally, and return a base64-encoded image.

    Args:
        kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
        stock_name (str): Stock symbol for organizing output directory.
        timeframe (str): Timeframe of the data (e.g., '1d', '1h', '15m').
        date_range (str): Date range string (e.g., '2024-01-01 ~ 2024-04-01').

    Returns:
        dict: Dictionary containing base64-encoded image string and local file path.
    """

    df = pd.DataFrame(kline_data)
    # take recent 40
    df = df.tail(40)

    try:
        df.index = pd.to_datetime(df["Datetime"])
    except (ValueError, KeyError):
        try:
            df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            print("ValueError: could not parse Datetime column in static_util.py\n")
            return {"pattern_image": "", "pattern_image_description": "Failed to parse datetime"}

    # 创建按股票+时间分组的目录
    record_dir = _make_record_dir(stock_name, run_ts=run_ts)

    # 保存原始数据，添加元数据注释
    csv_path = record_dir / "record.csv"
    df.to_csv(csv_path, index=False)
    # 在 CSV 开头插入元数据行
    if date_range:
        meta_lines = [f"# Stock: {stock_name}\n", f"# Timeframe: {timeframe}\n", f"# Date Range: {date_range}\n"]
    else:
        meta_lines = [f"# Stock: {stock_name}\n", f"# Timeframe: {timeframe}\n"]
    existing = csv_path.read_text(encoding="utf-8")
    csv_path.write_text("".join(meta_lines) + existing, encoding="utf-8")

    # Build chart title with annotations
    title_parts = [stock_name, timeframe]
    if date_range:
        title_parts.append(date_range)
    chart_title = "  |  ".join(title_parts)

    # Save image locally
    kline_path = record_dir / "kline_chart.png"
    fig, axlist = mpf.plot(
        df[["Open", "High", "Low", "Close"]],
        type="candle",
        style=color.my_color_style,
        figsize=(12, 6),
        returnfig=True,
        block=False,
        title=chart_title,
    )
    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    fig.savefig(
        fname=str(kline_path),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    # ---------- Encode to base64 -----------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # release memory

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "pattern_image": img_b64,
        "pattern_image_description": "Candlestick chart saved locally and returned as base64 string.",
        "record_dir": str(record_dir),
        "kline_path": str(kline_path),
    }


def generate_trend_image(kline_data, stock_name: str = "default", timeframe: str = "1d", date_range: str = "", run_ts: str = None) -> dict:
    """
    Generate a candlestick chart with trendlines from OHLCV data,
    save it locally, and return a base64-encoded image.

    Args:
        kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
        stock_name (str): Stock symbol for organizing output directory.
        timeframe (str): Timeframe of the data (e.g., '1d', '1h', '15m').
        date_range (str): Date range string (e.g., '2024-01-01 ~ 2024-04-01').

    Returns:
        dict: base64 image and description
    """
    data = pd.DataFrame(kline_data)
    candles = data.iloc[-50:].copy()

    candles["Datetime"] = pd.to_datetime(candles["Datetime"])
    candles.set_index("Datetime", inplace=True)

    # 创建按股票+时间分组的目录（复用已有目录或新建）
    record_dir = _make_record_dir(stock_name, run_ts=run_ts)

    # Build chart title with annotations
    title_parts = [stock_name, timeframe]
    if date_range:
        title_parts.append(date_range)
    chart_title = "  |  ".join(title_parts)

    # Trendline fit functions assumed to be defined outside this scope
    support_coefs_c, resist_coefs_c = fit_trendlines_single(candles["Close"])
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles["High"], candles["Low"], candles["Close"]
    )

    # Trendline values
    support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
    resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
    support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
    resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

    # Convert to time-anchored coordinates
    s_seq = get_line_points(candles, support_line)
    r_seq = get_line_points(candles, resist_line)
    s_seq2 = get_line_points(candles, support_line_c)
    r_seq2 = get_line_points(candles, resist_line_c)

    s_segments = split_line_into_segments(s_seq)
    r_segments = split_line_into_segments(r_seq)
    s2_segments = split_line_into_segments(s_seq2)
    r2_segments = split_line_into_segments(r_seq2)

    all_segments = s_segments + r_segments + s2_segments + r2_segments
    colors = (
        ["white"] * len(s_segments)
        + ["white"] * len(r_segments)
        + ["blue"] * len(s2_segments)
        + ["red"] * len(r2_segments)
    )

    # Create addplot lines for close-based support/resistance
    apds = [
        mpf.make_addplot(support_line_c, color="blue", width=1, label="Close Support"),
        mpf.make_addplot(resist_line_c, color="red", width=1, label="Close Resistance"),
    ]

    # Generate figure with legend and save locally
    trend_path = record_dir / "trend_graph.png"
    fig, axlist = mpf.plot(
        candles,
        type="candle",
        style=color.my_color_style,
        addplot=apds,
        alines=dict(alines=all_segments, colors=colors, linewidths=1),
        returnfig=True,
        figsize=(12, 6),
        block=False,
        title=chart_title,
    )

    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    # save fig locally
    fig.savefig(
        str(trend_path), format="png", dpi=600, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)

    # Add legend manually
    axlist[0].legend(loc="upper left")

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "trend_image": img_b64,
        "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
        "record_dir": str(record_dir),
        "trend_path": str(trend_path),
    }
