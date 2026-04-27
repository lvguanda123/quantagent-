"""
Yahoo Finance data provider.

Extracts market data from Yahoo Finance via the yfinance library and
normalizes it to the standard OHLCV schema.
"""

import logging
from datetime import datetime
from typing import ClassVar

import pandas as pd
import yfinance as yf

from .base import (
    BaseDataProvider,
    DataFetchError,
    FetchRequest,
)

logger = logging.getLogger(__name__)

# Symbol mapping: internal name -> Yahoo Finance ticker
YAHOO_SYMBOLS: dict[str, str] = {
    "SPX": "^GSPC",
    "BTC": "BTC-USD",
    "GC": "GC=F",
    "NQ": "NQ=F",
    "CL": "CL=F",
    "ES": "ES=F",
    "DJI": "^DJI",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
}

# Interval mapping: internal -> yfinance
YAHOO_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1wk",
    "1mo": "1mo",
}


class YahooFinanceProvider(BaseDataProvider):
    """Fetches OHLCV data from Yahoo Finance via yfinance.

    Handles symbol mapping, interval conversion, MultiIndex column
    normalization, and rate-limit errors.
    """

    name: ClassVar[str] = "yahoo"

    def __init__(
        self,
        symbol_map: dict[str, str] | None = None,
        interval_map: dict[str, str] | None = None,
    ):
        """Initialize the provider.

        Args:
            symbol_map: Optional custom symbol mapping. Merges with defaults.
            interval_map: Optional custom interval mapping. Merges with defaults.
        """
        self._symbols = {**YAHOO_SYMBOLS, **(symbol_map or {})}
        self._intervals = {**YAHOO_INTERVALS, **(interval_map or {})}

    def _raw_fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Download data from Yahoo Finance.

        Args:
            req: FetchRequest with symbol, interval, and date range

        Returns:
            Raw DataFrame from yfinance (may have MultiIndex columns)

        Raises:
            DataFetchError: on network errors, rate limits, or empty responses
        """
        yf_symbol = self._symbols.get(req.symbol, req.symbol)
        yf_interval = self._intervals.get(req.interval, req.interval)

        logger.info(
            "Fetching %s from Yahoo Finance: %s to %s, interval=%s",
            yf_symbol, req.start_date, req.end_date, yf_interval,
        )

        try:
            df = yf.download(
                tickers=yf_symbol,
                start=req.start_date,
                end=req.end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )
        except Exception as e:
            raise DataFetchError(
                f"Yahoo Finance download failed for {yf_symbol}: {e}"
            ) from e

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise DataFetchError(
                f"No data returned from Yahoo Finance for {yf_symbol}"
            )

        return self._postprocess_yfinance(df)

    def _postprocess_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle yfinance-specific DataFrame shapes.

        yfinance may return:
        - MultiIndex columns (Ticker, Price)
        - 'Date' as index rather than column
        - Series instead of DataFrame (single-column results)

        This method normalizes all cases to a flat-column DataFrame.
        """
        result = df.copy()

        # Handle Series
        if isinstance(result, pd.Series):
            result = result.to_frame()

        if not isinstance(result, pd.DataFrame):
            raise DataFetchError("Unexpected non-DataFrame response from yfinance")

        # Flatten MultiIndex columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)

        # Reset index if Date is the index
        if isinstance(result.index, pd.DatetimeIndex) and "Date" not in result.columns:
            result = result.reset_index()
            # Rename 'Date' -> 'Datetime' handled by base._normalize column mapping
            if "Date" in result.columns:
                result = result.rename(columns={"Date": "Datetime"})

        return result

    def _map_columns(self, columns) -> dict:
        """Map yfinance column names to standard names."""
        mapping = {}
        for col in columns:
            col_lower = col.lower() if isinstance(col, str) else col
            if col_lower == "datetime" or col_lower == "date":
                mapping[col] = "Datetime"
            elif col_lower == "open":
                mapping[col] = "Open"
            elif col_lower == "high":
                mapping[col] = "High"
            elif col_lower == "low":
                mapping[col] = "Low"
            elif col_lower == "close":
                mapping[col] = "Close"
            elif col_lower == "volume":
                mapping[col] = "Volume"
        return mapping
