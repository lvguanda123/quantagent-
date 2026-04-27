"""
AKShare data provider.

Fetches OHLCV data from Chinese financial data sources via the AKShare library.
Supports US stocks, international futures, and global indices — all routed
through a unified interface.

AKShare is completely free, requires no API key, and installs via pip.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar

import pandas as pd

from .base import (
    BaseDataProvider,
    DataFetchError,
    FetchRequest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset type enumeration
# ---------------------------------------------------------------------------

class AssetType(Enum):
    """Category of asset, used to route to the correct AKShare function."""
    US_STOCK = "us_stock"
    A_STOCK = "a_stock"
    A_INDEX = "a_index"
    FUTURES = "futures"
    INDEX = "index"


# ---------------------------------------------------------------------------
# Symbol routing table
# Maps internal asset codes to (AssetType, ak_symbol, display_name)
# ---------------------------------------------------------------------------

AKSHARE_SYMBOLS: dict[str, tuple[AssetType, str, str]] = {
    # A-Shares (via stock_zh_a_hist, 东方财富)
    # Codes: pure 6-digit stock codes (no sh/sz prefix needed internally)
    "000001": (AssetType.A_STOCK, "000001", "平安银行"),
    "600000": (AssetType.A_STOCK, "600000", "浦发银行"),
    "600519": (AssetType.A_STOCK, "600519", "贵州茅台"),
    "000858": (AssetType.A_STOCK, "000858", "五粮液"),
    "601318": (AssetType.A_STOCK, "601318", "中国平安"),
    "600036": (AssetType.A_STOCK, "600036", "招商银行"),
    "000002": (AssetType.A_STOCK, "000002", "万科A"),
    "601012": (AssetType.A_STOCK, "601012", "隆基绿能"),
    "300750": (AssetType.A_STOCK, "300750", "宁德时代"),
    "002714": (AssetType.A_STOCK, "002714", "牧原股份"),

    # A-Share Indices (via stock_zh_index_daily)
    "SH000001": (AssetType.A_INDEX, "sh000001", "上证指数"),
    "SZ399001": (AssetType.A_INDEX, "sz399001", "深证成指"),
    "SZ399006": (AssetType.A_INDEX, "sz399006", "创业板指"),
    "SH000300": (AssetType.A_INDEX, "sh000300", "沪深300"),

    # US Stocks (via stock_us_daily)
    "AAPL": (AssetType.US_STOCK, "AAPL", "Apple Inc."),
    "TSLA": (AssetType.US_STOCK, "TSLA", "Tesla Inc."),
    "GOOGL": (AssetType.US_STOCK, "GOOGL", "Alphabet Inc."),
    "MSFT": (AssetType.US_STOCK, "MSFT", "Microsoft Corp."),
    "AMZN": (AssetType.US_STOCK, "AMZN", "Amazon.com Inc."),
    "NVDA": (AssetType.US_STOCK, "NVDA", "NVIDIA Corp."),
    "META": (AssetType.US_STOCK, "META", "Meta Platforms"),
    "NFLX": (AssetType.US_STOCK, "NFLX", "Netflix Inc."),

    # US Index ETFs — used as proxies for major indices
    # (index_global_hist_sina does NOT cover S&P 500 / Dow / Nasdaq directly)
    "SPX": (AssetType.US_STOCK, "SPY", "S&P 500 ETF (SPY)"),
    "DJI": (AssetType.US_STOCK, "DIA", "Dow Jones ETF (DIA)"),
    "NQ": (AssetType.US_STOCK, "QQQ", "Nasdaq 100 ETF (QQQ)"),

    # Global Indices (via index_global_hist_sina, Chinese names required)
    # Only indices listed in ak.index_global_name_table() are available
    "NKY": (AssetType.INDEX, "日经225指数", "Nikkei 225"),
    "DAX": (AssetType.INDEX, "德国DAX30指数", "DAX 30"),
    "FTSE": (AssetType.INDEX, "英国富时100指数", "FTSE 100"),
    "HSI": (AssetType.INDEX, "恒生指数", "Hang Seng Index"),
    "KOSPI": (AssetType.INDEX, "韩国综合指数", "KOSPI"),

    # International Futures (via futures_foreign_hist)
    # Note: This endpoint may have intermittent network issues
    "GC": (AssetType.FUTURES, "黄金", "COMEX Gold"),
    "CL": (AssetType.FUTURES, "原油", "NYMEX Crude Oil"),
}


@dataclass(frozen=True)
class AKShareConfig:
    """Configuration for the AKShare provider.

    Attributes:
        adjust: Price adjustment method. 'qfq' = forward-adjusted, 'hfq' = backward-adjusted, '' = raw
        symbol_map: Override or extend the default symbol routing table
    """
    adjust: str = "qfq"
    symbol_map: dict[str, tuple[AssetType, str, str]] = field(
        default_factory=lambda: {}
    )


class AKShareProvider(BaseDataProvider):
    """Fetches OHLCV data via AKShare — a free Chinese financial data library.

    Supports three data sources:
    - US stocks (Sina via AKShare)
    - International futures (Sina via AKShare)
    - Global indices (Sina via AKShare)

    All three route through different AKShare functions but return
    data in the same standard OHLCV format.
    """

    name: ClassVar[str] = "akshare"

    def __init__(
        self,
        config: AKShareConfig | None = None,
        symbol_map: dict[str, tuple[AssetType, str, str]] | None = None,
    ):
        """Initialize the AKShare provider.

        Args:
            config: AKShareConfig with adjust mode and symbol overrides
            symbol_map: Quick override for symbol routing
        """
        self._config = config or AKShareConfig()
        self._symbols: dict[str, tuple[AssetType, str, str]] = {
            **AKSHARE_SYMBOLS,
            **self._config.symbol_map,
            **(symbol_map or {}),
        }

    def _raw_fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Fetch data from AKShare, routing to the correct function by asset type.

        Args:
            req: FetchRequest with symbol, interval, and date range

        Returns:
            Raw DataFrame from AKShare (columns may vary by source)

        Raises:
            DataFetchError: on network errors, unknown symbols, or empty responses
        """
        # Look up the asset type and AKShare symbol
        asset_info = self._resolve_symbol(req.symbol)
        asset_type, ak_symbol, display_name = asset_info

        logger.info(
            "Fetching %s (%s) from AKShare: %s to %s",
            display_name, req.symbol, req.start_date.date(), req.end_date.date(),
        )

        try:
            if asset_type == AssetType.A_STOCK:
                df = self._fetch_a_stock(ak_symbol)
            elif asset_type == AssetType.A_INDEX:
                df = self._fetch_a_index(ak_symbol)
            elif asset_type == AssetType.US_STOCK:
                df = self._fetch_us_stock(ak_symbol)
            elif asset_type == AssetType.FUTURES:
                df = self._fetch_futures(ak_symbol)
            elif asset_type == AssetType.INDEX:
                df = self._fetch_index(ak_symbol)
            else:
                raise DataFetchError(
                    f"Unsupported asset type: {asset_type.value}"
                )
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                f"AKShare fetch failed for {display_name} ({ak_symbol}): {e}"
            ) from e

        # Filter by date range (AKShare functions return all data, we slice locally)
        df["date"] = pd.to_datetime(df["date"])
        df = df[
            (df["date"] >= pd.Timestamp(req.start_date))
            & (df["date"] <= pd.Timestamp(req.end_date))
        ]

        if df.empty:
            raise DataFetchError(
                f"No data for {display_name} in range "
                f"[{req.start_date.date()}, {req.end_date.date()}]"
            )

        # Rename 'date' -> 'Datetime' so base._normalize can process it
        df = df.rename(columns={"date": "Datetime"})

        logger.info(
            "AKShare returned %d rows for %s (%s)",
            len(df), display_name, req.symbol,
        )

        return df

    def _resolve_symbol(self, symbol: str) -> tuple[AssetType, str, str]:
        """Look up the internal symbol in the routing table.

        Args:
            symbol: Internal asset code (e.g., 'AAPL', 'GC', 'SPX')

        Returns:
            Tuple of (AssetType, ak_symbol, display_name)

        Raises:
            DataFetchError: if symbol is not found
        """
        if symbol not in self._symbols:
            available = sorted(self._symbols.keys())
            raise DataFetchError(
                f"Unknown symbol: {symbol!r}. "
                f"Available symbols: {available}"
            )
        return self._symbols[symbol]

    # -----------------------------------------------------------------------
    # Data source implementations
    # -----------------------------------------------------------------------

    def _fetch_a_stock(self, symbol: str) -> pd.DataFrame:
        """Fetch A-share stock data via stock_zh_a_daily (Sina source).

        Args:
            symbol: 6-digit stock code (e.g., '000001', '600519')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        import akshare as ak

        # Determine Sina symbol prefix: sz (Shenzhen) or sh (Shanghai)
        if symbol.startswith("6"):
            sina_symbol = f"sh{symbol}"
        else:
            sina_symbol = f"sz{symbol}"

        df = ak.stock_zh_a_daily(symbol=sina_symbol)

        if df is None or df.empty:
            raise DataFetchError(
                f"No A-share data returned for {symbol}"
            )

        return df

    def _fetch_a_index(self, symbol: str) -> pd.DataFrame:
        """Fetch A-share index data via stock_zh_index_daily (Sina source).

        Args:
            symbol: Index symbol with sh/sz prefix (e.g., 'sh000001', 'sz399006')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=symbol)

        if df is None or df.empty:
            raise DataFetchError(
                f"No A-share index data returned for {symbol}"
            )

        return df

    def _fetch_us_stock(self, symbol: str) -> pd.DataFrame:
        """Fetch US stock data via stock_us_daily (Sina source).

        Args:
            symbol: US stock ticker (e.g., 'AAPL', 'TSLA')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        import akshare as ak

        df = ak.stock_us_daily(symbol=symbol, adjust=self._config.adjust)

        if df is None or df.empty:
            raise DataFetchError(
                f"No US stock data returned for {symbol}"
            )

        return df

    def _fetch_futures(self, symbol: str) -> pd.DataFrame:
        """Fetch international futures data via futures_foreign_hist (Sina source).

        Args:
            symbol: Futures symbol in Chinese (e.g., '黄金', '原油')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, ...
        """
        import akshare as ak

        df = ak.futures_foreign_hist(symbol=symbol)

        if df is None or df.empty:
            raise DataFetchError(
                f"No futures data returned for {symbol}"
            )

        return df

    def _fetch_index(self, symbol: str) -> pd.DataFrame:
        """Fetch global index data via index_global_hist_sina (Sina source).

        Args:
            symbol: Index name in Chinese (e.g., '标普500指数', '日经225指数')

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        import akshare as ak

        df = ak.index_global_hist_sina(symbol=symbol)

        if df is None or df.empty:
            raise DataFetchError(
                f"No index data returned for {symbol}"
            )

        return df

    # -----------------------------------------------------------------------
    # Column mapping
    # -----------------------------------------------------------------------

    def _map_columns(self, columns) -> dict:
        """Map AKShare column names to standard OHLCV names.

        Handles three different column naming conventions:
        - English lowercase: date, open, high, low, close, volume
        - Chinese A-share: 日期, 开盘, 最高, 最低, 收盘, 成交量
        - Already standardized: Datetime, Open, High, Low, Close, Volume
        """
        mapping = {}
        for col in columns:
            col_lower = str(col).lower()
            col_str = str(col)

            if col_lower == "date" or col_str == "日期":
                mapping[col] = "Datetime"
            elif col_lower == "open" or col_str == "开盘":
                mapping[col] = "Open"
            elif col_lower == "high" or col_str == "最高":
                mapping[col] = "High"
            elif col_lower == "low" or col_str == "最低":
                mapping[col] = "Low"
            elif col_lower == "close" or col_str == "收盘":
                mapping[col] = "Close"
            elif col_lower == "volume" or col_str == "成交量":
                mapping[col] = "Volume"
        return mapping

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_available_symbols(self, asset_type: AssetType | None = None) -> list[str]:
        """List all symbols available through this provider.

        Args:
            asset_type: Filter by asset type. None = all.

        Returns:
            List of symbol strings (e.g., ['AAPL', 'SPX', 'GC'])
        """
        if asset_type is None:
            return sorted(self._symbols.keys())

        return sorted(
            sym for sym, (at, _, _) in self._symbols.items()
            if at == asset_type
        )

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get metadata for a symbol.

        Args:
            symbol: Internal asset code

        Returns:
            Dict with 'type', 'ak_symbol', 'display_name' or None if not found
        """
        if symbol not in self._symbols:
            return None

        asset_type, ak_symbol, display_name = self._symbols[symbol]
        return {
            "type": asset_type.value,
            "ak_symbol": ak_symbol,
            "display_name": display_name,
        }
