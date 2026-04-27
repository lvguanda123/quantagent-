"""
Base classes, exceptions, and data structures for the data provider system.

Defines the contract that all data providers must implement, along with
a normalization and validation pipeline that ensures consistent output
regardless of the underlying data source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]
MIN_ROWS = 30


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataProviderError(Exception):
    """Base exception for all data provider errors."""


class DataSourceNotFoundError(DataProviderError):
    """Raised when the requested data source (symbol, file, etc.) does not exist."""


class DataValidationError(DataProviderError):
    """Raised when fetched data fails quality/structure validation."""


class DataFetchError(DataProviderError):
    """Raised when a data fetch operation fails (network error, API error, etc.)."""


# ---------------------------------------------------------------------------
# Request object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FetchRequest:
    """Immutable request specifying what data to fetch.

    Attributes:
        symbol: Asset identifier (e.g., "BTC", "sh600000", "AAPL")
        interval: Time interval ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1mo")
        start_date: Inclusive start of the data window
        end_date: Inclusive end of the data window
        extra: Optional provider-specific parameters
    """
    symbol: str
    interval: str
    start_date: datetime
    end_date: datetime
    extra: Optional[dict] = None

    def __post_init__(self):
        if not self.symbol or not self.symbol.strip():
            raise DataProviderError("symbol must not be empty")
        if self.start_date >= self.end_date:
            raise DataProviderError("start_date must be before end_date")


# ---------------------------------------------------------------------------
# Abstract base provider
# ---------------------------------------------------------------------------

class BaseDataProvider(ABC):
    """Abstract base class for all market data providers.

    Subclasses must implement `fetch()` and set the `name` class attribute.
    The base class provides a standardized normalization and validation pipeline
    that all providers inherit.
    """

    name: str = "base"

    def fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Fetch, normalize, and validate market data.

        This is the public entry point. It calls the subclass's `_raw_fetch()`
        (which returns a raw DataFrame in source-specific format), then pipes
        the result through `_normalize()` and `_validate()`.

        Args:
            req: A FetchRequest specifying the data window

        Returns:
            pd.DataFrame with columns: Datetime, Open, High, Low, Close, [Volume]
        """
        raw = self._raw_fetch(req)
        normed = self._normalize(raw)
        return self._validate(normed)

    @abstractmethod
    def _raw_fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Provider-specific data fetching logic.

        Implementations should return a DataFrame with any column names,
        as `_normalize()` will map them to the standard schema.

        Args:
            req: A FetchRequest

        Returns:
            Raw DataFrame in provider-specific format
        """
        ...

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a raw DataFrame to the standard OHLCV schema.

        Steps:
            1. Rename known source columns to standard names
            2. Ensure Datetime is datetime64[ns] and set as index
            3. Sort by time ascending
            4. Forward-fill NaN, then drop remaining NaN rows
            5. Return a new DataFrame (immutable)

        Subclasses may override `_map_columns()` to customize step 1.
        """
        result = df.copy()

        # Step 1: column mapping
        col_map = self._map_columns(result.columns)
        if col_map:
            existing = {k: v for k, v in col_map.items() if k in result.columns}
            result = result.rename(columns=existing)

        # Step 2: ensure Datetime column
        if "Datetime" not in result.columns:
            raise DataValidationError(
                "Cannot identify Datetime column after normalization"
            )
        result["Datetime"] = pd.to_datetime(result["Datetime"])

        # Ensure numeric columns
        for col in ["Open", "High", "Low", "Close"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        if "Volume" in result.columns:
            result["Volume"] = pd.to_numeric(result["Volume"], errors="coerce")

        # Step 3: sort ascending
        result = result.sort_values("Datetime").reset_index(drop=True)

        # Step 4: fill / drop NaN
        result = result.ffill()
        result = result.dropna(subset=["Open", "High", "Low", "Close"])

        return result.reset_index(drop=True)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate normalized data meets minimum quality standards.

        Checks:
            - Non-empty
            - All required columns present
            - Minimum row count
            - No all-NaN columns among required fields
            - OHLC values are non-negative

        Returns:
            The validated DataFrame (unchanged if valid)

        Raises:
            DataValidationError: on any validation failure
        """
        if df.empty:
            raise DataValidationError("Fetched data is empty after normalization")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required columns after normalization: {missing}"
            )

        if len(df) < MIN_ROWS:
            raise DataValidationError(
                f"Insufficient data: {len(df)} rows (minimum {MIN_ROWS})"
            )

        # Check for all-NaN required columns
        for col in REQUIRED_COLUMNS:
            if df[col].isna().all():
                raise DataValidationError(f"Column '{col}' is entirely NaN")

        # Sanity: prices should be non-negative
        for col in ["Open", "High", "Low", "Close"]:
            if (df[col] < 0).any():
                raise DataValidationError(
                    f"Negative values found in '{col}' column"
                )

        return df

    def _map_columns(self, columns) -> dict:
        """Map source-specific column names to standard names.

        Override in subclasses if the source uses non-standard column names.
        Default returns empty dict (assumes columns already match).

        Args:
            columns: The column index of the raw DataFrame

        Returns:
            Dict mapping {source_name: standard_name}
        """
        return {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
