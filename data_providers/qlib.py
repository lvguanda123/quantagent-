"""
Qlib local data provider.

Reads OHLCV data from a local Qlib data directory. Qlib stores each field
as a separate numpy binary file (``.day.bin`` for daily frequency) alongside
a calendar text file that maps array indices to dates.

Directory layout:
    <data_dir>/
        calendars/day.txt          # one date per line (e.g. "1990-12-19")
        features/<symbol>/
            open.day.bin
            high.day.bin
            low.day.bin
            close.day.bin
            volume.day.bin
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import (
    BaseDataProvider,
    DataFetchError,
    DataSourceNotFoundError,
    FetchRequest,
)

logger = logging.getLogger(__name__)

# Fields to read from Qlib features directory
QLIB_FIELDS = ["open", "high", "low", "close", "volume"]

# Standard column names for each Qlib field
FIELD_TO_COLUMN = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}

# Interval to Qlib frequency directory mapping
INTERVAL_TO_FREQ = {
    "1d": "day",
    "1w": "day",
    "1mo": "day",
}

# Interval to Qlib calendar file mapping
INTERVAL_TO_CALENDAR = {
    "1d": "day.txt",
    "1w": "day.txt",
    "1mo": "day.txt",
}


def _format_qlib_symbol(symbol: str) -> str:
    """Normalize a symbol to Qlib's lowercase-without-dot convention.

    Qlib uses lowercase symbols without the dot separator:
        "SH600000" -> "sh600000"
        "sh.600000" -> "sh600000"
    """
    return symbol.lower().replace(".", "")


class QlibProvider(BaseDataProvider):
    """Reads OHLCV data from a local Qlib data directory.

    Only daily frequency is currently supported, as the Qlib data
    directory uses ``.day.bin`` files.
    """

    name: ClassVar[str] = "qlib"

    def __init__(self, data_dir: str | Path):
        """Initialize the Qlib provider.

        Args:
            data_dir: Root path of the Qlib data directory
                      (e.g., ``~/.qlib/qlib_data/cn_data_rolling``)
        """
        self.data_dir = Path(data_dir).expanduser().resolve()

        if not self.data_dir.exists():
            raise DataSourceNotFoundError(
                f"Qlib data directory not found: {self.data_dir}"
            )

        if not self.data_dir.is_dir():
            raise DataSourceNotFoundError(
                f"Qlib data path is not a directory: {self.data_dir}"
            )

    def _raw_fetch(self, req: FetchRequest) -> pd.DataFrame:
        """Read OHLCV data from Qlib binary files.

        Args:
            req: FetchRequest with symbol, interval, and date range

        Returns:
            DataFrame with Datetime, Open, High, Low, Close, Volume columns

        Raises:
            DataSourceNotFoundError: if symbol or required files are missing
            DataFetchError: if the interval is unsupported or reading fails
        """
        qlib_symbol = _format_qlib_symbol(req.symbol)
        freq = INTERVAL_TO_FREQ.get(req.interval)
        calendar_file = INTERVAL_TO_CALENDAR.get(req.interval)

        if not freq:
            raise DataFetchError(
                f"Qlib provider only supports daily intervals. "
                f"Got interval: {req.interval}"
            )

        features_dir = self.data_dir / "features" / qlib_symbol
        bin_suffix = f".{freq}.bin"

        # Verify the symbol directory exists
        if not features_dir.exists():
            raise DataSourceNotFoundError(
                f"Symbol {req.symbol} (qlib: {qlib_symbol}) not found in "
                f"Qlib data directory: {features_dir}"
            )

        # Read calendar for date indexing
        calendar_path = self.data_dir / "calendars" / calendar_file
        if not calendar_path.exists():
            raise DataFetchError(
                f"Calendar file not found: {calendar_path}"
            )

        full_calendar = _read_calendar(calendar_path)
        if not full_calendar:
            raise DataFetchError("Calendar file is empty")

        # Read each field from binary files
        field_data: dict[str, np.ndarray] = {}
        data_length = None
        for field in QLIB_FIELDS:
            bin_path = features_dir / f"{field}{bin_suffix}"
            if not bin_path.exists():
                logger.warning(
                    "Field %s not found for %s, skipping", field, qlib_symbol
                )
                continue

            try:
                data = np.fromfile(str(bin_path), dtype=np.float32)
                field_data[field] = data
                if data_length is None:
                    data_length = len(data)
            except Exception as e:
                raise DataFetchError(
                    f"Failed to read {bin_path}: {e}"
                ) from e

        if not field_data or data_length is None:
            raise DataFetchError(
                f"No field data found for {qlib_symbol} in {features_dir}"
            )

        # Align calendar with data using the instruments file as a hint.
        # The instruments file maps each stock to its active date range,
        # but it may be stale. We use it to narrow the calendar search,
        # then trust the .bin data length as the ground truth.
        instrument_start, instrument_end = _get_instrument_dates(
            self.data_dir, qlib_symbol
        )

        if instrument_start and instrument_end:
            # Filter calendar to instrument's active trading period
            aligned_dates = [
                d for d in full_calendar
                if instrument_start <= d <= instrument_end
            ]
        else:
            # No instrument info: take the last `data_length` dates
            n_cal = len(full_calendar)
            aligned_dates = full_calendar[n_cal - data_length:]

        # Trust data length over calendar count
        # (instruments file may have stale delisting dates)
        if len(aligned_dates) != data_length:
            n_cal = len(full_calendar)
            aligned_dates = full_calendar[n_cal - data_length:]

        # Build DataFrame
        df = pd.DataFrame({
            FIELD_TO_COLUMN[field]: values[:data_length]
            for field, values in field_data.items()
        })
        df["Datetime"] = pd.to_datetime(aligned_dates)

        # Filter by requested date range
        df = df[
            (df["Datetime"] >= pd.Timestamp(req.start_date))
            & (df["Datetime"] <= pd.Timestamp(req.end_date))
        ]

        if df.empty:
            raise DataFetchError(
                f"No data for {req.symbol} in range "
                f"[{req.start_date.date()}, {req.end_date.date()}]"
            )

        logger.info(
            "Qlib returned %d rows for %s (%s)", len(df), req.symbol, qlib_symbol
        )

        return df

    def list_available_symbols(self) -> list[str]:
        """List all symbols available in the Qlib data directory.

        Returns:
            List of symbol strings (e.g., ["sh600000", "sz000001"])
        """
        features_dir = self.data_dir / "features"
        if not features_dir.exists():
            return []

        return sorted(
            d.name for d in features_dir.iterdir() if d.is_dir()
        )

    def list_available_instruments(self, instrument_file: str = "all.txt") -> list[str]:
        """Read an instrument list from the Qlib data directory.

        Args:
            instrument_file: Name of the instrument file
                             (e.g., "all.txt", "csi300.txt")

        Returns:
            List of symbol strings
        """
        instruments_path = self.data_dir / "instruments" / instrument_file
        if not instruments_path.exists():
            return []

        with open(instruments_path, "r") as f:
            return sorted(
                line.strip().lower().replace(".", "")
                for line in f
                if line.strip() and not line.startswith("#")
            )


def _read_calendar(calendar_path: Path) -> list[str]:
    """Read a Qlib calendar file.

    Args:
        calendar_path: Path to the calendar .txt file

    Returns:
        List of date strings (e.g., ["1990-12-19", "1990-12-20", ...])
    """
    with open(calendar_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _get_instrument_dates(
    data_dir: Path, qlib_symbol: str
) -> tuple[str | None, str | None]:
    """Look up the listing and delisting dates for a symbol.

    Reads from the instruments/all.txt file which has entries like:
        SH600000\t1999-11-10\t2026-03-13

    Args:
        data_dir: Root of the Qlib data directory
        qlib_symbol: Normalized Qlib symbol (e.g., "sh600000")

    Returns:
        Tuple of (start_date, end_date) as strings, or (None, None) if not found.
    """
    instruments_path = data_dir / "instruments" / "all.txt"
    if not instruments_path.exists():
        return None, None

    upper_symbol = qlib_symbol.upper()
    try:
        with open(instruments_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    sym = parts[0].lower()
                    if sym == qlib_symbol or sym == upper_symbol.lower():
                        return parts[1], parts[2]
    except Exception:
        pass

    return None, None
