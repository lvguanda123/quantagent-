"""
Data provider package.

A pluggable data access layer for quantitative trading. Each data provider
implements a common contract and returns data in a standard OHLCV format,
fully decoupled from downstream analysis agents.

Quick start:
    from data_providers import get_provider, FetchRequest

    provider = get_provider("yahoo")
    df = provider.fetch(FetchRequest(
        symbol="BTC",
        interval="1d",
        start_date=datetime(2026, 3, 1),
        end_date=datetime(2026, 4, 15),
    ))
"""

from .base import (
    BaseDataProvider,
    DataFetchError,
    DataProviderError,
    DataSourceNotFoundError,
    DataValidationError,
    FetchRequest,
)
from .qlib import QlibProvider
from .registry import (
    ProviderRegistry,
    create_default_registry,
    get_provider,
    list_providers,
)
from .yahoo import YahooFinanceProvider
from .akshare import AKShareProvider, AKShareConfig, AssetType

__all__ = [
    # Base
    "BaseDataProvider",
    "FetchRequest",
    # Exceptions
    "DataProviderError",
    "DataSourceNotFoundError",
    "DataValidationError",
    "DataFetchError",
    # Providers
    "YahooFinanceProvider",
    "QlibProvider",
    "AKShareProvider",
    "AKShareConfig",
    "AssetType",
    # Registry
    "ProviderRegistry",
    "create_default_registry",
    "get_provider",
    "list_providers",
]
