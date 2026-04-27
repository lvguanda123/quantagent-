"""
Provider registry and factory functions.

Provides a simple string-based lookup for data providers, along with
a factory function that instantiates the correct provider based on
a name and optional configuration.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseDataProvider, DataSourceNotFoundError

logger = logging.getLogger(__name__)

# Default Qlib data directory
DEFAULT_QLIB_DIR = Path(r"C:\Users\Administrator\.qlib\qlib_data\cn_data_rolling")


class ProviderRegistry:
    """Thread-safe registry for named data provider instances.

    Providers are registered with a string key and can be retrieved
    by name. The registry also supports a fallback factory function
    for lazy instantiation.
    """

    def __init__(self):
        self._providers: dict[str, BaseDataProvider] = {}
        self._factories: dict[str, callable] = {}

    def register(self, name: str, provider: BaseDataProvider) -> None:
        """Register an existing provider instance.

        Args:
            name: String key for lookup (e.g., "yahoo", "qlib")
            provider: A BaseDataProvider instance
        """
        self._providers[name] = provider
        logger.debug("Registered provider: %s -> %s", name, provider.name)

    def register_factory(self, name: str, factory: callable) -> None:
        """Register a factory function for lazy provider instantiation.

        The factory will be called the first time the provider is requested.

        Args:
            name: String key for lookup
            factory: A callable that returns a BaseDataProvider
        """
        self._factories[name] = factory
        logger.debug("Registered factory: %s", name)

    def get(self, name: str) -> BaseDataProvider:
        """Retrieve a provider by name.

        If the provider was registered directly, returns the cached instance.
        If only a factory was registered, instantiates and caches the provider.

        Args:
            name: The provider name (e.g., "yahoo", "qlib")

        Returns:
            A BaseDataProvider instance

        Raises:
            DataSourceNotFoundError: if the provider is not registered
        """
        # Check cached instances first
        if name in self._providers:
            return self._providers[name]

        # Try factory
        if name in self._factories:
            provider = self._factories[name]()
            self._providers[name] = provider
            logger.info("Instantiated provider via factory: %s", name)
            return provider

        raise DataSourceNotFoundError(
            f"Unknown data provider: {name!r}. "
            f"Available: {sorted(self.list_providers())}"
        )

    def list_providers(self) -> list[str]:
        """Return all registered provider names (instances + factories)."""
        return sorted(set(self._providers.keys()) | set(self._factories.keys()))


def create_default_registry(
    qlib_data_dir: Optional[Path] = None,
) -> ProviderRegistry:
    """Create a registry with all built-in providers.

    Args:
        qlib_data_dir: Custom Qlib data directory. Uses default if not provided.

    Returns:
        A populated ProviderRegistry
    """
    from .yahoo import YahooFinanceProvider
    from .qlib import QlibProvider
    from .akshare import AKShareProvider

    registry = ProviderRegistry()
    registry.register("yahoo", YahooFinanceProvider())
    registry.register("qlib", QlibProvider(data_dir=qlib_data_dir or DEFAULT_QLIB_DIR))
    registry.register("akshare", AKShareProvider())

    return registry


# Module-level singleton for convenience
_default_registry: Optional[ProviderRegistry] = None


def _get_default_registry() -> ProviderRegistry:
    """Get or create the default singleton registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry


def get_provider(name: str, **kwargs) -> BaseDataProvider:
    """Get a data provider by name from the default registry.

    Convenience function for the common case of using built-in providers.

    Args:
        name: Provider name ("yahoo" or "qlib")
        **kwargs: Passed to create_default_registry if creating the singleton

    Returns:
        A BaseDataProvider instance
    """
    return _get_default_registry().get(name)


def list_providers() -> list[str]:
    """List all available provider names."""
    return _get_default_registry().list_providers()
