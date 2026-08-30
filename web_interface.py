"""
Web Interface for QuantAgent Trading Analysis.
Supports AKShare: A-shares, funds, ETF minute data.
"""

import json
import io
import os
import re
import requests
import threading
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR.startswith("\\\\?\\"):
    _BASE_DIR = _BASE_DIR[4:]


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ.

    Lightweight, dependency-free loader so that local API keys can be kept
    out of source control (.env is git-ignored). Existing environment
    variables take precedence over values in the file.
    """
    env_path = os.path.join(_BASE_DIR, path)
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as error:
        print(f"Warning: could not read .env: {error}")


# Load the local, git-ignored .env first (development). Then import the build-time
# generated _builtin_keys module which provides baked-in keys for the desktop app
# via os.environ.setdefault. It is created by the packaging step from GitHub Secrets
# and is NOT committed to git. Using a .py file (instead of a .env dotfile) ensures
# it is always included by the Tauri/NSIS bundler, which has been observed to drop
# dotfiles from the installed resources.
_load_dotenv()
try:
    import _builtin_keys  # type: ignore  # noqa: F401
except ImportError:
    pass

import static_util
import history_store

# ---------------------------------------------------------------------------
# Output cache: the analyze response is too large to fit in a URL query
# string (base64 PNGs alone are 50-200KB). The desktop WebView2 shell
# navigates to /output after a successful analyze, so we stash the full
# response in an in-memory dict keyed by a short UUID and pass the key in
# the redirect URL. Only the most recent N results are kept to bound
# memory; older entries are evicted on insert.
# ---------------------------------------------------------------------------
_OUTPUT_CACHE_MAX = 8
_output_cache: Dict[str, dict] = {}
_output_cache_lock = threading.Lock()


def _stash_output_result(payload: dict) -> str:
    """Store ``payload`` in the output cache and return the lookup key.

    Evicts the oldest entry once the cache exceeds ``_OUTPUT_CACHE_MAX``.
    """
    key = uuid.uuid4().hex
    with _output_cache_lock:
        while len(_output_cache) >= _OUTPUT_CACHE_MAX:
            oldest_key = next(iter(_output_cache))
            _output_cache.pop(oldest_key, None)
        _output_cache[key] = payload
    return key


def _pop_output_result(key: str):
    """Return and remove the cached result for ``key`` (if any)."""
    if not key:
        return None
    with _output_cache_lock:
        return _output_cache.pop(key, None)
from trading_graph import TradingGraph

# PDF export is academic-only; we import lazily so trader builds without
# reportlab installed don't fail to start.
try:
    import report_export  # type: ignore
    REPORT_EXPORT_AVAILABLE = True
except ImportError as error:
    report_export = None
    REPORT_EXPORT_AVAILABLE = False
    print(f"Warning: report_export not available: {error}")

try:
    from data_providers import DataProviderError, FetchRequest, QlibProvider, YahooFinanceProvider
    DATA_PROVIDERS_AVAILABLE = True
except ImportError as error:
    DATA_PROVIDERS_AVAILABLE = False
    DataProviderError = Exception
    FetchRequest = None
    QlibProvider = None
    YahooFinanceProvider = None
    print(f"Warning: data providers not available: {error}")

# AKShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError as error:
    AKSHARE_AVAILABLE = False
    print(f"Warning: AKShare not available: {error}")


app = Flask(
    __name__,
    root_path=_BASE_DIR,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static"),
    static_url_path="/static",
)

# Initialise the local per-user analysis history database (SQLite).
try:
    history_store.init_db()
except Exception as error:  # noqa: BLE001 - history is non-critical to startup
    print(f"Warning: could not initialise history database: {error}")


# Application variant — selects trader vs academic template + capabilities.
# Priority: QUANTAGENT_APP_VARIANT env (set by lib.rs) > generated
# _variant.py (written by prepare-backend.mjs at build time) > "trader".
def _resolve_app_variant() -> str:
    env_value = os.environ.get("QUANTAGENT_APP_VARIANT", "").strip().lower()
    if env_value in ("trader", "academic"):
        return env_value
    try:
        import _variant  # type: ignore  # noqa: F401
        baked = getattr(_variant, "APP_VARIANT", "").strip().lower()
        if baked in ("trader", "academic"):
            return baked
    except ImportError:
        pass
    return "trader"


APP_VARIANT = _resolve_app_variant()
APP_VARIANT_FEATURES = {
    "academic": {
        "show_backtest": True,
        "show_export": True,
        "show_process": True,
    },
    "trader": {
        "show_backtest": False,
        "show_export": False,
        "show_process": False,
    },
}[APP_VARIANT]


def normalize_custom_base_url(base_url: str) -> str:
    """Normalize common OpenAI-compatible provider URLs to their base URL."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized == "https://ark.cn-beijing.volces.com/api/coding":
        normalized = "https://ark.cn-beijing.volces.com/api/v3"
    return normalized


def normalize_custom_base_url_for_mode(base_url: str, mode: str) -> str:
    """Only normalize OpenAI-compatible URLs; other protocols keep exact base URL."""
    if mode == "openai":
        return normalize_custom_base_url(base_url)
    return (base_url or "").strip().rstrip("/")


def format_analysis_error(message: str) -> str:
    """Turn provider-specific model errors into actionable UI text."""
    text = str(message)
    if "InvalidEndpointOrModel.NotFound" in text or "model or endpoint" in text:
        return (
            "模型或 Endpoint 不存在，或者当前 Key 没有权限。"
            "如果使用火山 Ark 的 OpenAI 兼容模式，Base URL 请填 "
            "https://ark.cn-beijing.volces.com/api/v3，模型名称请填控制台实际开通的 "
            "endpoint/model id；ark-code-latest 不一定是你的账号可直接调用的模型名。"
            f" 原始错误：{text}"
        )
    return text


def _apply_env_credentials(config: dict) -> None:
    """Override config with API credentials from the environment / bundled .env.

    Mutates ``config`` in place. Existing non-empty config values are kept only
    when the corresponding environment variable is absent. Called both before
    the trading graph is constructed (so its LLM clients pick up the key) and
    after (to keep the running instance in sync).
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        config["anthropic_api_key"] = anthropic_key

    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if anthropic_base:
        config["base_url"] = anthropic_base

    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "").strip()
    if anthropic_model:
        config["agent_llm_model"] = anthropic_model
        config["graph_llm_model"] = anthropic_model

    trial_key = os.environ.get("QUANTAGENT_TRIAL_API_KEY", "").strip()
    if trial_key:
        config["trial_api_key"] = trial_key


class WebTradingAnalyzer:
    def __init__(self):
        from default_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()
        # Apply credentials from environment / bundled .env BEFORE constructing
        # the trading graph, because its LLM clients are initialised immediately
        # and read the key from config (they do not fall back to os.environ).
        _apply_env_credentials(self.config)
        self.trading_graph = TradingGraph(config=self.config)

    def normalize_akshare_timeframe(self, timeframe: str) -> str:
        """Map UI interval names to AKShare interval names."""
        return {
            "1h": "60m",
            "60": "60m",
            "60min": "60m",
        }.get(timeframe, timeframe)

    def is_cn_market_symbol(self, symbol: str) -> bool:
        """Return whether a symbol is likely supported by AKShare A-share routes."""
        clean_upper = (symbol or "").strip().upper()
        clean = clean_upper.replace("SZ", "").replace("SH", "")
        return clean_upper.startswith(("SH", "SZ")) or (clean.isdigit() and len(clean) == 6)

    def update_api_key(
        self,
        api_key: str,
        provider: str = "openai",
        base_url: str = "",
        model: str = "",
        custom_options: dict | None = None,
    ):
        """Update a provider key using the trading graph's existing storage flow."""
        self.trading_graph.update_api_key(api_key, provider, base_url, model, custom_options)
        self.config = self.trading_graph.config

    def validate_sohu_api_key(self, api_key: str):
        """Validate Sohu with a minimal text-only Chat Completions request."""
        response = requests.post(
            "https://llm-api-test.tv.sohuno.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "qwen3.6-plus",
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
            },
            timeout=20,
        )
        if response.ok:
            return

        try:
            error_data = response.json()
            message = error_data.get("message") or error_data.get("error")
        except ValueError:
            message = response.text
        raise ValueError(f"搜狐接口验证失败（HTTP {response.status_code}）：{message or '未知错误'}")

    @staticmethod
    def _sina_tx_symbol(symbol: str) -> str:
        """Map a bare 6-digit A-share code to the Sina/Tencent sh/sz form."""
        s = symbol.strip().lower().replace("sh", "").replace("sz", "")
        if not s.isdigit() or len(s) != 6:
            return symbol
        # 6/9/5 -> Shanghai, otherwise Shenzhen
        return ("sh" if s[0] in ("6", "9", "5") else "sz") + s

    def _standardize_ohlcv(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize different source column names to the Datetime/OHLCV schema."""
        if df is None or df.empty:
            return pd.DataFrame()

        rename_map = {
            # 东方财富: 日期/开盘/最高/最低/收盘/成交量
            "日期": "Datetime", "开盘": "Open", "最高": "High",
            "最低": "Low", "收盘": "Close", "成交量": "Volume",
            # 新浪/腾讯: date/open/high/low/close/volume
            "date": "Datetime", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        # Some sources return "date" as the index name after reset_index
        if "Datetime" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex) or "date" in str(df.index.name).lower():
                df = df.reset_index().rename(columns=rename_map)

        required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            print(f"[{source}] missing columns, got: {df.columns.tolist()}")
            return pd.DataFrame()

        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df[required].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Open", "High", "Low", "Close"])

    def fetch_akshare_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch A-share daily OHLCV with three sources as fallbacks.

        Order: 东方财富 (stock_zh_a_hist) -> 新浪 (stock_zh_a_daily)
        -> 腾讯 (stock_zh_a_hist_tx). Each source is retried once on transient
        network errors before moving on, so a single provider being down or
        rate-limited does not block the whole market. Minute-level data is
        handled separately and is unaffected.
        """
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        clean = symbol.replace("sz", "").replace("SZ", "").replace("sh", "").replace("SH", "")
        st = start_date.replace("-", "")
        et = end_date.replace("-", "")
        sina_tx = self._sina_tx_symbol(symbol)

        def eastmoney():
            return ak.stock_zh_a_hist(
                symbol=clean, period="daily", start_date=st, end_date=et, adjust="qfq")

        def sina():
            return ak.stock_zh_a_daily(
                symbol=sina_tx, start_date=st, end_date=et, adjust="qfq")

        def tencent():
            return ak.stock_zh_a_hist_tx(
                symbol=sina_tx, start_date=st, end_date=et, adjust="qfq", timeout=15)

        sources = [("eastmoney", eastmoney), ("sina", sina), ("tencent", tencent)]
        errors = []
        for name, fn in sources:
            for attempt in range(2):
                try:
                    raw = fn()
                    df = self._standardize_ohlcv(raw, name)
                    if not df.empty:
                        if name != "eastmoney":
                            print(f"Using daily source fallback: {name} for {symbol}")
                        return df
                    errors.append(f"{name}: empty")
                    break  # empty is not a network error, try next source
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{name}: {e}")
                    print(f"[{name}] A-share daily fetch failed (try {attempt + 1}/2): {e}")
                    if attempt == 0:
                        import time
                        time.sleep(1.2)

        print(f"All daily sources failed for {symbol}: {' | '.join(errors)}")
        return pd.DataFrame()

    def fetch_akshare_fund_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch fund net value data (daily)."""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            df = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")

            if df is None or df.empty:
                return pd.DataFrame()

            df["Datetime"] = pd.to_datetime(df["净值日期"])
            df["Close"] = df["单位净值"]
            df["Open"] = df["单位净值"].shift(1).fillna(df["单位净值"].iloc[0])
            df["High"] = df[["Open", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Close"]].min(axis=1)
            df["Volume"] = 1000000

            df = df[
                (df["Datetime"] >= pd.Timestamp(start_date)) &
                (df["Datetime"] <= pd.Timestamp(end_date))
            ]

            return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            print(f"Error fetching fund data: {e}")
            return pd.DataFrame()

    def fetch_akshare_etf_min_data(
        self, symbol: str, interval: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        """Fetch ETF minute-level data (1m/5m/15m/30m/60m)."""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
            period = period_map.get(interval, "1")

            df = ak.fund_etf_hist_min_em(
                symbol=symbol,
                period=period,
                adjust="",
                start_date=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end_datetime.strftime("%Y-%m-%d %H:%M:%S")
            )

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "时间": "Datetime",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low"
            })

            required = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                return pd.DataFrame()

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            return df[required]

        except Exception as e:
            print(f"Error fetching ETF minute data: {e}")
            return pd.DataFrame()

    def fetch_akshare_stock_min_data(
        self, symbol: str, interval: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        """Fetch A-share stock minute-level data (1m/5m/15m/30m/60m).

        Uses Sina's stock_zh_a_minute (recent trading day only, but reliable).
        Falls back to East Money's stock_zh_a_hist_min_em if Sina fails.
        """
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
            period = period_map.get(interval, "1")

            clean_symbol = symbol.replace("sz", "").replace("SZ", "").replace("sh", "").replace("SH", "")

            # Determine Sina prefix: sz (Shenzhen) or sh (Shanghai)
            sina_symbol = f"sz{clean_symbol}" if not clean_symbol.startswith("6") else f"sh{clean_symbol}"

            # Primary: Sina stock_zh_a_minute (more reliable for minute data)
            df = ak.stock_zh_a_minute(symbol=sina_symbol, period=period, adjust="")

            if df is not None and not df.empty:
                df = df.rename(columns={
                    "day": "Datetime",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })

                required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in required):
                    # Convert OHLCV columns to numeric (Sina returns strings)
                    for c in ["Open", "High", "Low", "Close", "Volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    # Drop rows with NaN after conversion
                    df = df.dropna(subset=["Open", "High", "Low", "Close"])
                    df["Datetime"] = pd.to_datetime(df["Datetime"])
                    # Filter by date range
                    df = df[(df["Datetime"] >= start_datetime) & (df["Datetime"] <= end_datetime)]
                    if not df.empty:
                        return df[required]

        except Exception as e:
            print(f"Sina minute data failed, falling back to East Money: {e}")
            # Fallback: East Money stock_zh_a_hist_min_em
            try:
                clean_symbol = symbol.replace("sz", "").replace("SZ", "").replace("sh", "").replace("SH", "")
                period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
                period = period_map.get(interval, "1")
                adjust = "" if period == "1" else "hfq"

                df = ak.stock_zh_a_hist_min_em(
                    symbol=clean_symbol,
                    period=period,
                    adjust=adjust,
                    start_date=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=end_datetime.strftime("%Y-%m-%d %H:%M:%S")
                )

                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "时间": "Datetime",
                        "开盘": "Open",
                        "收盘": "Close",
                        "最高": "High",
                        "最低": "Low",
                        "成交量": "Volume"
                    })

                    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
                    if all(c in df.columns for c in required):
                        df["Datetime"] = pd.to_datetime(df["Datetime"])
                        return df[required]
            except Exception as e2:
                print(f"East Money fallback also failed: {e2}")

        return pd.DataFrame()

    def fetch_akshare_index_min_data(
        self, symbol: str, interval: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        """Fetch A-share index minute-level data (1m/5m/15m/30m/60m).

        Uses Sina's CN_MarketDataService.getKLineData API directly
        (akshare's index_zh_a_hist_min_em is unstable for East Money).
        """
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
            period = period_map.get(interval, "1")

            clean_symbol = symbol.replace("sz", "").replace("SZ", "").replace("sh", "").replace("SH", "")
            sina_prefix = "sh" if clean_symbol.startswith(("0", "9")) else "sz"
            sina_symbol = f"{sina_prefix}{clean_symbol}"

            url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
            params = {
                "symbol": sina_symbol,
                "scale": period,
                "ma": "no",
                "datalen": "1023",
            }
            r = requests.get(url, params=params, headers={"Referer": "https://finance.sina.com.cn"}, timeout=15)
            text = r.text
            # Parse JSONP: strip the callback wrapper
            json_start = text.find("(")
            json_end = text.rfind(")")
            if json_start == -1 or json_end == -1:
                print(f"Sina index minute JSONP parse failed: {text[:200]}")
                return pd.DataFrame()

            data = json.loads(text[json_start + 1:json_end])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "day": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })

            # Convert OHLCV to numeric
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Open", "High", "Low", "Close"])

            required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required):
                print(f"Index minute missing columns: {df.columns.tolist()}")
                return pd.DataFrame()

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            # Filter by date range
            df = df[(df["Datetime"] >= start_datetime) & (df["Datetime"] <= end_datetime)]
            if df.empty:
                return pd.DataFrame()

            return df[required]

        except Exception as e:
            print(f"Error fetching index minute data: {e}")
            return pd.DataFrame()

    def fetch_akshare_index_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch A-share index daily data via stock_zh_index_daily (Sina)."""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            # symbol like SH000001 -> sh000001
            sina_symbol = symbol.lower()
            if not sina_symbol.startswith(("sh", "sz")):
                # Try to infer: codes starting with 6 are Shanghai (sh), others Shenzhen (sz)
                clean = symbol.replace("sh", "").replace("SH", "").replace("sz", "").replace("SZ", "")
                if clean.startswith("6"):
                    sina_symbol = f"sh{clean}"
                else:
                    sina_symbol = f"sz{clean}"

            df = ak.stock_zh_index_daily(symbol=sina_symbol)

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })

            required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required):
                print(f"Index missing columns: {df.columns.tolist()}")
                return pd.DataFrame()

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df[
                (df["Datetime"] >= pd.Timestamp(start_date)) &
                (df["Datetime"] <= pd.Timestamp(end_date))
            ]

            return df[required]

        except Exception as e:
            print(f"Error fetching index data: {e}")
            return pd.DataFrame()

    def detect_asset_type(self, symbol: str) -> str:
        """Detect asset type from symbol format."""
        clean_upper = symbol.upper()
        clean = clean_upper.replace("SZ", "").replace("SH", "")

        # 6-digit codes — check by prefix, not by SH/SZ (stocks also use those)
        if clean.isdigit() and len(clean) == 6:
            # A-share indices: 000001(上证), 399001(深成), 399006(创业板), 000300(沪深300), etc.
            # They use specific index code ranges, distinct from regular stocks
            if clean_upper.startswith(("SH", "SZ")) and clean in (
                "000001", "399001", "399006", "000300", "000016", "000905",
                "000852", "000688", "000015", "000009", "000002", "000003",
            ):
                return "a_index"
            # ETFs
            if clean.startswith(("159", "510", "511", "512", "513", "515", "516", "518")):
                return "etf"
            # A-shares: 000xxx, 001xxx, 002xxx, 300xxx, 301xxx, 600xxx, 601xxx, 603xxx, 605xxx, 688xxx, 689xxx
            if clean.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689")):
                return "stock"
            # Open-end funds
            if clean.startswith(("004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "020")):
                return "fund"

        return "stock"

    def fetch_provider_data(
        self, data_source: str, symbol: str, timeframe: str, start_date: str, end_date: str,
        start_time: str = "00:00", end_time: str = "23:59"
    ) -> pd.DataFrame:
        """Fetch data through the pluggable provider layer."""
        if not DATA_PROVIDERS_AVAILABLE or FetchRequest is None:
            raise RuntimeError("数据源模块不可用")

        provider_name = "yahoo" if data_source in ("live", "yahoo", "yfinance") else data_source
        if provider_name not in ("yahoo", "qlib"):
            raise ValueError(f"不支持的数据源：{data_source}")

        try:
            start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
        except ValueError as error:
            raise ValueError("日期或时间格式不正确") from error

        if provider_name == "yahoo":
            provider = YahooFinanceProvider()
        else:
            provider = QlibProvider()
        req = FetchRequest(
            symbol=symbol,
            interval=timeframe,
            start_date=start_dt,
            end_date=end_dt,
        )

        # Use provider fetching/normalization but keep QuantAgent's existing tolerance
        # for short windows such as 15 trading days.
        raw = provider._raw_fetch(req)
        return provider._normalize(raw)

    def fetch_akshare_routed_data(
        self, symbol: str, timeframe: str, start_date: str, end_date: str,
        start_time: str = "00:00", end_time: str = "23:59"
    ) -> pd.DataFrame:
        """Route AKShare data fetching based on symbol and timeframe."""
        akshare_timeframe = self.normalize_akshare_timeframe(timeframe)
        asset_type = self.detect_asset_type(symbol)

        # Minute-level data (A-share stocks, ETFs, and indices)
        if akshare_timeframe in ["1m", "5m", "15m", "30m", "60m"]:
            try:
                start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
            except ValueError:
                return pd.DataFrame()

            if asset_type == "stock":
                return self.fetch_akshare_stock_min_data(symbol, akshare_timeframe, start_dt, end_dt)
            elif asset_type == "etf":
                return self.fetch_akshare_etf_min_data(symbol, akshare_timeframe, start_dt, end_dt)
            elif asset_type == "a_index":
                return self.fetch_akshare_index_min_data(symbol, akshare_timeframe, start_dt, end_dt)

        # Daily-level data
        if asset_type == "a_index":
            return self.fetch_akshare_index_data(symbol, start_date, end_date)
        elif asset_type == "fund" or asset_type == "etf":
            return self.fetch_akshare_fund_data(symbol, start_date, end_date)
        else:
            return self.fetch_akshare_stock_data(symbol, start_date, end_date)

    def fetch_data(
        self, symbol: str, timeframe: str, start_date: str, end_date: str,
        start_time: str = "00:00", end_time: str = "23:59", data_source: str = "akshare"
    ) -> pd.DataFrame:
        """Route to the chosen data source, with CN/Yahoo fallback for unstable networks."""
        normalized_source = (data_source or "akshare").strip().lower()
        errors = []

        if normalized_source in ("live", "yahoo", "yfinance"):
            try:
                df = self.fetch_provider_data(
                    normalized_source, symbol, timeframe, start_date, end_date, start_time, end_time
                )
                if df is not None and not df.empty:
                    return df
                errors.append("Yahoo Finance 返回空数据")
            except Exception as error:
                errors.append(f"Yahoo Finance 失败：{error}")

            if self.is_cn_market_symbol(symbol):
                df = self.fetch_akshare_routed_data(symbol, timeframe, start_date, end_date, start_time, end_time)
                if df is not None and not df.empty:
                    print("Yahoo Finance failed; fell back to AKShare successfully.")
                    return df
                errors.append("AKShare 兜底也没有返回数据")

            raise RuntimeError("；".join(errors))

        if normalized_source == "qlib":
            return self.fetch_provider_data(
                normalized_source, symbol, timeframe, start_date, end_date, start_time, end_time
            )

        df = self.fetch_akshare_routed_data(symbol, timeframe, start_date, end_date, start_time, end_time)
        if df is not None and not df.empty:
            return df

        if self.is_cn_market_symbol(symbol):
            errors.append("AKShare 返回空数据，可能是当前网络无法访问新浪/东方财富")
            try:
                fallback = self.fetch_provider_data(
                    "yahoo", symbol, timeframe, start_date, end_date, start_time, end_time
                )
                if fallback is not None and not fallback.empty:
                    print("AKShare failed; fell back to Yahoo Finance successfully.")
                    return fallback
                errors.append("Yahoo Finance 兜底也没有返回数据")
            except Exception as error:
                errors.append(f"Yahoo Finance 兜底失败：{error}")

        if errors:
            raise RuntimeError("；".join(errors))
        return pd.DataFrame()

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str = "1d",
        start_date: str = "", end_date: str = ""
    ) -> Dict[str, Any]:
        """Run full multi-agent analysis pipeline."""
        try:
            self.trading_graph.refresh_llms()
            df_slice = df.tail(45)

            required = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(c in df_slice.columns for c in required):
                return {"success": False, "error": f"Missing columns"}

            df_slice = df_slice.reset_index(drop=True)

            df_slice_dict = {}
            for col in required:
                if col == "Datetime":
                    df_slice_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                else:
                    df_slice_dict[col] = df_slice[col].tolist()

            date_range = ""
            if start_date and end_date:
                date_range = f"{start_date} ~ {end_date}"

            # Use a shared timestamp so both images land in the same directory
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p_image = static_util.generate_kline_image(df_slice_dict, stock_name=asset_name, timeframe=timeframe, date_range=date_range, run_ts=ts)
            t_image = static_util.generate_trend_image(df_slice_dict, stock_name=asset_name, timeframe=timeframe, date_range=date_range, run_ts=ts)

            initial_state = {
                "kline_data": df_slice_dict,
                "trend_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image.get("pattern_image", ""),
                "trend_image": t_image.get("trend_image", ""),
            }

            final_state = self.trading_graph.graph.invoke(initial_state)

            # Surface the per-node event stream collected during the run so
            # the expert output template can render the process timeline.
            collector = self.trading_graph.event_collector
            if collector is not None and not isinstance(
                final_state.get("process_events"), list
            ):
                final_state["process_events"] = list(collector.events)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "data_length": len(df_slice),
                "pattern_image": p_image.get("pattern_image", ""),
                "trend_image": t_image.get("trend_image", ""),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


analyzer = WebTradingAnalyzer()
analysis_jobs = {}
analysis_jobs_lock = threading.Lock()

# Re-apply env credentials to the already-constructed graph's config as well, so
# the running LLM clients see them. (They were already applied to self.config
# before the graph was built; this keeps the graph's copy consistent and picks
# up anything resolved later.)
_apply_env_credentials(analyzer.config)
_apply_env_credentials(analyzer.trading_graph.config)


def _has_report_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    placeholders = {
        "indicator analysis completed.",
        "trend analysis completed.",
        "waiting for analysis",
    }
    return text.lower() not in placeholders


def _latest_number(values: Any):
    if isinstance(values, list) and values:
        try:
            return float(values[-1])
        except (TypeError, ValueError):
            return None
    return None


def build_indicator_fallback_report(final_state: Dict[str, Any]) -> str:
    rsi = _latest_number(final_state.get("rsi"))
    macd = _latest_number(final_state.get("macd"))
    macd_signal = _latest_number(final_state.get("macd_signal"))
    stoch_k = _latest_number(final_state.get("stoch_k"))
    willr = _latest_number(final_state.get("willr"))

    lines = ["### 技术指标分析", ""]
    if rsi is not None:
        if rsi >= 70:
            lines.append(f"- RSI 当前约为 {rsi:.2f}，处于偏热区域，短线需留意回落风险。")
        elif rsi <= 30:
            lines.append(f"- RSI 当前约为 {rsi:.2f}，处于偏冷区域，可能存在技术修复需求。")
        else:
            lines.append(f"- RSI 当前约为 {rsi:.2f}，整体处于中性区间。")
    if macd is not None and macd_signal is not None:
        relation = "强于" if macd >= macd_signal else "弱于"
        lines.append(f"- MACD 当前为 {macd:.4f}，{relation}信号线 {macd_signal:.4f}。")
    if stoch_k is not None:
        lines.append(f"- 随机指标 %K 当前约为 {stoch_k:.2f}，可结合价格位置观察短线动能。")
    if willr is not None:
        lines.append(f"- 威廉指标 %R 当前约为 {willr:.2f}，用于辅助判断超买超卖状态。")
    if len(lines) == 2:
        lines.append("- 本次分析未获得完整技术指标文字报告，请参考下方指标数值表。")
    lines.extend(["", "以上为本地兜底生成的技术指标摘要。"])
    return "\n".join(lines)


def build_trend_fallback_report(final_state: Dict[str, Any]) -> str:
    kline_data = final_state.get("kline_data", {}) or {}
    close_values = kline_data.get("Close") or kline_data.get("close") or []
    trend = "震荡"
    change_text = ""
    try:
        closes = [float(v) for v in close_values if v is not None]
        if len(closes) >= 2:
            change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0
            if change > 1:
                trend = "偏强上行"
            elif change < -1:
                trend = "偏弱下行"
            change_text = f"，区间涨跌幅约 {change:.2f}%"
    except (TypeError, ValueError):
        closes = []

    lines = [
        "### 趋势分析",
        "",
        f"- 根据当前 K 线序列，价格整体表现为{trend}{change_text}。",
        "- 趋势图中的支撑/阻力线可作为观察价格突破或回踩的参考。",
        "- 若价格持续站上阻力区域，短线动能可能增强；若跌破支撑区域，需要留意回撤风险。",
        "",
        "以上为本地兜底生成的趋势摘要。",
    ]
    return "\n".join(lines)


def _auto_calc_trade_params(final_state: Dict[str, Any], decision_direction: str) -> Dict[str, Any]:
    """Compute entry / stop-loss / take-profit / risk-reward when the LLM
    didn't supply them. Uses last close as entry, recent N-bar true range
    to size the stop, and a default 2.0 risk-reward ratio for the target.

    The function never overrides values the LLM already returned — call
    it with the existing parsed values and merge only the missing keys.
    """
    kline = final_state.get("kline_data") or {}
    highs = kline.get("High") or kline.get("high") or []
    lows = kline.get("Low") or kline.get("low") or []
    closes = kline.get("Close") or kline.get("close") or []
    try:
        h = [float(v) for v in highs if v is not None]
        l = [float(v) for v in lows if v is not None]
        c = [float(v) for v in closes if v is not None]
    except (TypeError, ValueError):
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "risk_reward_ratio": None, "auto_calc": False}

    if not c:
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "risk_reward_ratio": None, "auto_calc": False}

    entry = round(c[-1], 4)
    # True range over the last 14 bars (or whatever's available). Fall back
    # to 2% of price if we don't have enough data for a meaningful range.
    n = min(14, len(c))
    if n >= 2 and len(h) >= n and len(l) >= n:
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(-n, 0)]
        atr = sum(trs) / len(trs) if trs else entry * 0.02
    else:
        atr = entry * 0.02
    # Stop at 1.5 ATR, target at 2x the stop distance (2:1 RR).
    stop_distance = round(atr * 1.5, 4)
    target_distance = round(stop_distance * 2.0, 4)

    direction = (decision_direction or "").strip()
    is_long = any(k in direction for k in ("做多", "LONG", "long", "买入"))
    is_short = any(k in direction for k in ("做空", "SHORT", "short", "卖出"))
    if is_long:
        stop = round(entry - stop_distance, 4)
        target = round(entry + target_distance, 4)
    elif is_short:
        stop = round(entry + stop_distance, 4)
        target = round(entry - target_distance, 4)
    else:
        # 观望 / unknown — still emit symmetric levels so the UI is populated.
        stop = round(entry - stop_distance, 4)
        target = round(entry + target_distance, 4)
    rr = round(target_distance / stop_distance, 2) if stop_distance else None
    return {
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward_ratio": rr,
        "auto_calc": True,
    }


@app.route("/")
def index():
    return render_template(
        "demo_new.html",
        app_variant=APP_VARIANT,
        features=APP_VARIANT_FEATURES,
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        asset = data.get("asset")
        timeframe = data.get("timeframe", "1d")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")
        data_source = data.get("data_source", "akshare")

        phone = (data.get("phone") or "").strip()

        if not all([asset, start_date, end_date]):
            return jsonify({"error": "Missing: asset, start_date, end_date"})

        print(f"Fetching data: {asset} {timeframe} {start_date} to {end_date} via {data_source}")
        try:
            df = analyzer.fetch_data(asset, timeframe, start_date, end_date, start_time, end_time, data_source)
        except Exception as data_error:
            return jsonify({
                "error": f"行情数据获取失败：{format_analysis_error(str(data_error))}",
                "stage": "data_fetch",
            })

        if df.empty:
            return jsonify({"error": f"行情数据获取失败：{asset} 没有返回可用数据", "stage": "data_fetch"})

        print(f"Got {len(df)} rows, running analysis...")
        results = analyzer.run_analysis(df, f"{asset}", timeframe, start_date, end_date)

        if not results.get("success"):
            return jsonify({
                "error": (
                    f"行情数据已获取 {len(df)} 条，但 AI 分析失败："
                    f"{format_analysis_error(results.get('error', 'Analysis failed'))}"
                ),
                "stage": "ai_analysis",
                "data_rows": len(df),
            })

        final_state = results.get("final_state", {})

        # Clean final_trade_decision: strip ```json code fence if present
        raw_decision = final_state.get("final_trade_decision", "")
        parsed = None
        cleaned_decision = ""
        if isinstance(raw_decision, str):
            # Remove ```json or ``` markdown fences
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_decision)
            if match:
                json_str = match.group(1)
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed = None
                # Display the JSON content without fences
                cleaned_decision = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else json_str
            else:
                try:
                    parsed = json.loads(raw_decision)
                    cleaned_decision = json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    cleaned_decision = raw_decision

        # Extract structured decision fields
        decision_direction = ""
        entry_price = None
        stop_loss = None
        take_profit = None
        if parsed:
            decision_direction = parsed.get("决策", parsed.get("direction", parsed.get("decision", "")))
            entry_price = parsed.get("入场价格", parsed.get("entry_price"))
            stop_loss = parsed.get("止损价格", parsed.get("stop_loss"))
            take_profit = parsed.get("止盈价格", parsed.get("take_profit"))
            # The LLM sometimes returns 0 / 0.0 as a placeholder for "unknown".
            # Treat those the same as missing so the auto-calc kicks in.
            if entry_price in (0, 0.0, "", "0", "0.0"):
                entry_price = None
            if stop_loss in (0, 0.0, "", "0", "0.0"):
                stop_loss = None
            if take_profit in (0, 0.0, "", "0", "0.0"):
                take_profit = None

        # Auto-fill missing trade params from the OHLC data so the UI
        # never shows "—" when there's enough price history to compute
        # a sensible level. The flag `auto_calc: True` is surfaced to the
        # template so it can mark the values as derived.
        trade_params_auto = False
        if entry_price is None or stop_loss is None or take_profit is None:
            auto = _auto_calc_trade_params(final_state, decision_direction)
            if auto.get("auto_calc"):
                trade_params_auto = True
                if entry_price is None:
                    entry_price = auto["entry_price"]
                if stop_loss is None:
                    stop_loss = auto["stop_loss"]
                if take_profit is None:
                    take_profit = auto["take_profit"]

        # Same treatment for risk-reward: derive from the (possibly
        # auto-filled) entry/stop/target if the LLM didn't return one.
        risk_reward_ratio = final_state.get("risk_reward_ratio")
        if risk_reward_ratio is None and entry_price and stop_loss and take_profit:
            try:
                if abs(float(entry_price) - float(stop_loss)) > 0:
                    risk_reward_ratio = round(
                        abs(float(take_profit) - float(entry_price))
                        / abs(float(entry_price) - float(stop_loss)),
                        2,
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                risk_reward_ratio = None

        indicator_report = final_state.get("indicator_report", "")
        trend_report = final_state.get("trend_report", "")
        if not _has_report_text(indicator_report):
            indicator_report = build_indicator_fallback_report(final_state)
        if not _has_report_text(trend_report):
            trend_report = build_trend_fallback_report(final_state)

        # Pull process events off the EventCollector if the trading_graph
        # didn't already surface them through the state. This keeps the
        # `process_events` key populated for the academic output template
        # even on legacy invocations.
        process_events: list = final_state.get("process_events") or []
        if not process_events:
            collector = getattr(analyzer, "trading_graph", None)
            collector = getattr(collector, "event_collector", None) if collector is not None else None
            if collector is not None and hasattr(collector, "events"):
                process_events = list(collector.events)

        full_response = {
            "success": True,
            "redirect": "/output",
            "mode": APP_VARIANT,
            "app_variant": APP_VARIANT,
            "asset_name": results["asset_name"],
            "timeframe": timeframe,
            "data_length": results["data_length"],
            "indicator_report": indicator_report,
            "pattern_report": final_state.get("pattern_report", ""),
            "trend_report": trend_report,
            "final_trade_decision": cleaned_decision,
            "decision_direction": decision_direction,
            "entry_price": entry_price or final_state.get("entry_price"),
            "stop_loss": stop_loss or final_state.get("stop_loss"),
            "take_profit": take_profit or final_state.get("take_profit"),
            "indicators": {
                "rsi": final_state.get("rsi"),
                "macd": final_state.get("macd"),
                "macd_signal": final_state.get("macd_signal"),
                "macd_hist": final_state.get("macd_hist"),
                "stoch_k": final_state.get("stoch_k"),
                "stoch_d": final_state.get("stoch_d"),
                "roc": final_state.get("roc"),
                "willr": final_state.get("willr"),
            },
            "pattern_chart": results.get("pattern_image", ""),
            "trend_chart": results.get("trend_image", ""),
            # Cross-Checker outputs (always present, defaults to 0.5 if missing)
            "consensus_score": final_state.get("consensus_score", 0.5),
            "conflicts": final_state.get("conflicts", "") or "",
            "key_points_summary": final_state.get("key_points_summary", "") or "",
            "core_view": final_state.get("core_view", "") or "",
            # Decision Maker structured outputs
            "justification": final_state.get("justification", "") or "",
            "forecast_horizon": final_state.get("forecast_horizon", "") or "",
            "risk_reward_ratio": risk_reward_ratio,
            "trade_params_auto": trade_params_auto,
            "decision": final_state.get("decision", "") or decision_direction,
            # Process event stream (academic output template)
            "process_events": process_events,
        }

        # Persist to the user's local analysis history (best-effort).
        if phone:
            try:
                history_store.save_record(
                    phone,
                    full_response,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                )
            except Exception as history_error:  # noqa: BLE001
                print(f"Warning: could not save analysis history: {history_error}")

        # Return as both response and full_results for redirect handling.
        # The desktop WebView2 shell navigates to /output after a successful
        # analyze; we stash the (potentially large) full_response in the
        # in-memory output cache and pass the key in the redirect URL so the
        # output page can re-render with the real data instead of the empty
        # placeholder template.
        result_key = _stash_output_result(full_response)

        return jsonify({
            "redirect": f"/output?key={result_key}",
            "full_results": full_response
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": format_analysis_error(str(e))})


def run_analysis_job(job_id: str, data: Dict[str, Any]):
    """Run the existing synchronous analysis in a background thread."""
    try:
        with app.test_request_context("/api/analyze", method="POST", json=data):
            response = analyze()
            result = response.get_json()
        with analysis_jobs_lock:
            analysis_jobs[job_id] = {"status": "completed", "result": result}
    except Exception as error:
        with analysis_jobs_lock:
            analysis_jobs[job_id] = {"status": "failed", "error": format_analysis_error(str(error))}


@app.route("/api/analyze/start", methods=["POST"])
def start_analysis():
    """Start a long-running analysis without holding the WebView connection."""
    data = request.get_json() or {}
    if not all([data.get("asset"), data.get("start_date"), data.get("end_date")]):
        return jsonify({"error": "Missing: asset, start_date, end_date"}), 400

    job_id = uuid.uuid4().hex
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {"status": "running"}

    threading.Thread(
        target=run_analysis_job,
        args=(job_id, data),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/analyze/status/<job_id>")
def analysis_status(job_id: str):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Analysis job not found"}), 404
    return jsonify(job)


@app.route("/api/asset-info/<symbol>")
def asset_info(symbol):
    """Return detected asset type info."""
    asset_type = analyzer.detect_asset_type(symbol)
    return jsonify({"symbol": symbol, "type": asset_type})


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "desktop": os.environ.get("QUANTAGENT_DESKTOP") == "1",
        "token": os.environ.get("QUANTAGENT_DESKTOP_TOKEN", ""),
        "port": int(os.environ.get("QUANTAGENT_FLASK_PORT", "5000")),
        "backend_root": _BASE_DIR,
        "cwd": os.getcwd(),
    })


@app.route("/api/test-data", methods=["POST"])
def test_data():
    """Test data fetching without running analysis."""
    try:
        data = request.get_json()
        asset = data.get("asset")
        timeframe = data.get("timeframe", "1d")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")
        data_source = data.get("data_source", "akshare")

        if not all([asset, start_date, end_date]):
            return jsonify({"error": "Missing: asset, start_date, end_date"})

        print(f"Fetching data: {asset} {timeframe} {start_date} to {end_date} via {data_source}")
        df = analyzer.fetch_data(asset, timeframe, start_date, end_date, start_time, end_time, data_source)

        if df.empty:
            return jsonify({"error": f"No data for {asset}"})

        # Return data summary
        return jsonify({
            "success": True,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "first_row": df.head(1).to_dict("records")[0] if len(df) > 0 else None,
            "last_row": df.tail(1).to_dict("records")[0] if len(df) > 0 else None,
            "date_range": {
                "start": str(df["Datetime"].min()),
                "end": str(df["Datetime"].max()),
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


@app.route("/output")
def output():
    # Pick the output template that matches the application variant. The
    # trader variant uses a K-line-first layout; the academic variant adds
    # the process-timeline + PDF export + backtest surfaces.
    force_full = request.args.get("view") == "full"
    if APP_VARIANT == "academic" and not force_full:
        output_template = "output_expert.html"
    else:
        output_template = "output_trader.html"

    results_key = request.args.get("key")
    if results_key:
        cached = _pop_output_result(results_key)
        if cached:
            return render_template(
                output_template,
                results=cached,
                app_variant=APP_VARIANT,
                features=APP_VARIANT_FEATURES,
            )

    results = request.args.get("results")
    if results:
        try:
            results = urllib.parse.unquote(results)
            parsed_results = json.loads(results)
            # Backfill the mode/justification/forecast/etc. fields the
            # earlier client-side `JSON.stringify(full_response)` may have
            # already produced — we just need to plumb them through.
            return render_template(
                output_template,
                results=parsed_results,
                app_variant=APP_VARIANT,
                features=APP_VARIANT_FEATURES,
            )
        except Exception as e:
            print(f"Error parsing results: {e}")

    empty_results = {
        "asset_name": "",
        "timeframe": "1d",
        "data_length": 0,
        "indicator_report": "",
        "pattern_report": "",
        "trend_report": "",
        "final_trade_decision": "",
        "decision_direction": "",
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "indicators": {},
        "pattern_chart": "",
        "trend_chart": "",
        "consensus_score": 0.5,
        "conflicts": "",
        "key_points_summary": "",
        "core_view": "",
        "justification": "",
        "forecast_horizon": "",
        "risk_reward_ratio": None,
        "trade_params_auto": False,
        "process_events": [],
        "mode": APP_VARIANT,
    }
    return render_template(
        output_template,
        results=empty_results,
        app_variant=APP_VARIANT,
        features=APP_VARIANT_FEATURES,
    )


@app.route("/api/output/prepare", methods=["POST"])
def prepare_output():
    """Stash a results payload in the output cache and return a redirect key.

    Used by the JS clients that already have the full result in memory
    (analyze flow, history detail) and want to navigate to /output without
    re-running the analysis. Without this shim the output page would have
    no way to find the (potentially very large) results payload.
    """
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "missing or invalid results payload"}), 400
    key = _stash_output_result(payload)
    return jsonify({"key": key, "redirect": f"/output?key={key}"})


@app.route("/api/export", methods=["POST"])
def export_report():
    """Render a PDF of the most recent analysis result.

    The client posts ``{ results: <full_response dict> }`` (the same JSON
    that ``/api/analyze`` returned). The endpoint is academic-only — the
    trader variant returns 403 so the UI never even shows the button.
    """
    if APP_VARIANT != "academic":
        return jsonify({"error": "PDF 导出仅高校版可用"}), 403
    if not REPORT_EXPORT_AVAILABLE or report_export is None:
        return jsonify({"error": "请先安装 reportlab：pip install reportlab"}), 501

    payload = request.get_json(silent=True) or {}
    results = payload.get("results") or {}
    if not isinstance(results, dict) or not results.get("asset_name"):
        return jsonify({"error": "缺少分析结果（请先运行一次分析）"}), 400

    try:
        pdf_bytes = report_export.build_analysis_pdf(
            results,
            equity_chart=payload.get("equity_chart"),
            backtest_metrics=payload.get("backtest_metrics"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"PDF export failed: {exc}")
        return jsonify({"error": f"PDF 生成失败：{exc}"}), 500

    asset = (results.get("asset_name") or "analysis").replace("/", "_")
    filename = f"quantagent-{asset}-{results.get('timeframe', '1d')}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    """Signal-driven backtest (academic only)."""
    if APP_VARIANT != "academic":
        return jsonify({"error": "回测功能仅高校版可用"}), 403

    try:
        import backtest as backtest_mod  # local import — optional dep
    except ImportError as exc:
        return jsonify({"error": f"回测模块未就绪：{exc}"}), 501

    payload = request.get_json(silent=True) or {}
    asset = (payload.get("asset") or "").strip()
    timeframe = (payload.get("timeframe") or "1d").strip()
    start_date = (payload.get("start_date") or "").strip()
    end_date = (payload.get("end_date") or "").strip()
    data_source = (payload.get("data_source") or "akshare").strip()
    if not asset or not start_date or not end_date:
        return jsonify({"error": "缺少 asset / start_date / end_date"}), 400

    initial_capital = float(payload.get("initial_capital") or 100000)
    hold_bars = int(payload.get("hold_bars") or 5)
    signal_mode = (payload.get("signal_mode") or "decision").strip().lower()
    if signal_mode not in ("decision", "always_long", "always_short"):
        signal_mode = "decision"

    start_time = (payload.get("start_time") or "00:00").strip()
    end_time = (payload.get("end_time") or "23:59").strip()

    try:
        df = analyzer.fetch_data(
            asset,
            timeframe,
            start_date,
            end_date,
            start_time,
            end_time,
            data_source,
        )
    except Exception as exc:
        return jsonify({"error": f"行情数据获取失败：{exc}", "stage": "data_fetch"}), 500
    if df is None or len(df) == 0:
        return jsonify({"error": "回测无可用数据", "stage": "data_fetch"}), 400

    try:
        metrics, equity_b64 = backtest_mod.run_simple_backtest(
            df,
            signal_mode=signal_mode,
            hold_bars=hold_bars,
            initial_capital=initial_capital,
        )
    except Exception as exc:
        return jsonify({"error": f"回测执行失败：{exc}"}), 500

    return jsonify({"metrics": metrics, "equity_chart": equity_b64})


@app.route("/api/custom-assets")
def get_custom_assets():
    """Return list of custom assets saved on the server."""
    try:
        assets_file = Path("data/custom_assets.json")
        if assets_file.exists():
            with open(assets_file, "r", encoding="utf-8") as f:
                return jsonify({"custom_assets": json.load(f)})
        return jsonify({"custom_assets": []})
    except Exception as e:
        print(f"Error loading custom assets: {e}")
        return jsonify({"custom_assets": []})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol to server."""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "No symbol provided"})

        assets_file = Path("data/custom_assets.json")
        assets_file.parent.mkdir(exist_ok=True)

        custom_assets = []
        if assets_file.exists():
            with open(assets_file, "r", encoding="utf-8") as f:
                custom_assets = json.load(f)

        if symbol not in custom_assets:
            custom_assets.append(symbol)
            with open(assets_file, "w", encoding="utf-8") as f:
                json.dump(custom_assets, f)

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        print(f"Error saving custom asset: {e}")
        return jsonify({"success": False, "error": str(e)})


def _request_phone() -> str:
    """Pull the phone identifier from JSON body or query string."""
    payload = request.get_json(silent=True) or {}
    phone = payload.get("phone") or request.args.get("phone") or ""
    return str(phone).strip()


@app.route("/api/history", methods=["GET"])
def get_history():
    """List analysis history summaries for the current phone."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户", "records": []}), 400
    asset = request.args.get("asset") or None
    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    records = history_store.list_records(phone, asset=asset, limit=limit)
    return jsonify({"records": records})


@app.route("/api/history/assets", methods=["GET"])
def get_history_assets():
    """List distinct assets the current phone has analysed, for filtering."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户", "assets": []}), 400
    return jsonify({"assets": history_store.list_asset_buttons(phone)})


@app.route("/api/history/<int:record_id>", methods=["GET"])
def get_history_detail(record_id: int):
    """Return one full analysis record for the current phone."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户"}), 400
    record = history_store.get_record(phone, record_id)
    if record is None:
        return jsonify({"error": "记录不存在或无权查看"}), 404
    return jsonify(record)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id: int):
    """Delete one analysis record for the current phone."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户"}), 400
    removed = history_store.delete_record(phone, record_id)
    if not removed:
        return jsonify({"error": "记录不存在或无权删除"}), 404
    return jsonify({"success": True})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear all analysis history for the current phone."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户"}), 400
    removed = history_store.clear_records(phone)
    return jsonify({"success": True, "removed": removed})


@app.route("/api/history/export", methods=["GET"])
def export_history():
    """Export all history for the current phone as JSON."""
    phone = _request_phone()
    if not phone:
        return jsonify({"error": "未识别到登录用户"}), 400
    records = history_store.export_records(phone)
    return jsonify({"phone": phone, "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "records": records})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """Update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")
        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
        if provider == "custom":
            if data.get("mode"):
                analyzer.config["custom_mode"] = data.get("mode", "openai")
            if data.get("base_url"):
                analyzer.config["custom_base_url"] = normalize_custom_base_url_for_mode(
                    data.get("base_url", ""),
                    analyzer.config.get("custom_mode", "openai"),
                )
            if data.get("endpoint_url"):
                analyzer.config["custom_endpoint_url"] = data.get("endpoint_url", "").strip()
            if data.get("model"):
                analyzer.config["custom_model"] = data.get("model", "").strip()
            if data.get("headers_template"):
                analyzer.config["custom_headers_template"] = data.get("headers_template")
            if data.get("body_template"):
                analyzer.config["custom_body_template"] = data.get("body_template")
            if data.get("response_path"):
                analyzer.config["custom_response_path"] = data.get("response_path")
        elif provider == "trial":
            analyzer.config["agent_llm_model"] = analyzer.config.get("trial_model", "deepseek-chat")
            analyzer.config["graph_llm_model"] = analyzer.config.get("trial_model", "deepseek-chat")
        elif provider == "anthropic":
            # 火山方舟（进阶版）走 Anthropic 兼容协议，model/base_url 从环境变量读取
            ark_model = os.environ.get("ANTHROPIC_MODEL", "ark-code-latest").strip()
            ark_base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
            analyzer.config["agent_llm_model"] = ark_model
            analyzer.config["graph_llm_model"] = ark_model
            if ark_base:
                analyzer.config["base_url"] = ark_base
            ark_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if ark_key:
                analyzer.config["anthropic_api_key"] = ark_key
        return jsonify({"success": True, "provider": provider})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """Update API key for a provider."""
    try:
        data = request.get_json()
        api_key = data.get("api_key", "").strip()
        provider = data.get("provider", "openai")
        custom_mode = data.get("mode", "openai")
        base_url = normalize_custom_base_url_for_mode(data.get("base_url", ""), custom_mode)
        endpoint_url = data.get("endpoint_url", "").strip()
        model = data.get("model", "").strip()

        if not api_key:
            return jsonify({"success": False, "error": "No API key provided"})

        if provider == "custom" and custom_mode in ("openai", "anthropic") and not base_url:
            return jsonify({"success": False, "error": "请填写通用 AI 的 Base URL"})
        if provider == "custom" and custom_mode == "http" and not endpoint_url:
            return jsonify({"success": False, "error": "请填写高级自定义接口的完整请求 URL"})

        if provider == "sohu":
            analyzer.validate_sohu_api_key(api_key)
        custom_options = None
        if provider == "custom":
            custom_options = {
                "mode": custom_mode,
                "endpoint_url": endpoint_url,
                "headers_template": data.get("headers_template", ""),
                "body_template": data.get("body_template", ""),
                "response_path": data.get("response_path", ""),
            }
        analyzer.update_api_key(api_key, provider, base_url, model, custom_options)
        return jsonify({"success": True, "provider": provider})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/clear-api-key", methods=["POST"])
def clear_api_key():
    """Clear API key for a provider without falling back to the previous model instance."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")
        if provider == "trial":
            return jsonify({
                "success": False,
                "error": "使用内置模型(Deepseek)使用内置 Key，不能清除。"
            })
        key_map = {
            "openai": ("api_key", "OPENAI_API_KEY"),
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "qwen": ("qwen_api_key", "DASHSCOPE_API_KEY"),
            "minimax": ("minimax_api_key", "MINIMAX_API_KEY"),
            "sohu": ("sohu_api_key", "SOHU_API_KEY"),
            "trial": ("trial_api_key", "QUANTAGENT_TRIAL_API_KEY"),
            "custom": ("custom_api_key", "CUSTOM_OPENAI_API_KEY"),
        }
        key_name, env_name = key_map.get(provider, ("api_key", "OPENAI_API_KEY"))
        analyzer.config[key_name] = ""
        analyzer.trading_graph.config[key_name] = ""
        os.environ.pop(env_name, None)
        return jsonify({"success": True, "provider": provider})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """Return masked API key status for a provider."""
    try:
        provider = request.args.get("provider", "openai")
        key_map = {
            "openai": "api_key",
            "anthropic": "anthropic_api_key",
            "qwen": "qwen_api_key",
            "minimax": "minimax_api_key",
            "sohu": "sohu_api_key",
            "trial": "trial_api_key",
            "custom": "custom_api_key",
        }
        key_name = key_map.get(provider, "api_key")
        api_key = analyzer.config.get(key_name, "")

        if api_key and api_key not in ("sk-", ""):
            masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            response = {"has_key": True, "masked_key": masked}
            if request.args.get("reveal") == "1" and provider != "trial":
                response["api_key"] = api_key
            return jsonify(response)
        return jsonify({"has_key": False})
    except Exception as e:
        return jsonify({"error": str(e), "has_key": False})


if __name__ == "__main__":
    Path(_BASE_DIR, "templates").mkdir(exist_ok=True)
    Path(_BASE_DIR, "static").mkdir(exist_ok=True)
    desktop_mode = os.environ.get("QUANTAGENT_DESKTOP") == "1"
    if desktop_mode:
        from waitress import serve

        port = int(os.environ.get("QUANTAGENT_FLASK_PORT", "5000"))
        serve(app, host="127.0.0.1", port=port, threads=8)
    else:
        app.run(debug=True, use_reloader=True, host="127.0.0.1", port=5000)
