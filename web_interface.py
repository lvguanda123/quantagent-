"""
Web Interface for QuantAgent Trading Analysis.
Supports AKShare: A-shares, funds, ETF minute data.
"""

import json
import os
import re
import requests
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

import static_util
from trading_graph import TradingGraph

# AKShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: AKShare not available")

app = Flask(__name__, static_folder="static", static_url_path="/static")


class WebTradingAnalyzer:
    def __init__(self):
        from default_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()
        self.trading_graph = TradingGraph(config=self.config)

    def fetch_akshare_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch A-share stock daily data via stock_zh_a_hist (东方财富)."""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            # stock_zh_a_hist uses pure 6-digit code
            clean_symbol = symbol.replace("sz", "").replace("SZ", "").replace("sh", "").replace("SH", "")

            df = ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Standardize columns
            df = df.rename(columns={
                "日期": "Datetime",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume"
            })

            required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required):
                print(f"Missing columns: {df.columns.tolist()}")
                return pd.DataFrame()

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            return df[required]

        except Exception as e:
            print(f"Error fetching A-share stock: {e}")
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

    def fetch_data(
        self, symbol: str, timeframe: str, start_date: str, end_date: str,
        start_time: str = "00:00", end_time: str = "23:59"
    ) -> pd.DataFrame:
        """Route to correct data fetcher based on symbol and timeframe."""
        asset_type = self.detect_asset_type(symbol)

        # Minute-level data (A-share stocks, ETFs, and indices)
        if timeframe in ["1m", "5m", "15m", "30m", "60m"]:
            try:
                start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
            except ValueError:
                return pd.DataFrame()

            if asset_type == "stock":
                return self.fetch_akshare_stock_min_data(symbol, timeframe, start_dt, end_dt)
            elif asset_type == "etf":
                return self.fetch_akshare_etf_min_data(symbol, timeframe, start_dt, end_dt)
            elif asset_type == "a_index":
                return self.fetch_akshare_index_min_data(symbol, timeframe, start_dt, end_dt)

        # Daily-level data
        if asset_type == "a_index":
            return self.fetch_akshare_index_data(symbol, start_date, end_date)
        elif asset_type == "fund" or asset_type == "etf":
            return self.fetch_akshare_fund_data(symbol, start_date, end_date)
        else:
            return self.fetch_akshare_stock_data(symbol, start_date, end_date)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str = "1d",
        start_date: str = "", end_date: str = ""
    ) -> Dict[str, Any]:
        """Run full multi-agent analysis pipeline."""
        try:
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


@app.route("/")
def index():
    return render_template("demo_new.html")


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

        if not all([asset, start_date, end_date]):
            return jsonify({"error": "Missing: asset, start_date, end_date"})

        # Fetch data
        print(f"Fetching data: {asset} {timeframe} {start_date} to {end_date}")
        df = analyzer.fetch_data(asset, timeframe, start_date, end_date, start_time, end_time)

        if df.empty:
            return jsonify({"error": f"No data for {asset}"})

        print(f"Got {len(df)} rows, running analysis...")
        results = analyzer.run_analysis(df, f"{asset}", timeframe, start_date, end_date)

        if not results.get("success"):
            return jsonify({"error": results.get("error", "Analysis failed")})

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

        full_response = {
            "success": True,
            "redirect": "/output",
            "asset_name": results["asset_name"],
            "data_length": results["data_length"],
            "indicator_report": final_state.get("indicator_report", ""),
            "pattern_report": final_state.get("pattern_report", ""),
            "trend_report": final_state.get("trend_report", ""),
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
        }

        # Return as both response and full_results for redirect handling
        return jsonify({
            "redirect": "/output",
            "full_results": full_response
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/asset-info/<symbol>")
def asset_info(symbol):
    """Return detected asset type info."""
    asset_type = analyzer.detect_asset_type(symbol)
    return jsonify({"symbol": symbol, "type": asset_type})


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


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

        if not all([asset, start_date, end_date]):
            return jsonify({"error": "Missing: asset, start_date, end_date"})

        print(f"Fetching data: {asset} {timeframe} {start_date} to {end_date}")
        df = analyzer.fetch_data(asset, timeframe, start_date, end_date, start_time, end_time)

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
    results = request.args.get("results")
    if results:
        try:
            results = urllib.parse.unquote(results)
            return render_template("output.html", results=json.loads(results))
        except Exception as e:
            print(f"Error parsing results: {e}")

    return render_template("output.html", results={
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
    })


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


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """Update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")
        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
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

        if not api_key:
            return jsonify({"success": False, "error": "No API key provided"})

        analyzer.update_api_key(api_key, provider)
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
        }
        key_name = key_map.get(provider, "api_key")
        api_key = analyzer.config.get(key_name, "")

        if api_key and api_key not in ("sk-", ""):
            masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            return jsonify({"has_key": True, "masked_key": masked})
        return jsonify({"has_key": False})
    except Exception as e:
        return jsonify({"error": str(e), "has_key": False})


if __name__ == "__main__":
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    app.run(debug=True, host="127.0.0.1", port=5000)
