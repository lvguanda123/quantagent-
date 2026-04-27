"""
QuantAgent 使用示例 - 使用 Qlib 数据和 Qwen 模型

此脚本演示如何：
1. 从本地 Qlib 数据目录加载数据
2. 使用阿里云 Qwen 模型进行分析
3. 运行完整的 QuantAgent 多 Agent 分析流程
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# 设置 Qwen API 配置
os.environ["DASHSCOPE_API_KEY"] = "sk-sp-c7815c43449c42a598cd8717b9b3c053"

from default_config import DEFAULT_CONFIG
from trading_graph import TradingGraph
import pandas as pd


def load_qlib_sample_data(symbol: str = "000001", start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
    """
    从 Qlib 数据目录加载示例数据

    Args:
        symbol: 股票代码（如 000001 代表平安银行）
        start_date: 开始日期
        end_date: 结束日期
    """
    qlib_data_dir = Path(r"C:\Users\Administrator\.qlib\qlib_data\cn_data_rolling")
    freq_dir = qlib_data_dir / "day"
    features_dir = freq_dir / "features"
    symbol_dir = features_dir / symbol

    if not symbol_dir.exists():
        print(f"未找到股票 {symbol} 的数据目录")
        return None

    # 读取 OHLCV 数据
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    field_data = {}

    for field in fields:
        field_file = symbol_dir / f"{field}.pkl"
        if field_file.exists():
            import pickle
            with open(field_file, "rb") as f:
                field_data[field] = pickle.load(f)

    if not field_data:
        print(f"未找到股票 {symbol} 的任何数据字段")
        return None

    # 创建 DataFrame
    df = pd.DataFrame({
        "Open": field_data.get("$open", []),
        "High": field_data.get("$high", []),
        "Low": field_data.get("$low", []),
        "Close": field_data.get("$close", []),
        "Volume": field_data.get("$volume", [])
    })

    # 读取日历文件
    calendar_file = freq_dir / "calendars.txt"
    if calendar_file.exists():
        with open(calendar_file, "r") as f:
            dates = [line.strip() for line in f.readlines()]
        df.index = pd.to_datetime(dates[:len(df)])

    # 重置索引并添加 Datetime 列
    df = df.reset_index()
    df.columns = ["Datetime"] + list(df.columns[1:])
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # 按日期过滤
    df = df[(df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)]

    print(f"成功加载 {len(df)} 条数据")
    print(f"日期范围：{df['Datetime'].min()} 到 {df['Datetime'].max()}")

    return df


def prepare_kline_data(df: pd.DataFrame) -> dict:
    """
    将 DataFrame 转换为 QuantAgent 所需的字典格式
    """
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]

    df_dict = {}
    for col in required_columns:
        if col == "Datetime":
            df_dict[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_dict[col] = df[col].tolist()

    return df_dict


def main():
    print("=" * 60)
    print("QuantAgent - Qlib 数据 + Qwen 模型分析示例")
    print("=" * 60)

    # 步骤 1: 加载 Qlib 数据
    print("\n[步骤 1] 加载 Qlib 数据...")
    df = load_qlib_sample_data(symbol="000001", start_date="2024-01-01", end_date="2024-12-31")

    if df is None or df.empty:
        print("无法加载数据，请检查 Qlib 数据目录配置")
        return

    # 步骤 2: 初始化 TradingGraph（使用 Qwen 配置）
    print("\n[步骤 2] 初始化 TradingGraph（Qwen 模型）...")
    config = DEFAULT_CONFIG.copy()
    config["agent_llm_provider"] = "qwen"
    config["graph_llm_provider"] = "qwen"
    config["agent_llm_model"] = "qwen3.6-plus"
    config["graph_llm_model"] = "qwen3-vl-plus"
    config["qwen_base_url"] = "https://coding.dashscope.aliyuncs.com/v1"

    trading_graph = TradingGraph(config=config)
    print(f"  Agent LLM: {config['agent_llm_model']}")
    print(f"  Graph LLM: {config['graph_llm_model']}")

    # 步骤 3: 准备数据
    print("\n[步骤 3] 准备 K 线数据...")
    df_slice = df.tail(45)  # 取最近 45 条数据
    kline_data = prepare_kline_data(df_slice)

    # 步骤 4: 生成图表（可选，加速分析）
    print("\n[步骤 4] 生成图表...")
    import static_util

    try:
        p_image = static_util.generate_kline_image(kline_data)
        t_image = static_util.generate_trend_image(kline_data)
        print("  ✓ 图表生成成功")
    except Exception as e:
        print(f"  图表生成失败：{e}")
        p_image = {"pattern_image": None}
        t_image = {"trend_image": None}

    # 步骤 5: 运行分析
    print("\n[步骤 5] 运行 QuantAgent 分析...")
    initial_state = {
        "kline_data": kline_data,
        "analysis_results": None,
        "messages": [],
        "time_frame": "1day",
        "stock_name": "平安银行 (000001)",
        "pattern_image": p_image.get("pattern_image"),
        "trend_image": t_image.get("trend_image"),
    }

    try:
        final_state = trading_graph.graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print("分析结果")
        print("=" * 60)

        print("\n【技术指标分析】")
        print(final_state.get("indicator_report", "N/A")[:500] + "...")

        print("\n【形态识别分析】")
        print(final_state.get("pattern_report", "N/A")[:500] + "...")

        print("\n【趋势分析】")
        print(final_state.get("trend_report", "N/A")[:500] + "...")

        print("\n【最终交易决策】")
        print(final_state.get("final_trade_decision", "N/A"))

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

    except Exception as e:
        print(f"分析过程中出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
