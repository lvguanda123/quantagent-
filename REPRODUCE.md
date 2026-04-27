# QuantAgent 复现指南

本文档说明如何保存当前工作状态，以及如何让其他人完整复现你的代码框架。

---

## 一、保存当前工作

当前所有修改尚未提交。执行以下命令保存：

```bash
git add .
git commit -m "feat: save current working QuantAgent state"
git push origin main
```

> 注意：`.gitignore` 已更新，虚拟环境（`venv/`, `envs/`）、缓存、数据文件等不会被提交。

---

## 二、他人一键复现

### 方法 1：Git Clone + 自动脚本

```bash
# 1. 克隆仓库
git clone <你的仓库地址>
cd quantagent

# 2. 运行安装脚本（Windows）
install.bat

# 或 Linux/macOS
chmod +x setup.sh && ./setup.sh

# 3. 配置 API Key（复制模板后填入真实值）
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key

# 4. 启动
python web_interface.py
```

### 方法 2：手动步骤

```bash
# 1. 克隆仓库
git clone <你的仓库地址>
cd quantagent

# 2. 创建虚拟环境
python -m venv venv          # Windows
source venv/bin/activate     # Linux/macOS

# 3. 安装依赖
pip install -r requirements.txt

# 注意：TA-Lib 可能需要先安装 C 库
# Windows: 下载 https://github.com/ta-lib/ta-lib/releases
# macOS:   brew install ta-lib
# Ubuntu:  sudo apt-get install -y ta-lib

# 4. 配置环境变量
# 方式 A：复制 .env.example 为 .env 并填入 API Key
# 方式 B：直接在 Web 界面输入

# 5. 启动 Web 界面
python web_interface.py
# 浏览器访问 http://127.0.0.1:5000
```

---

## 三、复现清单

确保以下条件满足：

- [ ] Python 3.11 已安装
- [ ] Git 仓库已 clone
- [ ] 虚拟环境已创建并激活
- [ ] `requirements.txt` 所有依赖安装成功
- [ ] TA-Lib C 库已安装（如遇到安装失败）
- [ ] 至少配置了一个 LLM Provider 的 API Key
- [ ] `web_interface.py` 启动成功，端口 5000 未被占用

---

## 四、支持的 LLM Provider

| Provider | 环境变量 | 默认模型 |
|----------|---------|---------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o / gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-haiku-4-5-20251001 |
| Qwen | `DASHSCOPE_API_KEY` | qwen3-max / qwen3-vl-plus |
| MiniMax | `MINIMAX_API_KEY` | MiniMax-M2.7 |

> 系统需要支持图像输入的 LLM（Pattern Agent 和 Trend Agent 会分析 K 线图）。

---

## 五、常见问题

### TA-Lib 安装失败

TA-Lib 需要先安装底层 C 库：
- **Windows**: 从 [ta-lib/releases](https://github.com/ta-lib/ta-lib/releases) 下载并安装
- **macOS**: `brew install ta-lib`
- **Ubuntu/Debian**: `sudo apt-get install -y ta-lib`

然后执行：`pip install TA-Lib`

### 端口被占用

修改 `web_interface.py` 中的端口号，或关闭占用 5000 端口的进程。

### LLM API 调用失败

1. 确认 API Key 正确
2. 确认账户余额充足
3. 检查网络连接（Qwen 服务端在新加坡，可能有延迟）
