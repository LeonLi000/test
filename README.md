# ETHUSDT 短线交易策略

基于 Jesse 框架实现的 ETH/USDT 15 分钟短线交易策略，目标是实现年化收益率超过 50%，最大回撤低于 5%，并保持年交易次数超过 50 笔。
本项目提供了完整的目录结构、
策略代码、回测脚本以及 Docker 运行环境，方便直接开始研究或根据实际需求扩展。

## 项目结构

```
eth_trading_strategy/
├── strategies/
│   └── ETHShortTermStrategy.py    # 主策略代码
├── config.py                      # Jesse 配置文件（回测默认参数）
├── backtest.py                    # Python 与 CLI 双模式回测脚本
├── requirements.txt               # 依赖列表
├── Dockerfile                     # Docker 镜像配置
├── docker-compose.yml             # Docker Compose 入口
└── README.md                      # 项目说明
```

## 策略概述

### 核心特点
- **交易对**: ETHUSDT
- **时间框架**: 15 分钟 K 线
- **策略类型**: 多指标复合短线策略
- **风险管理**: 严格的止损止盈机制
- **目标收益**: 年化收益率 > 50%
- **风险控制**: 最大回撤 < 5%
- **交易频率**: 年交易量 > 50 笔

### 技术指标
1. **EMA 指标组合**
   - 快速 EMA(8): 短期趋势判断
   - 慢速 EMA(21): 中期趋势确认
   - 趋势 EMA(55): 长期趋势过滤
2. **动量指标**
   - RSI(14): 超买超卖判断
   - MACD(12,26,9): 趋势转换信号
3. **波动性指标**
   - 布林带(20,2): 价格通道分析
   - ATR(14): 动态止损设置
4. **成交量分析**
   - 成交量 SMA(20): 确认信号强度

### 交易逻辑摘要
- **做多条件**
  1. EMA 快线(8) > EMA 慢线(21)
  2. 当前价格 > 趋势 EMA(55)
  3. RSI 位于 30-70 之间（避免极端区域）
  4. MACD 金叉或 MACD 线 > 信号线
  5. 价格处于布林带下轨之上、中轨之下
  6. 成交量放大（> 20 期均量的 1.2 倍）
  7. 同方向信号冷却时间 ≥ 4 根 K 线
- **做空条件** 与做多条件对称
- **仓位管理**
  - 每次交易使用 2% 资金，基于市价计算下单数量
  - 固定止损 1.5% 与 ATR×1.5 二者取较小值
  - 固定止盈 3%，并结合 ATR 动态追踪止损

## 快速开始

### 1. 安装依赖并回测
```bash
cd eth_trading_strategy
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
python backtest.py
```
- 若已安装 `jesse`，脚本会优先调用 Python API；否则会提示使用 `jesse backtest` CLI。
- `config.py` 中的 `start_date` 与 `finish_date` 控制回测区间，可按需修改。

### 2. 使用 Docker
```bash
cd eth_trading_strategy
docker compose up --build
```
构建完成后容器会自动执行 `python backtest.py`。如需交互式调试，可在命令中追加
`--entrypoint /bin/bash`。

## 策略文件说明

`strategies/ETHShortTermStrategy.py` 将 README 中的交易规则映射为 Jesse
策略类，实现要点包括：

- 组合 EMA、RSI、MACD、布林带、ATR 与成交量过滤信号。
- 每笔交易分配 2% 资金，并在多空两个方向设置对称的止损/止盈。
- 通过 `update_position` 持续调整追踪止损，以保护已实现利润。
- 使用最少 55 根 K 线作为热身，避免在指标未收敛时过早交易。
- 维护信号冷却机制，避免高频重复进场。

## 回测脚本

`backtest.py` 支持两种执行方式：

1. **Python API**：检测到 `jesse.research` 模块后直接调用 `research.backtest` 并输出
   摘要结果。
2. **CLI**：若 API 不可用则尝试调用 `jesse backtest <start> <finish>` 命令。

脚本会自动设置 `JESSE_PROJECT` 环境变量，确保 Jesse 在当前目录加载配置和策略。

## 下一步

- 调整 `config.py` 以适配真实数据源或实时交易环境。
- 在 `STRATEGY_ROUTES` 中添加更多交易对，构建多资产组合。
- 结合 Jesse 的指标或自定义统计函数，完善绩效评估与可视化。

> ⚠️ 本项目示例仅用于教学与研究目的，不构成任何投资建议，请在真实交易前
> 充分评估风险。
